"""Durable v0.1 task and event registry.

The registry is deliberately independent from Botmux, Feishu, or any worker
implementation. It is the operational source of truth for the harness.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from .models import Route, TaskEvent, TaskRecord, TaskStatus


def _now() -> float:
    import time

    return time.time()


class TaskRegistry:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                request TEXT NOT NULL,
                route TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                worker_kind TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                progress_text TEXT NOT NULL DEFAULT '',
                summary TEXT,
                error TEXT,
                question TEXT,
                requires_confirmation INTEGER NOT NULL DEFAULT 0,
                confirmed INTEGER NOT NULL DEFAULT 0,
                merged_into TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS tasks_status_idx ON tasks(status, updated_at);
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                source TEXT NOT NULL,
                task_id TEXT,
                occurred_at REAL NOT NULL,
                priority TEXT NOT NULL,
                confidence REAL NOT NULL,
                sensitivity TEXT NOT NULL,
                dedupe_key TEXT,
                ttl_seconds REAL NOT NULL,
                payload_json TEXT NOT NULL,
                schema_version INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS events_time_idx ON events(occurred_at);
            CREATE INDEX IF NOT EXISTS events_task_idx ON events(task_id, occurred_at);
            """
        )
        self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def create_task(
        self,
        request: str,
        route: Route,
        owner_id: str = "default",
        worker_kind: str = "demo",
        requires_confirmation: bool = False,
        confirmed: bool = False,
        metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> TaskRecord:
        created = _now() if now is None else now
        task = TaskRecord(
            task_id=uuid.uuid4().hex[:12],
            request=request.strip(),
            route=route,
            owner_id=owner_id,
            worker_kind=worker_kind,
            status="accepted" if (not requires_confirmation or confirmed) else "waiting_confirmation",
            created_at=created,
            updated_at=created,
            requires_confirmation=requires_confirmation,
            confirmed=confirmed,
            metadata=metadata or {},
        )
        self._write_task(task)
        return task

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return self._row_task(row) if row else None

    def list(self, status: TaskStatus | None = None, owner_id: str | None = None) -> list[TaskRecord]:
        clauses: list[str] = []
        args: list[Any] = []
        if status:
            clauses.append("status = ?")
            args.append(status)
        if owner_id:
            clauses.append("owner_id = ?")
            args.append(owner_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._db.execute(f"SELECT * FROM tasks{where} ORDER BY created_at DESC", args).fetchall()
        return [self._row_task(row) for row in rows]

    def update(self, task_id: str, now: float | None = None, **changes: Any) -> TaskRecord:
        task = self.get(task_id)
        if task is None:
            raise KeyError(f"unknown task {task_id}")
        for key, value in changes.items():
            if not hasattr(task, key):
                raise ValueError(f"unknown task field {key}")
            setattr(task, key, value)
        task.updated_at = _now() if now is None else now
        self._write_task(task)
        return task

    def add_event(self, event: TaskEvent) -> TaskEvent:
        with self._lock:
            if event.dedupe_key:
                row = self._db.execute(
                    "SELECT * FROM events WHERE dedupe_key = ? ORDER BY occurred_at DESC LIMIT 1",
                    (event.dedupe_key,),
                ).fetchone()
                if row:
                    return self._row_event(row)
            self._db.execute(
                """INSERT INTO events
                (event_id, type, source, task_id, occurred_at, priority, confidence,
                 sensitivity, dedupe_key, ttl_seconds, payload_json, schema_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id,
                    event.type,
                    event.source,
                    event.task_id,
                    event.occurred_at,
                    event.priority,
                    event.confidence,
                    event.sensitivity,
                    event.dedupe_key,
                    event.ttl_seconds,
                    json.dumps(event.payload, ensure_ascii=False),
                    event.schema_version,
                ),
            )
            self._db.commit()
        return event

    def events(self, task_id: str | None = None, limit: int = 100) -> list[TaskEvent]:
        if task_id:
            sql = "SELECT * FROM events WHERE task_id = ? ORDER BY occurred_at ASC LIMIT ?"
            args: tuple[Any, ...] = (task_id, limit)
        else:
            sql = "SELECT * FROM events ORDER BY occurred_at ASC LIMIT ?"
            args = (limit,)
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [self._row_event(row) for row in rows]

    def _write_task(self, task: TaskRecord) -> None:
        with self._lock:
            self._db.execute(
                """INSERT INTO tasks
                (task_id, request, route, owner_id, worker_kind, status, created_at,
                 updated_at, progress, progress_text, summary, error, question,
                 requires_confirmation, confirmed, merged_into, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                  request=excluded.request, route=excluded.route, owner_id=excluded.owner_id,
                  worker_kind=excluded.worker_kind, status=excluded.status,
                  created_at=excluded.created_at, updated_at=excluded.updated_at,
                  progress=excluded.progress, progress_text=excluded.progress_text,
                  summary=excluded.summary, error=excluded.error, question=excluded.question,
                  requires_confirmation=excluded.requires_confirmation, confirmed=excluded.confirmed,
                  merged_into=excluded.merged_into, metadata_json=excluded.metadata_json""",
                (
                    task.task_id,
                    task.request,
                    task.route,
                    task.owner_id,
                    task.worker_kind,
                    task.status,
                    task.created_at,
                    task.updated_at,
                    task.progress,
                    task.progress_text,
                    task.summary,
                    task.error,
                    task.question,
                    int(task.requires_confirmation),
                    int(task.confirmed),
                    task.merged_into,
                    json.dumps(task.metadata, ensure_ascii=False),
                ),
            )
            self._db.commit()

    @staticmethod
    def _row_task(row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            task_id=row["task_id"],
            request=row["request"],
            route=row["route"],
            owner_id=row["owner_id"],
            worker_kind=row["worker_kind"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            progress=row["progress"],
            progress_text=row["progress_text"],
            summary=row["summary"],
            error=row["error"],
            question=row["question"],
            requires_confirmation=bool(row["requires_confirmation"]),
            confirmed=bool(row["confirmed"]),
            merged_into=row["merged_into"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    @staticmethod
    def _row_event(row: sqlite3.Row) -> TaskEvent:
        return TaskEvent(
            event_id=row["event_id"],
            type=row["type"],
            source=row["source"],
            occurred_at=row["occurred_at"],
            task_id=row["task_id"],
            priority=row["priority"],
            confidence=row["confidence"],
            sensitivity=row["sensitivity"],
            dedupe_key=row["dedupe_key"],
            ttl_seconds=row["ttl_seconds"],
            payload=json.loads(row["payload_json"] or "{}"),
            schema_version=row["schema_version"],
        )
