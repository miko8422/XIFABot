"""Harness supervisor: one task record and one worker for each heavy request."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .capabilities import Capability, CapabilityRegistry
from .inbox import EventInbox
from .models import Route, TaskEvent, TaskRecord
from .registry import TaskRegistry
from .workers import DemoWorker, Worker


class Harness:
    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        worker_factory: Callable[[TaskRecord], Worker] | None = None,
        worker_delay_s: float = 0.08,
    ) -> None:
        self.registry = TaskRegistry(db_path)
        self.inbox = EventInbox(self.registry)
        self.worker_factory = worker_factory or (lambda task: DemoWorker(worker_delay_s))
        self._running: dict[str, asyncio.Task[None]] = {}
        self.capabilities = CapabilityRegistry()
        self.capabilities.register(Capability(
            "tasks", "live", "knows every task's state, progress and result; can cancel a task",
            ("cancel_task", "confirm_task")))
        # whoever installs a real runner sets this live (P3); the default workers do no real work
        self.capabilities.register(Capability(
            "sub_agents", "simulated",
            "hands heavy work to a background sub-agent: web search and research (weather, news, docs), "
            "coding questions, changing files, running commands, operating apps",
            ("delegate",), "practice mode: no real work is done and results are not real until P3"))

    def recover(self) -> list[TaskRecord]:
        """On startup: tasks left active by the previous process are marked `interrupted` (§6)."""
        out = []
        for task in self.registry.list():
            if task.status in {"accepted", "running", "needs_input", "waiting_confirmation"}:
                task = self.registry.update(task.task_id, status="interrupted", error="the Maid was restarted")
                self._emit("task.interrupted", task, {"reason": task.error})
                out.append(task)
        return out

    async def submit(
        self,
        request: str,
        route: Route = "task",
        owner_id: str = "default",
        worker_kind: str = "demo",
        *,
        requires_confirmation: bool = False,
        confirmed: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> TaskRecord:
        task = self.registry.create_task(
            request,
            route,
            owner_id,
            worker_kind,
            requires_confirmation,
            confirmed,
            metadata,
        )
        self._emit("task.accepted", task, {"status": task.status})
        if task.status == "waiting_confirmation":
            return task
        self._start(task)
        return task

    async def confirm(self, task_id: str) -> TaskRecord:
        task = self.registry.get(task_id)
        if task is None:
            raise KeyError(f"unknown task {task_id}")
        if task.status != "waiting_confirmation":
            return task
        task = self.registry.update(task_id, status="accepted", confirmed=True)
        self._emit("task.accepted", task, {"status": task.status, "confirmed": True})
        self._start(task)
        return task

    async def cancel(self, task_id: str, reason: str = "cancelled by user") -> TaskRecord:
        task = self.registry.get(task_id)
        if task is None:
            raise KeyError(f"unknown task {task_id}")
        running = self._running.get(task_id)
        if running and not running.done():
            running.cancel()
        if task.status not in {"completed", "failed", "cancelled", "interrupted"}:
            task = self.registry.update(task_id, status="cancelled", error=reason)
            self._emit("task.cancelled", task, {"reason": reason}, priority="high")
        return task

    def publish_event(self, event_type: str, source: str, payload: dict[str, Any] | None = None, **kwargs: Any) -> TaskEvent:
        return self.inbox.publish(event_type, source, payload=payload, **kwargs)

    async def wait(self, task_id: str, timeout: float = 5.0) -> TaskRecord:
        running = self._running.get(task_id)
        if running:
            try:
                await asyncio.wait_for(asyncio.shield(running), timeout)
            except asyncio.TimeoutError:
                pass
        task = self.registry.get(task_id)
        if task is None:
            raise KeyError(f"unknown task {task_id}")
        return task

    def _start(self, task: TaskRecord) -> None:
        if task.task_id in self._running and not self._running[task.task_id].done():
            return
        self._running[task.task_id] = asyncio.create_task(self._run(task.task_id), name=f"maid-task-{task.task_id}")

    async def _run(self, task_id: str) -> None:
        task = self.registry.get(task_id)
        if task is None:
            return
        task = self.registry.update(task_id, status="running")
        self._emit("task.started", task, {"worker_kind": task.worker_kind})

        async def report(progress: int, text: str) -> None:
            current = self.registry.update(task_id, progress=max(0, min(100, int(progress))), progress_text=text)
            self._emit("task.progress", current, {"progress": current.progress, "text": text})

        try:
            summary = await self.worker_factory(task).run(task, report)
        except asyncio.CancelledError:
            # cancel() already writes the user-facing state; do not emit a fake failure.
            return
        except Exception as exc:  # noqa: BLE001 - worker failures belong in the task record
            current = self.registry.update(task_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            self._emit("task.failed", current, {"error": current.error}, priority="high")
            return
        current = self.registry.get(task_id)
        if current is None or current.status == "cancelled":
            return
        current = self.registry.update(task_id, status="completed", progress=100, summary=str(summary))
        self._emit("task.completed", current, {"summary": current.summary}, priority="high")

    def _emit(self, event_type: str, task: TaskRecord, payload: dict[str, Any], priority: str = "normal") -> None:
        self.inbox.publish(
            event_type,
            source="harness",
            task_id=task.task_id,
            payload=payload,
            priority=priority,
            dedupe_key=f"{task.task_id}:{event_type}:{task.progress if event_type == 'task.progress' else task.updated_at}",
        )
