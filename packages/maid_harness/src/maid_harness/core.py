"""Harness supervisor: one task record and one worker for each heavy request.

A run ends in one of three ways (ARCHITECTURE §5.2):
  result   the sub-agent called submit_result      -> completed (the only way to complete)
  pause    it called ask_user / request_permission -> needs_input / needs_permission, resumed by
           answer() / grant() with the reply as the next message of the same agent session
  neither  it exited without submitting           -> failed, never completed
Old-style workers that just return a summary string still complete as before.
"""

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


ACTIVE = ("accepted", "running", "needs_input", "needs_permission", "waiting_confirmation")
RESUMABLE = ("completed", "failed", "interrupted")


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
            if task.status in ACTIVE:
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

    # ---- the user's side of a paused or finished task (called by the Maid) ----------
    async def answer(self, task_id: str, answer: str) -> TaskRecord:
        """Reply to ask_user; the same agent session continues with the answer."""
        task = self._get(task_id)
        if task.status != "needs_input":
            return task
        self._emit("task.answered", task, {"answer": answer})
        return self._resume(task, f"The user answered your question ({task.question}): {answer}", question=None)

    async def grant(self, task_id: str, approve: bool) -> TaskRecord:
        """Yes/no to request_permission. A grant applies to this task only."""
        task = self._get(task_id)
        if task.status != "needs_permission":
            return task
        perm = task.metadata.get("permission") or {}
        meta = dict(task.metadata)
        if approve:
            meta["grants"] = [*meta.get("grants", []), perm]
            message = f"The user granted your permission request: {perm.get('what')}. Continue."
        else:
            message = (f"The user denied your permission request: {perm.get('what')}. Continue without it, "
                       "or call submit_result explaining what you could not do.")
        self._emit("task.permission", task, {"approved": approve, **perm})
        task = self.registry.update(task_id, metadata=meta)
        return self._resume(task, message)

    async def follow_up(self, task_id: str, message: str) -> TaskRecord:
        """More instructions for a finished, failed or interrupted task: the same session continues."""
        task = self._get(task_id)
        if task.status not in RESUMABLE:
            return task
        task = self.registry.update(task_id, metadata={**task.metadata,
                                                       "follow_ups": [*task.metadata.get("follow_ups", []), message]})
        return self._resume(task, f"Follow-up from the user: {message}", summary=None, error=None)

    def _resume(self, task: TaskRecord, message: str, **changes: Any) -> TaskRecord:
        meta = {k: v for k, v in task.metadata.items() if k not in ("pause", "result")}
        meta["pending_message"] = message
        task = self.registry.update(task.task_id, status="accepted", metadata=meta, **changes)
        self._start(task)
        return task

    # ---- the sub-agent's side (called by the sub-agent MCP tools) ------------------------
    def sub_progress(self, task_id: str, percent: int | None, text: str) -> TaskRecord:
        task = self._get(task_id)
        progress = task.progress if percent is None else max(0, min(100, int(percent)))
        task = self.registry.update(task_id, progress=progress, progress_text=text[:200])
        self._emit("task.progress", task, {"progress": task.progress, "text": task.progress_text})
        return task

    def sub_pause(self, task_id: str, state: str, **detail: Any) -> TaskRecord:
        """ask_user / request_permission: takes effect when the agent ends its turn. state: needs_input | needs_permission"""
        task = self._get(task_id)
        return self.registry.update(task_id, metadata={**task.metadata, "pause": {**detail, "state": state}})

    def sub_result(self, task_id: str, user_summary: str, details: str = "", artifacts: list | None = None) -> TaskRecord:
        task = self._get(task_id)
        result = {"user_summary": user_summary.strip(), "details": details, "artifacts": list(artifacts or [])}
        return self.registry.update(task_id, metadata={**task.metadata, "result": result})

    def _get(self, task_id: str) -> TaskRecord:
        task = self.registry.get(task_id)
        if task is None:
            raise KeyError(f"unknown task {task_id}")
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
        meta = {k: v for k, v in current.metadata.items() if k != "pending_message"}
        result, pause = meta.get("result"), meta.get("pause")
        if result:
            current = self.registry.update(task_id, status="completed", progress=100, metadata=meta,
                                           summary=result["user_summary"] or "Done.")
            self._emit("task.completed", current, {"summary": current.summary}, priority="high")
        elif pause:
            state = pause["state"]
            if state == "needs_permission":
                meta["permission"] = {k: pause.get(k) for k in ("what", "why", "kind", "target")}
            current = self.registry.update(task_id, status=state, metadata=meta, question=pause.get("question"))
            self._emit(f"task.{state}", current, {k: v for k, v in pause.items() if k != "state"}, priority="high")
        elif summary is not None:
            current = self.registry.update(task_id, status="completed", progress=100, summary=str(summary), metadata=meta)
            self._emit("task.completed", current, {"summary": current.summary}, priority="high")
        else:
            current = self.registry.update(task_id, status="failed", metadata=meta,
                                           error="the sub-agent ended without submitting a result")
            self._emit("task.failed", current, {"error": current.error}, priority="high")

    def _emit(self, event_type: str, task: TaskRecord, payload: dict[str, Any], priority: str = "normal") -> None:
        self.inbox.publish(
            event_type,
            source="harness",
            task_id=task.task_id,
            payload=payload,
            priority=priority,
            dedupe_key=f"{task.task_id}:{event_type}:{task.progress if event_type == 'task.progress' else task.updated_at}",
        )
