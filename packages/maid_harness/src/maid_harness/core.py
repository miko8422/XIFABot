"""Harness supervisor: one task record and one worker for each heavy request.

A run ends in one of three ways (ARCHITECTURE §5.2), decided by `settle()`:
  result   the sub-agent called submit_result      -> completed (the only way to complete)
  pause    it called ask_user / request_permission -> needs_input / needs_permission, resumed by
           answer() / grant() with the reply as the next message of the same agent session
  neither  it exited without submitting           -> failed, never completed
Old-style workers that just return a summary string still complete as before.

Two kinds of workers:
  in-process  `Worker.run()` as an asyncio task; dies with the Maid (fakes, tests, computer-use until P6)
  detached    `DetachedWorker.spawn()` starts a supervisor process that outlives the Maid (Codex, P3.5).
              It settles the run itself, in the shared SQLite file; `watch()` here polls the events it
              writes, delivers them to the inbox, keeps at most `max_running` going, and notices a
              supervisor that died. `recover()` reconciles after a restart.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .capabilities import Capability, CapabilityRegistry
from .inbox import EventInbox
from .models import Route, TaskEvent, TaskRecord
from .registry import TaskRegistry
from .workers import DemoWorker, DetachedWorker, Worker

ACTIVE = ("accepted", "running", "needs_input", "needs_permission", "waiting_confirmation")
RESUMABLE = ("completed", "failed", "interrupted", "cancelled")
LIFECYCLE_TTL_S = 3 * 86400.0  # results that arrive while the Maid is down are still told when she is back
HEARTBEAT_STALE_S = 60.0


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Harness:
    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        worker_factory: Callable[[TaskRecord], Worker | DetachedWorker] | None = None,
        worker_delay_s: float = 0.08,
        event_source: str = "harness",
        max_running: int = 2,
    ) -> None:
        self.registry = TaskRegistry(db_path)
        self.inbox = EventInbox(self.registry)
        self.worker_factory = worker_factory or (lambda task: DemoWorker(worker_delay_s))
        self.event_source = event_source
        self.max_running = max_running
        self._running: dict[str, asyncio.Task[None]] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._cursor: int | None = None
        self.capabilities = CapabilityRegistry()
        self.capabilities.register(Capability(
            "tasks", "live", "knows every task's state, progress and full result; can cancel a task; finds older "
            "tasks by search", ("cancel_task", "confirm_task", "read_task", "search_tasks")))
        # whoever installs a real runner replaces this; the default workers do no real work
        self.capabilities.register(Capability(
            "sub_agents", "simulated",
            "hands heavy work to a background sub-agent: web search and research (weather, news, docs), "
            "coding questions, changing files, running commands, operating apps",
            ("delegate",), "practice mode: no real work is done and results are not real"))

    # ---- startup -----------------------------------------------------------------
    def recover(self) -> list[TaskRecord]:
        """After a restart (§6, P3.5): detached sub-agents that are still alive keep running and are watched;
        dead ones and in-process ones are `interrupted`; questions and permission requests survive and are
        asked again once. Returns the tasks that were interrupted."""
        out = []
        for task in self.registry.list():
            if task.status not in ACTIVE:
                continue
            m = task.metadata
            if task.status in ("needs_input", "needs_permission", "waiting_confirmation"):
                task = self.registry.update(task.task_id)  # touch: asked again now (dedupe key, yes/no window)
                self._emit(f"task.{task.status}" if task.status != "waiting_confirmation" else "task.accepted", task,
                           {**(m.get("pause") or m.get("permission") or {}), "question": task.question, "again": True},
                           priority="high")
                continue
            if m.get("detached") and task.status == "running" and pid_alive(m.get("pid")) \
                    and time.time() - m.get("alive_at", 0) < HEARTBEAT_STALE_S:
                continue
            if m.get("detached") and task.status == "accepted":
                continue  # queued: watch() starts it
            task = self.registry.update(task.task_id, status="interrupted", error="the Maid was restarted")
            self._emit("task.interrupted", task, {"reason": task.error}, priority="high")
            out.append(task)
        return out

    # ---- the Maid's side ---------------------------------------------------------------
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
        task = self.registry.create_task(request, route, owner_id, worker_kind, requires_confirmation, confirmed,
                                         metadata)
        self._emit("task.accepted", task, {"status": task.status})
        if task.status == "waiting_confirmation":
            return task
        self._start(task)
        return self.registry.get(task.task_id)

    async def confirm(self, task_id: str) -> TaskRecord:
        task = self._get(task_id)
        if task.status != "waiting_confirmation":
            return task
        task = self.registry.update(task_id, status="accepted", confirmed=True)
        self._emit("task.accepted", task, {"status": task.status, "confirmed": True})
        self._start(task)
        return self.registry.get(task_id)

    async def cancel(self, task_id: str, reason: str = "cancelled by user") -> TaskRecord:
        task = self._get(task_id)
        running = self._running.get(task_id)
        if running and not running.done():
            running.cancel()
        if task.status not in ("completed", "failed", "cancelled", "interrupted"):
            task = self.registry.update(task_id, status="cancelled", error=reason)
            self._emit("task.cancelled", task, {"reason": reason}, priority="high")
        self._kill(task)
        self._dispatch()
        return task

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
        if approve:
            task = self.registry.update(task_id, meta={"grants": [*task.metadata.get("grants", []), perm]})
            message = f"The user granted your permission request: {perm.get('what')}. Continue."
        else:
            message = (f"The user denied your permission request: {perm.get('what')}. Continue without it, "
                       "or call submit_result explaining what you could not do.")
        self._emit("task.permission", task, {"approved": approve, **perm})
        return self._resume(task, message)

    async def follow_up(self, task_id: str, message: str) -> TaskRecord:
        """More instructions for a finished, failed, interrupted or cancelled task: the same session continues."""
        task = self._get(task_id)
        if task.status not in RESUMABLE:
            return task
        task = self.registry.update(task_id, meta={"follow_ups": [*task.metadata.get("follow_ups", []), message]})
        return self._resume(task, f"Follow-up from the user: {message}", summary=None, error=None)

    def _resume(self, task: TaskRecord, message: str, **changes: Any) -> TaskRecord:
        task = self.registry.update(task.task_id, status="accepted", meta={"pending_message": message},
                                    drop=("pause", "result"), **changes)
        self._start(task)
        return self.registry.get(task.task_id)

    # ---- the sub-agent's side (its MCP tools, possibly in another process) --------------
    def sub_progress(self, task_id: str, percent: int | None, text: str) -> TaskRecord:
        task = self._get(task_id)
        progress = task.progress if percent is None else max(0, min(100, int(percent)))
        task = self.registry.update(task_id, progress=progress, progress_text=text[:200])
        self._emit("task.progress", task, {"progress": task.progress, "text": task.progress_text})
        return task

    def sub_pause(self, task_id: str, state: str, **detail: Any) -> TaskRecord:
        """ask_user / request_permission: takes effect when the agent ends its turn. state: needs_input | needs_permission"""
        return self.registry.update(task_id, meta={"pause": {**detail, "state": state}})

    def sub_result(self, task_id: str, user_summary: str, details: str = "", artifacts: list | None = None) -> TaskRecord:
        result = {"user_summary": user_summary.strip(), "details": details, "artifacts": list(artifacts or [])}
        return self.registry.update(task_id, meta={"result": result})

    def settle(self, task_id: str, summary: str | None = None, error: str | None = None) -> TaskRecord | None:
        """The end of one run, wherever it ran: result > pause > error > plain summary > 'no result' failure."""
        current = self.registry.get(task_id)
        if current is None or current.status in ("cancelled", "completed", "failed", "interrupted"):
            return current
        meta = current.metadata
        result, pause = meta.get("result"), meta.get("pause")
        if result:
            current = self.registry.update(task_id, status="completed", progress=100, drop=("pending_message",),
                                           summary=result["user_summary"] or "Done.", error=None)
            self._emit("task.completed", current, {"summary": current.summary}, priority="high")
        elif pause:
            state = pause["state"]
            extra = {"permission": {k: pause.get(k) for k in ("what", "why", "kind", "target")}} \
                if state == "needs_permission" else {}
            current = self.registry.update(task_id, status=state, meta=extra, drop=("pending_message",),
                                           question=pause.get("question"))
            self._emit(f"task.{state}", current, {k: v for k, v in pause.items() if k != "state"}, priority="high")
        elif error is None and summary is not None:
            current = self.registry.update(task_id, status="completed", progress=100, summary=str(summary),
                                           drop=("pending_message",))
            self._emit("task.completed", current, {"summary": current.summary}, priority="high")
        else:
            current = self.registry.update(task_id, status="failed", drop=("pending_message",),
                                           error=error or "the sub-agent ended without submitting a result")
            self._emit("task.failed", current, {"error": current.error}, priority="high")
        return current

    # ---- running ----------------------------------------------------------------------------
    def _start(self, task: TaskRecord) -> None:
        if task.task_id in self._running and not self._running[task.task_id].done():
            return
        worker = self.worker_factory(task)
        if isinstance(worker, DetachedWorker):
            self._start_detached(task, worker)
            return
        self._running[task.task_id] = asyncio.create_task(self._run(task.task_id, worker),
                                                          name=f"maid-task-{task.task_id}")

    def _detached_running(self) -> int:
        return sum(1 for t in self.registry.list(status="running") if t.metadata.get("detached"))

    def _start_detached(self, task: TaskRecord, worker: DetachedWorker) -> None:
        if self._detached_running() >= self.max_running:
            if not task.metadata.get("queued"):
                task = self.registry.update(task.task_id, meta={"detached": True, "queued": True})
                self._emit("task.queued", task, {"running": self.max_running})
            return
        task = self.registry.update(task.task_id, status="running", drop=("queued",),
                                    meta={"detached": True, "alive_at": time.time()})
        self._emit("task.started", task, {"worker_kind": task.worker_kind})
        try:
            proc = worker.spawn(task)
        except Exception as exc:  # noqa: BLE001 - a failed launch is a failed run
            self.settle(task.task_id, error=f"could not start the sub-agent: {type(exc).__name__}: {exc}")
            return
        self._procs[task.task_id] = proc
        self.registry.update(task.task_id, meta={"pid": proc.pid})

    def _dispatch(self) -> None:
        """Start queued detached tasks, oldest first, while there is room."""
        queued = sorted((t for t in self.registry.list(status="accepted") if t.metadata.get("queued")),
                        key=lambda t: t.created_at)
        for task in queued:
            if self._detached_running() >= self.max_running:
                return
            self._start(task)

    def _kill(self, task: TaskRecord) -> None:
        pid = task.metadata.get("pid")
        if not task.metadata.get("detached") or not pid_alive(pid):
            return
        try:
            os.killpg(pid, signal.SIGTERM)  # the supervisor's group: it and its Codex
        except (ProcessLookupError, PermissionError):
            return
        try:
            asyncio.get_running_loop().call_later(3.0, self._kill_hard, pid)
        except RuntimeError:
            pass

    @staticmethod
    def _kill_hard(pid: int) -> None:
        if pid_alive(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    async def _run(self, task_id: str, worker: Worker) -> None:
        task = self.registry.get(task_id)
        if task is None:
            return
        task = self.registry.update(task_id, status="running")
        self._emit("task.started", task, {"worker_kind": task.worker_kind})

        async def report(progress: int, text: str) -> None:
            current = self.registry.update(task_id, progress=max(0, min(100, int(progress))), progress_text=text)
            self._emit("task.progress", current, {"progress": current.progress, "text": text})

        try:
            summary = await worker.run(task, report)
        except asyncio.CancelledError:
            return  # cancel() already wrote the user-facing state; no fake failure
        except Exception as exc:  # noqa: BLE001 - worker failures belong in the task record
            self.settle(task_id, error=f"{type(exc).__name__}: {exc}")
            return
        self.settle(task_id, summary=summary)

    # ---- watching detached sub-agents --------------------------------------------------------
    def pump(self) -> int:
        """Deliver events other processes wrote since the last call. The position is saved in the
        registry, so results that arrived while the Maid was down are delivered when she is back."""
        if self._cursor is None:
            saved = self.registry.get_cursor(self.event_source)
            self._cursor = saved if saved is not None else self.registry.last_event_rowid()
            self.registry.set_cursor(self.event_source, self._cursor)
        delivered = 0
        rows = self.registry.events_after(self._cursor)
        now = time.time()
        for rowid, event in rows:
            self._cursor = rowid
            if event.event_id in self.inbox.published or event.expired(now):
                continue
            self.inbox.deliver(event)
            delivered += 1
        if rows:
            self.registry.set_cursor(self.event_source, self._cursor)
        return delivered

    def check_supervisors(self) -> None:
        """A detached run whose supervisor is gone without settling becomes `interrupted` (resumable)."""
        for task_id, proc in list(self._procs.items()):
            if proc.poll() is not None:
                del self._procs[task_id]  # reaped; its outcome is in the registry
        for task in self.registry.list(status="running"):
            m = task.metadata
            if not m.get("detached") or task.task_id in self._procs:
                continue
            if not pid_alive(m.get("pid")) or time.time() - m.get("alive_at", 0) > HEARTBEAT_STALE_S:
                self.registry.update(task.task_id, status="interrupted", error="the sub-agent stopped unexpectedly")
                self._emit("task.interrupted", self._get(task.task_id), {"reason": "the sub-agent stopped unexpectedly"},
                           priority="high")

    async def watch(self, interval_s: float = 0.5) -> None:
        while True:
            self.pump()
            self.check_supervisors()
            self._dispatch()
            await asyncio.sleep(interval_s)

    async def wait(self, task_id: str, timeout: float = 5.0) -> TaskRecord:
        """Until the task leaves accepted/running (any kind of worker), or the timeout."""
        running = self._running.get(task_id)
        if running:
            try:
                await asyncio.wait_for(asyncio.shield(running), timeout)
            except asyncio.TimeoutError:
                pass
            return self._get(task_id)
        deadline = time.monotonic() + timeout
        while self._get(task_id).status in ("accepted", "running") and time.monotonic() < deadline:
            self.pump()
            await asyncio.sleep(0.05)
        return self._get(task_id)

    # ---- helpers --------------------------------------------------------------------------------
    def _get(self, task_id: str) -> TaskRecord:
        task = self.registry.get(task_id)
        if task is None:
            raise KeyError(f"unknown task {task_id}")
        return task

    def publish_event(self, event_type: str, source: str, payload: dict[str, Any] | None = None, **kwargs: Any) -> TaskEvent:
        return self.inbox.publish(event_type, source, payload=payload, **kwargs)

    def _emit(self, event_type: str, task: TaskRecord, payload: dict[str, Any], priority: str = "normal") -> None:
        self.inbox.publish(
            event_type,
            source=self.event_source,
            task_id=task.task_id,
            payload=payload,
            priority=priority,
            dedupe_key=f"{task.task_id}:{event_type}:{task.progress if event_type == 'task.progress' else task.updated_at}",
            ttl_seconds=3600.0 if event_type == "task.progress" else LIFECYCLE_TTL_S,
        )
