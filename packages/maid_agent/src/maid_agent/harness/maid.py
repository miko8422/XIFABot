"""Thin Maid-facing layer over the harness.

It owns routing and concise reports. It does not execute tools or inspect
worker internals.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import Harness
from .models import RouteDecision, TaskEvent, TaskRecord
from .router import route_text


@dataclass(frozen=True)
class MaidReply:
    text: str
    route: RouteDecision
    task_id: str | None = None
    needs_confirmation: bool = False


@dataclass(frozen=True)
class MaidNotification:
    text: str
    interrupt: bool
    task_ids: tuple[str, ...] = ()
    event_types: tuple[str, ...] = ()


class MaidRuntime:
    def __init__(self, harness: Harness, owner_id: str = "default") -> None:
        self.harness = harness
        self.owner_id = owner_id
        self._pending: list[TaskEvent] = []
        self.harness.inbox.subscribe(self._on_event)

    async def handle_text(self, text: str) -> MaidReply:
        decision = route_text(text)
        if decision.route == "direct":
            return MaidReply("ん。ワタシは聞いてる。", decision)
        if decision.route == "clarify":
            return MaidReply("もう少しだけ、何をすればいいか教えて。", decision)
        if decision.route == "reporter_query":
            return MaidReply("確認してみる。", decision)
        task = await self.harness.submit(
            text,
            route=decision.route,
            owner_id=self.owner_id,
            worker_kind="computer-use" if decision.route == "computer_use" else "demo",
            requires_confirmation=decision.route == "computer_use",
        )
        if task.status == "waiting_confirmation":
            return MaidReply("画面操作が必要だから、確認してから始めるね。", decision, task.task_id, True)
        return MaidReply("わかった。始めるね。終わったら知らせる。", decision, task.task_id)

    async def confirm(self, task_id: str) -> TaskRecord:
        return await self.harness.confirm(task_id)

    def poll_notification(self) -> MaidNotification | None:
        if not self._pending:
            return None
        events = self._pending
        self._pending = []
        grouped: dict[str, list[TaskEvent]] = {}
        for event in events:
            grouped.setdefault(event.task_id or "system", []).append(event)

        parts: list[str] = []
        ids: list[str] = []
        latest_items = [(task_id, task_events[-1]) for task_id, task_events in grouped.items()]
        started_count = sum(1 for _, event in latest_items if event.type == "task.started")
        progress_count = sum(1 for _, event in latest_items if event.type == "task.progress")
        if started_count > 1:
            parts.append(f"{started_count}件の作業を始めたよ。")
        if progress_count > 1:
            parts.append("複数の作業が進行中。")
        for task_id, task_events in grouped.items():
            latest = task_events[-1]
            if task_id != "system":
                ids.append(task_id)
            if latest.type == "task.completed":
                parts.append(f"終わった：{latest.payload.get('summary', '作業は完了したよ')}")
            elif latest.type == "task.failed":
                parts.append(f"失敗した：{latest.payload.get('error', '作業に失敗した')}")
            elif latest.type == "task.cancelled":
                parts.append("キャンセルしたよ。")
            elif latest.type == "task.needs_input":
                parts.append(f"確認が必要：{latest.payload.get('question', '答えてほしいことがある')}")
            elif latest.type == "task.progress":
                if progress_count == 1:
                    parts.append(f"まだ進行中：{latest.payload.get('text', '作業中')}、{latest.payload.get('progress', 0)}パーセント。")
            elif latest.type == "system.changed" or latest.type == "reporter.warning":
                parts.append(str(latest.payload.get("text", "新しい変化がある。")))
            elif latest.type == "task.started":
                if started_count == 1:
                    parts.append("始めたよ。")

        if not parts:
            return None
        return MaidNotification(
            " ".join(parts),
            interrupt=True,
            task_ids=tuple(ids),
            event_types=tuple(event.type for event in events),
        )

    def _on_event(self, event: TaskEvent) -> None:
        self._pending.append(event)
