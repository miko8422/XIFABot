"""Worker protocol and a deterministic demo worker.

The demo worker is intentionally small. It proves the lifecycle and MCP
boundary without pretending that Codex, browsing, or computer use already
exists in this repository.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .models import TaskRecord

Progress = Callable[[int, str], Awaitable[None]]


class Worker:
    async def run(self, task: TaskRecord, report: Progress) -> str:
        raise NotImplementedError


@dataclass
class DemoWorker(Worker):
    delay_s: float = 0.08

    async def run(self, task: TaskRecord, report: Progress) -> str:
        steps = (
            (20, "範囲を整理中"),
            (50, "作業を進めてる"),
            (80, "結果を確認中"),
            (100, "仕上げ中"),
        )
        for progress, text in steps:
            await asyncio.sleep(self.delay_s)
            await report(progress, text)
        return f"完了：{task.request[:80]}"


@dataclass
class FakeSubAgent(Worker):
    """Stands in for the Codex runner until P3: takes about `duration_s`, reports English progress,
    and says plainly in its summary that nothing real was done."""
    duration_s: float = 30.0

    async def run(self, task: TaskRecord, report: Progress) -> str:
        steps = ((10, "planning"), (35, "gathering information"), (65, "working on it"), (90, "checking the result"))
        for progress, text in steps:
            await asyncio.sleep(self.duration_s / len(steps))
            await report(progress, text)
        return "Simulated sub-agent (P1 stand-in): no real work was done."
