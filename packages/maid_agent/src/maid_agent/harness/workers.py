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
