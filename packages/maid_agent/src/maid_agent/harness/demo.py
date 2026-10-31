"""Scripted v0.1 demo used by the CLI and smoke tests."""

from __future__ import annotations

import asyncio

from .core import Harness
from .maid import MaidRuntime


async def run_demo() -> None:
    harness = Harness(worker_delay_s=0.05)
    maid = MaidRuntime(harness)
    print("Maid Agent v0.1 demo")
    print("transport: MCP stdio | worker: demo | computer-use confirmation: enabled")

    reply = await maid.handle_text("ちょっと話そう")
    print(f"user: ちょっと話そう\nmaid: {reply.text}")

    reply = await maid.handle_text("このプロジェクトを検索して")
    print(f"user: このプロジェクトを検索して\nmaid: {reply.text} [task={reply.task_id}]")

    reply = await maid.handle_text("Steamを開いて")
    print(f"user: Steamを開いて\nmaid: {reply.text} [task={reply.task_id}]")
    if reply.task_id:
        await maid.confirm(reply.task_id)
        print("user: 確認した")

    while True:
        await asyncio.sleep(0.03)
        while notification := maid.poll_notification():
            print(f"maid event: {notification.text} [interrupt={notification.interrupt}]")
        active = [task for task in harness.registry.list() if task.status in {"accepted", "running", "waiting_confirmation"}]
        if not active:
            break
    print("demo complete")
