"""`maid-agent serve`: the Maid inside the maid_voice gateway (ARCHITECTURE §6, one process).

  voice turn ──▶ reply LLM (+ delegate tools, T1 state block) ──▶ MaidTurns.run ──▶ Harness
                                                                                     │ events
  ctl.notify ◀── SpeechArbiter (waits for a pause, merges) ◀── inbox ◀───────────────┘
  ctl.say    ◀── ScreenProactive (unchanged until P4)

Config comes from the [agent] section of maid_voice's config file.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from maid_harness import Capability, Harness
from maid_harness.models import TaskEvent, TaskRecord
from maid_harness.workers import FakeSubAgent

from .arbiter import ArbiterConfig, SpeechArbiter
from .conversation import Conversation, ConversationLog
from .turns import MaidTurns


@dataclass
class AgentConfig:
    routing: str = "tool"                      # tool: delegate inside the reply call | keyword: router.py decides
    db: str = "runs/harness/tasks.sqlite3"     # relative to the maid_agent package
    memory_db: str = "runs/maid/memory.sqlite3"  # the Maid's own file: conversation log (P2), memory bank (P5)
    restore_turns: int = 8                     # messages put back into her history on startup
    fake_quick_s: float = 20.0                 # how long the fake sub-agent takes per tier (P1 only)
    fake_work_s: float = 45.0
    fake_computer_s: float = 15.0
    quiet_gap_s: float = 1.5
    heartbeat_s: float = 180.0

    @classmethod
    def from_raw(cls, raw: dict) -> "AgentConfig":
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})


def task_view(t: TaskRecord) -> dict:
    return {"task_id": t.task_id, "goal": t.metadata.get("goal") or t.request, "tier": t.metadata.get("tier"),
            "status": t.status, "progress": t.progress, "progress_text": t.progress_text,
            "summary": t.summary, "error": t.error, "created_at": t.created_at}


class MaidAgent:
    def __init__(self, root: Path, cfg: AgentConfig, screen=None) -> None:
        self.root = root
        self.cfg = cfg
        self.screen = screen
        durations = {"quick": cfg.fake_quick_s, "work": cfg.fake_work_s, "computer": cfg.fake_computer_s}
        self.harness = Harness(root / cfg.db, worker_factory=lambda task: FakeSubAgent(
            durations.get(task.metadata.get("tier"), cfg.fake_quick_s)))
        self.arbiter = SpeechArbiter(self.harness, ArbiterConfig(cfg.quiet_gap_s, cfg.heartbeat_s),
                                     log_dir=root / "runs" / "agent")
        self.turns = MaidTurns(self.harness, cfg.routing, log_dir=root / "runs" / "agent",
                               look=self._look_when_quiet if screen else None, followup=self.arbiter.answer,
                               facts=screen.facts if screen else None)

    # ---- extension API (called by maid_voice.app.serve) ----------------
    async def start(self, ctx) -> list[asyncio.Task]:
        self.ctx = ctx
        self.conversation = Conversation(ConversationLog(self.root / self.cfg.memory_db), ctx.ctl.session_id)
        self.turns.conversation = self.conversation
        ctx.ctl.history = self.conversation.restore(min(self.cfg.restore_turns, 2 * ctx.ctl.history_turns))
        ctx.ctl.hooks = self.turns
        self.harness.inbox.subscribe(self._on_event)
        interrupted = self.harness.recover()  # after the arbiter subscribed, so she mentions them once
        ctx.info.update(agent=self._info())
        print(f"maid agent: routing={self.cfg.routing}, fake sub-agent, db {self.root / self.cfg.db}, "
              f"{len(ctx.ctl.history)} message(s) of conversation restored"
              + (f", {len(interrupted)} task(s) interrupted by the last restart" if interrupted else ""), flush=True)
        tasks = [asyncio.create_task(self.arbiter.run(ctx))]
        if self.screen is not None:
            tasks += await self.screen.start(ctx)
            self.harness.capabilities.register(Capability(
                "screen", "live", "can look at the user's main screen when asked (local vision, a few seconds), "
                "and notices big changes on its own", ("look_at_screen",), "the main display only"))
        caps = self.harness.capabilities.usable()
        print("capabilities: " + ", ".join(f"{c.name}={c.status}" for c in caps), flush=True)
        return tasks

    async def command(self, cmd: dict) -> bool:
        c = cmd.get("cmd")
        if c == "agent_mute":
            self.arbiter.cfg.muted = bool(cmd.get("on"))
        elif c == "task_cancel" and cmd.get("task_id"):
            await self.harness.cancel(cmd["task_id"])
        elif c == "tasks":  # a page (re)connected: send it the current tasks and the restored conversation
            await self._broadcast_tasks()
            await self.ctx.broadcast({"type": "history", "turns": [
                {"role": t.role, "content": t.content} for t in self.ctx.ctl.history]})
            return True
        elif self.screen is not None:
            return await self.screen.command(cmd)
        else:
            return False
        self.ctx.info.update(agent=self._info())
        await self.ctx.broadcast(self.ctx.info)
        return True

    async def _look_when_quiet(self, question: str) -> str:
        """Vision and TTS share oMLX and the GPU: looking while she still says 「ちょっと見る」 made that line
        stutter (2.7 s for a 1 s line, 2026-10-31). So wait until her acknowledgement has played."""
        ctl = self.ctx.ctl
        for _ in range(int(self.LOOK_WAIT_S / 0.05)):
            if not ctl.active_gen and ctl.state == "listening":
                break
            await asyncio.sleep(0.05)
        return await self.screen.look(question)

    LOOK_WAIT_S = 5.0

    # ---- UI -------------------------------------------------------------
    def _on_event(self, event: TaskEvent) -> None:
        asyncio.ensure_future(self.ctx.broadcast({"type": "task.event", "event": event.type,
                                                  "task_id": event.task_id, "payload": event.payload}))
        asyncio.ensure_future(self._broadcast_tasks())

    async def _broadcast_tasks(self) -> None:
        tasks = sorted(self.harness.registry.list(), key=lambda t: -t.created_at)[:12]
        await self.ctx.broadcast({"type": "tasks", "tasks": [task_view(t) for t in tasks]})

    def _info(self) -> dict:
        return {"routing": self.cfg.routing, "muted": self.arbiter.cfg.muted,
                "capabilities": [{"name": c.name, "status": c.status, "note": c.note}
                                 for c in self.harness.capabilities.all()]}
