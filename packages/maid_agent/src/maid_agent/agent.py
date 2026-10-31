"""`maid-agent serve`: the Maid inside the maid_voice gateway (ARCHITECTURE §6, one process).

  voice turn ──▶ reply LLM (+ delegate tools, T1 state block) ──▶ MaidTurns.run ──▶ Harness ──▶ codex exec
                                                                                     │ events      │ maid MCP
  ctl.notify ◀── SpeechArbiter (waits for a pause, merges) ◀── inbox ◀───────────────┴─────────────┘
  ctl.say    ◀── ScreenProactive (unchanged until P4)

Config comes from the [agent] section of maid_voice's config file.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from maid_harness import Capability, Harness
from maid_harness.codex_runner import PROFILES, CodexRunner, Profile
from maid_harness.models import TaskEvent, TaskRecord
from maid_harness.subagent_mcp import SubAgentTools
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
    runner: str = "codex"                      # codex: real sub-agents for quick/work (P3) | fake: the P1 stand-in
    codex: str = "codex"                       # the Codex CLI
    tasks_dir: str = "runs/tasks"              # one workspace per task: events.jsonl, stderr.log, its files
    quick_effort: str = "low"
    quick_timeout_s: float = 300.0
    work_effort: str = "medium"
    work_timeout_s: float = 1800.0
    fake_quick_s: float = 20.0                 # how long the fake sub-agent takes per tier
    fake_work_s: float = 45.0
    fake_computer_s: float = 15.0
    quiet_gap_s: float = 1.5
    heartbeat_s: float = 180.0

    @classmethod
    def from_raw(cls, raw: dict) -> "AgentConfig":
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})


def task_view(t: TaskRecord) -> dict:
    return {"task_id": t.task_id, "goal": t.goal, "tier": t.metadata.get("tier"),
            "status": t.status, "progress": t.progress, "progress_text": t.progress_text,
            "summary": t.summary, "error": t.error, "created_at": t.created_at}


class MaidAgent:
    def __init__(self, root: Path, cfg: AgentConfig, screen=None) -> None:
        self.root = root
        self.cfg = cfg
        self.screen = screen
        self.harness = Harness(root / cfg.db, worker_factory=self._worker)
        self.tools: SubAgentTools | None = None
        self.profiles = {
            "quick": Profile(**{**PROFILES["quick"].__dict__, "effort": cfg.quick_effort, "timeout_s": cfg.quick_timeout_s}),
            "work": Profile(**{**PROFILES["work"].__dict__, "effort": cfg.work_effort, "timeout_s": cfg.work_timeout_s}),
        }
        self.arbiter = SpeechArbiter(self.harness, ArbiterConfig(cfg.quiet_gap_s, cfg.heartbeat_s),
                                     log_dir=root / "runs" / "agent")
        self.turns = MaidTurns(self.harness, cfg.routing, log_dir=root / "runs" / "agent",
                               look=self._look_when_quiet if screen else None, followup=self.arbiter.answer,
                               facts=screen.facts if screen else None)

    def _worker(self, task: TaskRecord):
        tier = task.metadata.get("tier", "quick")
        if self.tools is not None and tier in self.profiles:
            return CodexRunner(self.harness, self.tools, self.root / self.cfg.tasks_dir, self.profiles[tier], self.cfg.codex)
        durations = {"quick": self.cfg.fake_quick_s, "work": self.cfg.fake_work_s, "computer": self.cfg.fake_computer_s}
        return FakeSubAgent(durations.get(tier, self.cfg.fake_quick_s))

    def _register_runners(self) -> None:
        """What she can hand off, per tier. computer stays simulated until P6."""
        caps = self.harness.capabilities
        if self.tools is None:
            return  # the harness default: one simulated sub_agents capability
        caps.remove("sub_agents")
        caps.register(Capability(
            "research", "live", "hands questions to a background sub-agent with live web search: weather, news, "
            "prices, docs, coding questions (tier quick, read-only, usually under a minute or two)",
            ("delegate", "answer_task", "follow_up_task"), "it may ask the user a question back"))
        caps.register(Capability(
            "work", "live", "hands real work to a background sub-agent: write code or files, run commands and tests "
            "(tier work), inside its own task folder; anything outside needs the user's permission",
            ("delegate", "answer_task", "follow_up_task"), "it asks for permission by voice before touching anything else"))
        caps.register(Capability(
            "computer_use", "simulated", "operating apps and the GUI (tier computer)", ("delegate",),
            "practice mode: nothing is really operated until P6"))

    # ---- extension API (called by maid_voice.app.serve) ----------------
    async def start(self, ctx) -> list[asyncio.Task]:
        self.ctx = ctx
        if self.cfg.runner == "codex":
            self.tools = SubAgentTools(self.harness)
            await self.tools.start()
            self._register_runners()
        self.conversation = Conversation(ConversationLog(self.root / self.cfg.memory_db), ctx.ctl.session_id)
        self.turns.conversation = self.conversation
        ctx.ctl.history = self.conversation.restore(min(self.cfg.restore_turns, 2 * ctx.ctl.history_turns))
        ctx.ctl.hooks = self.turns
        self.harness.inbox.subscribe(self._on_event)
        interrupted = self.harness.recover()  # after the arbiter subscribed, so she mentions them once
        ctx.info.update(agent=self._info())
        runner = f"codex sub-agents (tools on :{self.tools.port})" if self.tools else "fake sub-agent"
        print(f"maid agent: routing={self.cfg.routing}, {runner}, db {self.root / self.cfg.db}, "
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
