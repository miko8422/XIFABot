"""`maid-agent serve`: the Maid inside the maid_voice gateway (ARCHITECTURE §6, one process).

  voice turn ──▶ reply LLM (+ tools, T1 state block) ──▶ MaidTurns.run ──▶ Harness ──spawn──▶ maid_harness.subagent
                                                                              │ events        (detached) └ codex exec
  ctl.notify ◀── SpeechArbiter (waits for a pause, merges) ◀── inbox ◀── Harness.watch() ◀── tasks.sqlite3 ◀──┘
  ctl.say    ◀── ScreenProactive (unchanged until P4)

Config comes from the [agent] section of maid_voice's config file.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from maid_harness import Capability, Harness
from maid_harness.codex_runner import PROFILES, CodexRunner, Profile, prune_workspaces
from maid_harness.codex_home import build_codex_home
from maid_harness.models import TaskEvent, TaskRecord
from maid_harness.workers import FakeSubAgent

from .arbiter import ArbiterConfig, SpeechArbiter
from .conversation import Conversation, ConversationLog
from .tools import usage_since
from .turns import MaidTurns, midnight


@dataclass
class AgentConfig:
    routing: str = "tool"                      # tool: delegate inside the reply call | keyword: router.py decides
    db: str = "runs/harness/tasks.sqlite3"     # relative to the maid_agent package
    memory_db: str = "runs/maid/memory.sqlite3"  # the Maid's own file: conversation log (P2), memory bank (P5)
    restore_turns: int = 8                     # messages put back into her history on startup
    runner: str = "codex"                      # codex: real sub-agents for quick/work (P3) | fake: the P1 stand-in
    codex: str = "codex"                       # the Codex CLI (a command line; tests use a fake)
    codex_home: str = "runs/codex_home"        # the sub-agents' own clean CODEX_HOME, rebuilt from ~/.codex per run
    model: str = "gpt-5.6-luna"                # sub-agent model through the user's own Codex provider/account
    max_running: int = 2                       # sub-agents at once; the rest wait in line
    daily_token_budget: int = 0                # 0 = no limit; else no new sub-agent once today's tokens reach it
    retention_days: float = 7.0                # a finished task's files are deleted after this; its logs stay
    tasks_dir: str = "runs/tasks"              # one workspace per task: events.jsonl, stderr.log, its files
    quick_effort: str = "low"
    quick_timeout_s: float = 300.0
    work_effort: str = "low"
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
            "summary": t.summary, "error": t.error, "created_at": t.created_at, "codex": t.metadata.get("codex"),
            "tokens": sum(int(u.get("input_tokens", 0)) + int(u.get("output_tokens", 0)) for u in t.metadata.get("usage", []))}


class MaidAgent:
    def __init__(self, root: Path, cfg: AgentConfig, screen=None) -> None:
        self.root = root
        self.cfg = cfg
        self.screen = screen
        self.harness = Harness(root / cfg.db, worker_factory=self._worker, max_running=cfg.max_running)
        self.codex_on = cfg.runner == "codex"
        self.profiles = {
            "quick": Profile(**{**PROFILES["quick"].__dict__, "effort": cfg.quick_effort,
                                "timeout_s": cfg.quick_timeout_s, "model": cfg.model}),
            "work": Profile(**{**PROFILES["work"].__dict__, "effort": cfg.work_effort,
                               "timeout_s": cfg.work_timeout_s, "model": cfg.model}),
        }
        self.arbiter = SpeechArbiter(self.harness, ArbiterConfig(cfg.quiet_gap_s, cfg.heartbeat_s),
                                     log_dir=root / "runs" / "agent")
        self.turns = MaidTurns(self.harness, cfg.routing, log_dir=root / "runs" / "agent",
                               daily_token_budget=cfg.daily_token_budget,
                               look=self._look_when_quiet if screen else None, followup=self.arbiter.answer,
                               facts=screen.facts if screen else None)

    def _worker(self, task: TaskRecord):
        tier = task.metadata.get("tier", "quick")
        if self.codex_on and tier in self.profiles:
            return CodexRunner(self.root / self.cfg.db, self.root / self.cfg.tasks_dir, self.root / self.cfg.codex_home,
                               self.profiles[tier], self.cfg.codex)
        durations = {"quick": self.cfg.fake_quick_s, "work": self.cfg.fake_work_s, "computer": self.cfg.fake_computer_s}
        return FakeSubAgent(durations.get(tier, self.cfg.fake_quick_s))

    def _register_runners(self) -> None:
        """What she can hand off, per tier. computer stays simulated until P6."""
        caps = self.harness.capabilities
        if not self.codex_on:
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
        if self.codex_on:
            self._register_runners()
            self.codex_fingerprint = build_codex_home(self.root / self.cfg.codex_home, self.cfg.model,
                                                      self.cfg.quick_effort)
        self.conversation = Conversation(ConversationLog(self.root / self.cfg.memory_db), ctx.ctl.session_id)
        self.turns.conversation = self.conversation
        self.turns.llm = ctx.ctl.llm  # for the correction pass
        self.harness.capabilities.register(Capability(
            "memory", "live", "remembers the recent conversation, also across restarts; finds older talk by search",
            ("search_conversation",), "facts about the user across days come later (P5)"))
        ctx.ctl.history = self.conversation.restore(min(self.cfg.restore_turns, 2 * ctx.ctl.history_turns))
        ctx.ctl.hooks = self.turns
        self.harness.inbox.subscribe(self._on_event)
        interrupted = self.harness.recover()  # after the arbiter subscribed, so she mentions them once
        pruned = prune_workspaces(self.root / self.cfg.tasks_dir, self.harness.registry, self.cfg.retention_days)
        still = [t for t in self.harness.registry.list(status="running") if t.metadata.get("detached")]
        ctx.info.update(agent=self._info())
        runner = (f"codex sub-agents ({self.codex_fingerprint['model']} via {self.codex_fingerprint['provider']}/"
                  f"{self.codex_fingerprint['endpoint']}, account #{self.codex_fingerprint['account']})"
                  if self.codex_on else "fake sub-agent")
        print(f"maid agent: routing={self.cfg.routing}, {runner}, db {self.root / self.cfg.db}, "
              f"{len(ctx.ctl.history)} message(s) of conversation restored"
              + (f", {len(interrupted)} task(s) interrupted by the last restart" if interrupted else "")
              + (f", {len(still)} sub-agent(s) still running from before" if still else "")
              + (f", {pruned} old workspace(s) pruned" if pruned else "")
              + f", sub-agent tokens today: {self._tokens_today()}"
              + (f" of {self.cfg.daily_token_budget}" if self.cfg.daily_token_budget else ""), flush=True)
        tasks = [asyncio.create_task(self.arbiter.run(ctx)), asyncio.create_task(self.harness.watch())]
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

    def _tokens_today(self) -> int:
        return usage_since(self.harness.registry.list(), midnight())

    def _info(self) -> dict:
        return {"routing": self.cfg.routing, "muted": self.arbiter.cfg.muted, "tokens_today": self._tokens_today(),
                "token_budget": self.cfg.daily_token_budget,
                "capabilities": [{"name": c.name, "status": c.status, "note": c.note}
                                 for c in self.harness.capabilities.all()]}
