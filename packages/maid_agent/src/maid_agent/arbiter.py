"""Speech arbiter v1 (ARCHITECTURE §4.3): the one place that decides when task news is spoken.

  inbox event / heartbeat ─▶ pending ─▶ wait for a pause ─▶ merge into one notice ─▶ ctl.notify()
                                         never over あきら, never over her own reply,
                                         and a short quiet gap after either

Rules (numbers match §4.3):
  1/2  only when nobody is speaking and the last activity was `quiet_gap_s` ago
  3    everything pending goes out as one line
  4    completed / failed / needs_input / interrupted are spoken unprompted
  5    progress only when asked (the T1 state block), or as a heartbeat every `heartbeat_s` per long task
  6    per-activity rules (P4, `SpeechRules`): in a call or away, task news is held; while gaming, one short
       sentence; back after `greet_after_s` away -> she greets and tells what came in meanwhile
  7    mute holds everything until unmuted
A notice that gets interrupted goes back into the queue once. Every decision is logged with its rule.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from maid_harness import Harness
from maid_harness.models import TaskEvent

SPOKEN = {"task.completed", "task.failed", "task.needs_input", "task.needs_permission", "task.interrupted"}
HEADER = "（あきらの発言ではない。作業の知らせや、確かめた結果）次のことを伝えて:"


@dataclass
class Rule:
    task_news: str = "speak"        # speak | hold
    proactive: bool = True          # unprompted remarks about the screen
    cooldown_s: float = 600.0       # at least this long between unprompted remarks
    style: str = ""                 # how to say things in this activity, for the LLM


@dataclass
class SpeechRules:
    """When she may speak, per activity (§4.3 rule 6, §5.6). Configured in [agent.speech.<activity>]."""
    rules: dict[str, Rule] = field(default_factory=lambda: {
        "working": Rule("speak", True, 600.0),
        "gaming": Rule("speak", True, 300.0, "あきらはゲーム中。邪魔にならないよう、一文だけ短く。"),
        "call": Rule("hold", False),
        "away": Rule("hold", False),
    })
    greet_after_s: float = 600.0    # back after this long away: she says something

    def get(self, activity: str) -> Rule:
        return self.rules.get(activity) or self.rules["working"]

    @classmethod
    def from_raw(cls, raw: dict) -> "SpeechRules":
        out = cls()
        for name, r in raw.items():
            if isinstance(r, dict):
                base = out.rules.get(name, Rule())
                out.rules[name] = Rule(**{**base.__dict__, **{k: r[k] for k in Rule.__dataclass_fields__ if k in r}})
        if "greet_after_s" in raw:
            out.greet_after_s = float(raw["greet_after_s"])
        return out


@dataclass
class ArbiterConfig:
    quiet_gap_s: float = 1.5      # silence required after anyone spoke
    heartbeat_s: float = 180.0    # unprompted progress for a long task, at most this often
    muted: bool = False
    tick_s: float = 0.25


@dataclass
class Item:
    kind: str          # completed | failed | needs_input | interrupted | progress | lookup | back
    task_id: str
    text: str
    retried: bool = False


class SpeechArbiter:
    def __init__(self, harness: Harness, cfg: ArbiterConfig | None = None, log_dir: Path | None = None,
                 clock=time.time, rules: SpeechRules | None = None,
                 activity: Callable[[], str] = lambda: "working") -> None:
        self.harness = harness
        self.cfg = cfg or ArbiterConfig()
        self.rules = rules or SpeechRules()
        self.activity = activity
        self.log_dir = log_dir
        self.clock = clock
        self.pending: list[Item] = []
        self._last_busy = 0.0
        self._heartbeat: dict[str, float] = {}  # task_id -> last time progress was spoken (or the task started)
        harness.inbox.subscribe(self._on_event)

    # ---- inbox -> pending -------------------------------------------------------
    def _on_event(self, event: TaskEvent) -> None:
        if event.type == "user.back":
            away = float(event.payload.get("away_s", 0))
            if away >= self.rules.greet_after_s:
                self.pending = [i for i in self.pending if i.kind != "back"]
                self.pending.insert(0, Item("back", "user", f"[the user is back after {int(away // 60)} min away; "
                                                            "greet them briefly first]"))
            return
        if event.type not in SPOKEN or not event.task_id:
            return
        task = self.harness.registry.get(event.task_id)
        goal = task.goal if task else ""
        p = event.payload
        detail = {"task.completed": p.get("summary"), "task.failed": p.get("error"),
                  "task.needs_input": f"the sub-agent asks the user: {p.get('question')}",
                  "task.needs_permission": f"the sub-agent asks permission to {p.get('what')} (because {p.get('why')}); "
                                           "ask the user yes or no",
                  "task.interrupted": p.get("reason")}[event.type]
        kind = event.type.split(".", 1)[1]
        # a newer state for the same task replaces an older one still waiting
        self.pending = [i for i in self.pending if i.task_id != event.task_id]
        self.pending.append(Item(kind, event.task_id, f"[{kind}] {goal}: {detail or ''}".rstrip(": ")))
        self._heartbeat.pop(event.task_id, None)

    def _due_heartbeats(self, now: float) -> list[Item]:
        out = []
        for t in self.harness.registry.list(status="running"):
            last = self._heartbeat.setdefault(t.task_id, t.created_at)
            if now - last >= self.cfg.heartbeat_s and not any(i.task_id == t.task_id for i in self.pending):
                self._heartbeat[t.task_id] = now
                goal = t.goal
                out.append(Item("progress", t.task_id, f"[still running, {int((now - t.created_at) // 60)} min] "
                                f"{goal}: {t.progress}% {t.progress_text}".rstrip()))
        return out

    def answer(self, task_id: str, text: str) -> None:
        """The result of an in-turn lookup (look_at_screen): first in line, same pause rules."""
        self.pending.insert(0, Item("lookup", task_id, text))

    # ---- the loop ------------------------------------------------------------------
    def can_speak(self, ctx, now: float) -> bool:
        busy = ctx.user_speaking() or ctx.ctl.state != "listening" or bool(ctx.ctl.active_gen)
        if busy:
            self._last_busy = now
        return not busy and now - self._last_busy >= self.cfg.quiet_gap_s

    def _fresh(self, items: list[Item]) -> list[Item]:
        """Drop a question or permission request whose task has moved on (answered meanwhile, or late)."""
        out = []
        for i in items:
            if i.kind in ("needs_input", "needs_permission"):
                task = self.harness.registry.get(i.task_id)
                if task is None or task.status != i.kind:
                    self._log("4 stale", "drop", [i])
                    continue
            out.append(i)
        return out

    def prompt(self, items: list[Item], style: str = "") -> str:
        return HEADER + "\n" + "\n".join(f"- {i.text}" for i in items) + (f"\n（{style}）" if style else "")

    def deliverable(self, rule: Rule) -> tuple[list[Item], list[Item]]:
        """(say now, keep). Answers to what あきら just asked (lookups) are never held by an activity rule."""
        if rule.task_news == "speak":
            return self.pending, []
        return [i for i in self.pending if i.kind == "lookup"], [i for i in self.pending if i.kind != "lookup"]

    async def run(self, ctx) -> None:
        held_logged = None
        while True:
            await asyncio.sleep(self.cfg.tick_s)
            now = self.clock()
            self.pending += self._due_heartbeats(now)
            free = self.can_speak(ctx, now)
            if not self.pending:
                continue
            activity = self.activity()
            rule = self.rules.get(activity)
            if self.cfg.muted:
                if held_logged != "mute":
                    self._log("7 mute", "hold", self.pending)
                    held_logged = "mute"
                continue
            say, keep = self.deliverable(rule)
            if not say:
                if held_logged != activity:
                    self._log(f"6 {activity}", "hold", keep)
                    held_logged = activity
                continue
            held_logged = None
            if not free:
                continue
            items, self.pending = self._fresh(say), keep
            if not items:
                continue
            task = ctx.ctl.notify(self.prompt(items, rule.style), source="arbiter")
            if task is None:  # lost a race with a turn that just started
                self.pending = items + self.pending
                continue
            self._log("3/4 merged notice" if len(items) > 1 else "4 notice", "speak", items)
            await asyncio.wait([task])
            if task.cancelled():  # barge-in or a new turn cut her off: say it once more later
                again = [i for i in items if not i.retried]
                for i in again:
                    i.retried = True
                self.pending = again + self.pending
                self._log("1 interrupted", "requeue" if again else "drop", items)

    # ---- bookkeeping -------------------------------------------------------------------
    def _log(self, rule: str, action: str, items: list[Item]) -> None:
        if self.log_dir is None:
            return
        d = self.log_dir / date.today().isoformat()
        d.mkdir(parents=True, exist_ok=True)
        rec = {"time": datetime.now().isoformat(timespec="seconds"), "rule": rule, "action": action,
               "items": [{"kind": i.kind, "task_id": i.task_id, "text": i.text} for i in items]}
        with open(d / "arbiter.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
