"""Routing inside the reply call (ARCHITECTURE §4.2): the TurnHooks that maid_voice's controller calls.

plan()  adds 「できること」 (from the harness capability list), the tool rules and the T1 state block
        to every reply call, plus keyword backstops
ack()   a fixed line for when she calls a tool without saying anything
run()   the side effects: submit / cancel / confirm through the harness API, or a lookup (look_at_screen)
        whose result goes back to her as a follow-up line, once the turn is committed
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from datetime import date, datetime
from pathlib import Path

from maid_harness import Harness
from maid_voice.controller import TurnPlan
from maid_voice.llm import ToolCall

from .router import route_text, wants_cancel, yes_no
from .tools import ACTIVE, INSTRUCTIONS, TIERS, capability_block, state_block, tools_for

KEYWORD_NOTE = "# この発言について\n作業として受け付けた。ごく短く引き受けるだけにして、結果や進み具合は言わない。"
CANCEL_NOTE = "# この発言について\n作業を止めた。短く伝えるだけにする。"
NOTICE_RULES = """# 知らせ方
これはあきらの発言ではなく、作業についての知らせ。あきらに向けて、ナツメらしく一、二文の日本語で伝える。
内容は変えない、足さない。終わっていない作業を終わったと言わない。英語の名前はカタカナで言う。"""
CONFIRM_WINDOW_S = 120.0  # a bare 「うん」 only confirms a task she asked about this recently
LOOK_TIMEOUT_S = 15.0     # vision is ~5 s warm; a capture without screen-recording permission stalls for 30 s


class MaidTurns:
    def __init__(self, harness: Harness, routing: str = "tool", owner_id: str = "default",
                 log_dir: Path | None = None, *,
                 look: Callable[[str], Awaitable[str]] | None = None,
                 followup: Callable[[str, str], None] | None = None,
                 facts: Callable[[], list[str]] | None = None) -> None:
        """look(question) -> what was seen, in English; followup(id, text) speaks a lookup result;
        facts() -> the latest reporter facts for the state block."""
        if routing not in ("tool", "keyword"):
            raise ValueError(f"routing must be 'tool' or 'keyword', not {routing!r}")
        self.harness = harness
        self.routing = routing
        self.owner_id = owner_id
        self.log_dir = log_dir
        self.look, self.followup, self.facts = look, followup, facts

    # ---- TurnHooks ------------------------------------------------------------
    async def plan(self, text: str, kind: str) -> TurnPlan:
        tasks = self.harness.registry.list(owner_id=self.owner_id)
        caps = self.harness.capabilities.usable()
        state = state_block(tasks, facts=self.facts() if self.facts else ())
        if kind == "notice":
            return TurnPlan(context=capability_block(caps) + "\n\n" + NOTICE_RULES + "\n\n" + state)
        active = sorted((t for t in tasks if t.status in ACTIVE), key=lambda t: -t.created_at)
        backstop = self._backstop(text, active)
        if self.routing == "keyword":
            if backstop:
                return TurnPlan(context=state + "\n\n" + CANCEL_NOTE, calls=backstop)
            decision = route_text(text)
            if decision.route in ("task", "computer_use"):
                tier = "computer" if decision.route == "computer_use" else "quick"
                call = ToolCall("delegate", {"goal": text, "tier": tier, "needs": list(decision.required_capabilities)})
                return TurnPlan(context=state + "\n\n" + KEYWORD_NOTE, calls=[call])
            return TurnPlan(context=capability_block(caps) + "\n\n" + state)
        # tool routing; the backstop only runs when the LLM called nothing (she once said 「止める」 without the call)
        context = capability_block(caps) + "\n\n" + INSTRUCTIONS + "\n\n" + state
        return TurnPlan(context=context, tools=tools_for(caps), fallback_calls=backstop)

    def ack(self, calls: list[ToolCall]) -> str | None:
        c = calls[0]
        if c.name == "delegate":
            if c.arguments.get("tier") == "computer":
                return "画面を操作することになるけど、始めていい？"
            return "わかった、やっておく。"
        if c.name == "cancel_task":
            return "わかった、止める。"
        if c.name == "confirm_task":
            return "じゃあ、始める。" if c.arguments.get("approve") else "わかった、やめておく。"
        if c.name == "look_at_screen":
            return "ちょっと見る。"
        return None

    async def run(self, calls: list[ToolCall], user_text: str) -> None:
        for c in calls:
            result = await self._run_one(c, user_text)
            self._log({"time": datetime.now().isoformat(timespec="seconds"), "user_text": user_text,
                       "call": c.name, "arguments": c.arguments, "result": result})

    # ---- internals --------------------------------------------------------------
    def _backstop(self, text: str, active) -> list[ToolCall]:
        if wants_cancel(text) and len(active) == 1 and active[0].status != "waiting_confirmation":
            return [ToolCall("cancel_task", {"task_id": active[0].task_id})]
        waiting = [t for t in active if t.status == "waiting_confirmation"]
        if len(waiting) == 1 and time.time() - waiting[0].created_at < CONFIRM_WINDOW_S:
            answer = yes_no(text)
            if answer is not None:
                return [ToolCall("confirm_task", {"task_id": waiting[0].task_id, "approve": answer})]
        return []

    def _resolve(self, task_id: str | None):
        """The task the LLM named; if it mangled the id and only one task is active, that one."""
        task = self.harness.registry.get(task_id) if task_id else None
        if task is not None:
            return task
        active = [t for t in self.harness.registry.list(owner_id=self.owner_id) if t.status in ACTIVE]
        return active[0] if len(active) == 1 else None

    async def _run_one(self, c: ToolCall, user_text: str) -> dict:
        a = c.arguments
        if c.name == "delegate":
            goal = str(a.get("goal") or user_text).strip()
            tier = a.get("tier") if a.get("tier") in TIERS else "quick"
            needs = [str(n) for n in a.get("needs") or []]
            task = await self.harness.submit(
                user_text or goal, route="computer_use" if tier == "computer" else "task",
                owner_id=self.owner_id, worker_kind=f"fake-{tier}", requires_confirmation=tier == "computer",
                metadata={"goal": goal, "tier": tier, "needs": needs, "routing": self.routing})
            return {"task_id": task.task_id, "status": task.status}
        if c.name in ("cancel_task", "confirm_task"):
            task = self._resolve(a.get("task_id"))
            if task is None:
                return {"error": f"no such task {a.get('task_id')!r}"}
            if c.name == "cancel_task":
                task = await self.harness.cancel(task.task_id)
            elif a.get("approve"):
                task = await self.harness.confirm(task.task_id)
            else:
                task = await self.harness.cancel(task.task_id, reason="declined by user")
            return {"task_id": task.task_id, "status": task.status}
        if c.name == "look_at_screen":
            if self.look is None:
                return {"error": "screen is not connected"}
            question = str(a.get("question") or user_text)
            try:
                seen = await asyncio.wait_for(self.look(question), LOOK_TIMEOUT_S)
            except Exception as e:  # noqa: BLE001 - she said 「ちょっと見る」, so she must say something back
                seen = f"looking failed ({type(e).__name__}); say you could not see it this time"
            if self.followup:
                self.followup("screen", f"[screen, asked: {question}] {seen}")
            return {"seen": seen}
        return {"error": f"unknown tool {c.name!r}"}

    def _log(self, rec: dict) -> None:
        if self.log_dir is None:
            return
        d = self.log_dir / date.today().isoformat()
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "tool_calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
