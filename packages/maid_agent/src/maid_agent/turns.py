"""Routing inside the reply call (ARCHITECTURE §4.2): the TurnHooks that maid_voice's controller calls.

plan()  adds 「できること」 (from the harness capability list), the tool rules and the T1 state block
        to every reply call, plus keyword backstops
ack()   a fixed line for when she calls a tool without saying anything
remember()  every turn that entered her history goes to the conversation log (P2); and if her reply claimed
        an action but called no tool, one correction call runs (P3.5: 「探し直してくる」 with no call)
run()   the side effects: submit / cancel / confirm / answer / follow up through the harness API, or a lookup
        (look_at_screen, read_task, search_*) whose result goes back to her as a follow-up line. A call that
        changed nothing is reported back too, so she corrects herself instead of claiming it worked.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from datetime import date, datetime
from pathlib import Path

from maid_harness import Harness
from maid_voice.controller import TurnPlan
from maid_voice.llm import ChatTurn, LLMAdapter, ToolCall

from .conversation import Conversation

from .router import route_text, wants_cancel, yes_no
from .tools import ACTIVE, INSTRUCTIONS, TIERS, capability_block, state_block, tools_for, usage_since

KEYWORD_NOTE = "# この発言について\n作業として受け付けた。ごく短く引き受けるだけにして、結果や進み具合は言わない。"
CANCEL_NOTE = "# この発言について\n作業を止めた。短く伝えるだけにする。"
NOTICE_RULES = """# 知らせ方
これはあきらの発言ではなく、作業についての知らせ。あきらに向けて、ナツメらしく一、二文の日本語で伝える。
内容は変えない、足さない。終わっていない作業を終わったと言わない。英語の名前はカタカナで言う。"""
CORRECTION = ("（システム）いまの返答で「{reply}」と言ったが、ツールを呼んでいない。言ったとおりの作業を始める・続ける・"
              "止める・答えるなら、今そのツールを一つ呼ぶ。言っただけで本当は何もしないなら、何も呼ばない。文章はいらない。")
# her words for doing something; a reply with one of these and no tool call gets the correction pass
CLAIMS = re.compile(r"(探し(て|に)く|探し直|調べ(て|に)く|調べ直|見てくる|確認してくる|やっておく|やってくる|やらせ|頼んで(おく|くる)|"
                    r"続けさせ|続けてもら|もう一度(やら|探|調べ)|止め(る|とく|ておく)|やめ(る|とく|ておく)|伝えて?おく|始め(る|させ))")
CONFIRM_WINDOW_S = 120.0  # a bare 「うん」 only confirms a task she asked about this recently
LOOK_TIMEOUT_S = 20.0     # waiting for her acknowledgement (<=5 s) + vision (~5 s warm); a capture
                          # without screen-recording permission stalls for 30 s


class MaidTurns:
    def __init__(self, harness: Harness, routing: str = "tool", owner_id: str = "default",
                 log_dir: Path | None = None, *,
                 look: Callable[[str], Awaitable[str]] | None = None,
                 followup: Callable[[str, str], None] | None = None,
                 facts: Callable[[], list[str]] | None = None,
                 conversation: Conversation | None = None,
                 llm: LLMAdapter | None = None, daily_token_budget: int = 0) -> None:
        """look(question) -> what was seen, in English; followup(id, text) speaks a lookup result;
        facts() -> the latest reporter facts for the state block."""
        if routing not in ("tool", "keyword"):
            raise ValueError(f"routing must be 'tool' or 'keyword', not {routing!r}")
        self.harness = harness
        self.routing = routing
        self.owner_id = owner_id
        self.log_dir = log_dir
        self.look, self.followup, self.facts = look, followup, facts
        self.conversation = conversation
        self.llm = llm                      # for the correction pass; None = no correction
        self.daily_token_budget = daily_token_budget
        self._plans: dict[str, TurnPlan] = {}  # user text -> the plan of its reply call, for the correction pass

    # ---- TurnHooks ------------------------------------------------------------
    async def plan(self, text: str, kind: str) -> TurnPlan:
        tasks = self.harness.registry.list(owner_id=self.owner_id)
        caps = self.harness.capabilities.usable()
        state = state_block(tasks, facts=self.facts() if self.facts else ())
        if self.conversation and (note := self.conversation.note()):
            state = note + "\n\n" + state
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
        plan = TurnPlan(context=context, tools=tools_for(caps), fallback_calls=backstop)
        self._plans = {text: plan}
        return plan

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
        if c.name == "answer_task":
            return "わかった、伝えておく。"
        if c.name == "follow_up_task":
            return "わかった、続けさせる。"
        if c.name == "read_task":
            return "ちょっと見てみる。"
        if c.name in ("search_tasks", "search_conversation"):
            return "探してみる。"
        return None

    def remember(self, turns: list[ChatTurn], mode: str, tool_calls: list[dict] = ()) -> None:
        if self.conversation:
            self.conversation.remember(turns, mode)
        user = next((t.content for t in turns if t.role == "user"), None)
        reply = next((t.content for t in turns if t.role == "assistant"), "")
        plan = self._plans.pop(user, None) if user is not None else None
        if plan and plan.tools and not tool_calls and self.llm and CLAIMS.search(reply):
            asyncio.ensure_future(self._correct(user, reply, plan))

    async def _correct(self, user_text: str, reply: str, plan: TurnPlan) -> None:
        """One extra call, only when she said she would do something and called nothing."""
        messages = [{"role": "system", "content": plan.context}, {"role": "user", "content": user_text},
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": CORRECTION.format(reply=reply)}]
        rec = {"time": datetime.now().isoformat(timespec="seconds"), "user_text": user_text, "reply": reply}
        try:
            _, calls = await self.llm.complete(messages, plan.tools)
        except Exception as e:  # noqa: BLE001 - the correction is best effort, off the voice path
            self._log({**rec, "call": "correction", "result": {"error": f"{type(e).__name__}: {e}"}}, "corrections")
            return
        self._log({**rec, "call": "correction", "result": {"calls": [c.name for c in calls]}}, "corrections")
        if calls:
            await self.run(calls[:1], user_text)

    async def run(self, calls: list[ToolCall], user_text: str) -> None:
        for c in calls:
            result = await self._run_one(c, user_text)
            self._log({"time": datetime.now().isoformat(timespec="seconds"), "user_text": user_text,
                       "call": c.name, "arguments": c.arguments, "result": result})
            if result.get("no_effect") and self.followup:
                self.followup(result.get("task_id") or "tool",
                              f"[correction] {c.name} did nothing: {result['no_effect']}. Tell the user honestly, "
                              "and say what you can do instead.")

    # ---- internals --------------------------------------------------------------
    def _backstop(self, text: str, active) -> list[ToolCall]:
        if wants_cancel(text) and len(active) == 1 and active[0].status != "waiting_confirmation":
            return [ToolCall("cancel_task", {"task_id": active[0].task_id})]
        waiting = [t for t in active if t.status in ("waiting_confirmation", "needs_permission")]
        if len(waiting) == 1 and time.time() - waiting[0].updated_at < CONFIRM_WINDOW_S:
            answer = yes_no(text)
            if answer is not None:
                return [ToolCall("confirm_task", {"task_id": waiting[0].task_id, "approve": answer})]
        return []

    def _resolve(self, task_id: str | None, any_status: bool = False):
        """The task the LLM named; if it mangled the id and only one task fits, that one
        (one active task, or for a follow-up the most recent task)."""
        task = self.harness.registry.get(task_id) if task_id else None
        if task is not None:
            return task
        tasks = self.harness.registry.list(owner_id=self.owner_id)  # newest first
        if any_status:
            return tasks[0] if tasks else None
        active = [t for t in tasks if t.status in ACTIVE]
        return active[0] if len(active) == 1 else None

    async def _run_one(self, c: ToolCall, user_text: str) -> dict:
        a = c.arguments
        if c.name == "delegate":
            if self.daily_token_budget and (used := usage_since(self.harness.registry.list(), midnight())) \
                    >= self.daily_token_budget:
                return {"no_effect": f"today's sub-agent budget is used up ({used} of {self.daily_token_budget} tokens)"}
            goal = str(a.get("goal") or user_text).strip()
            tier = a.get("tier") if a.get("tier") in TIERS else "quick"
            needs = [str(n) for n in a.get("needs") or []]
            task = await self.harness.submit(
                user_text or goal, route="computer_use" if tier == "computer" else "task",
                owner_id=self.owner_id, worker_kind=f"subagent-{tier}", requires_confirmation=tier == "computer",
                metadata={"goal": goal, "tier": tier, "needs": needs, "routing": self.routing})
            return {"task_id": task.task_id, "status": task.status}
        if c.name in ("cancel_task", "confirm_task", "answer_task", "follow_up_task"):
            task = self._resolve(a.get("task_id"), any_status=c.name == "follow_up_task")
            if task is None:
                return {"error": f"no such task {a.get('task_id')!r}"}
            h, before = self.harness, task.status
            if c.name == "cancel_task":
                task = await h.cancel(task.task_id)
            elif c.name == "answer_task":
                task = await h.answer(task.task_id, str(a.get("answer") or user_text))
            elif c.name == "follow_up_task":
                task = await h.follow_up(task.task_id, str(a.get("message") or user_text))
            elif task.status == "needs_permission":
                task = await h.grant(task.task_id, bool(a.get("approve")))
            elif a.get("approve"):
                task = await h.confirm(task.task_id)
            else:
                task = await h.cancel(task.task_id, reason="declined by user")
            out = {"task_id": task.task_id, "status": task.status}
            if task.status == before and c.name != "cancel_task" or (c.name == "cancel_task" and before not in ACTIVE):
                out["no_effect"] = f"task {task.task_id} ({task.goal[:80]}) is {task.status}, so {c.name} could not apply"
            return out
        if c.name == "read_task":
            task = self._resolve(a.get("task_id"), any_status=True)
            if task is None:
                return {"no_effect": f"no such task {a.get('task_id')!r}"}
            m = task.metadata
            body = (m.get("result") or {}).get("details") or m.get("last_message") or task.progress_text or "nothing yet"
            if self.followup:
                self.followup(task.task_id, f"[task {task.task_id}, {task.status}] {task.goal[:120]}: {body[:2500]}")
            return {"task_id": task.task_id, "read": len(body)}
        if c.name == "search_tasks":
            found = self.harness.registry.search(str(a.get("query") or user_text))
            text = "; ".join(f"{t.task_id} ({datetime.fromtimestamp(t.created_at):%m/%d}, {t.status}): {t.goal[:90]}"
                             + (f" -> {t.summary[:120]}" if t.summary else "") for t in found) or "nothing found"
            if self.followup:
                self.followup("search", f"[task search: {a.get('query')}] {text}")
            return {"found": [t.task_id for t in found]}
        if c.name == "search_conversation":
            found = self.conversation.log.search(str(a.get("query") or user_text)) if self.conversation else []
            text = "; ".join(f"{datetime.fromtimestamp(t.t):%m/%d %H:%M} {'あきら' if t.role == 'user' else 'ナツメ'}: "
                             f"{t.content[:120]}" for t in found) or "nothing found"
            if self.followup:
                self.followup("search", f"[conversation search: {a.get('query')}] {text}")
            return {"found": len(found)}
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

    def _log(self, rec: dict, name: str = "tool_calls") -> None:
        if self.log_dir is None:
            return
        d = self.log_dir / date.today().isoformat()
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{name}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def midnight() -> float:
    return datetime.combine(date.today(), datetime.min.time()).timestamp()
