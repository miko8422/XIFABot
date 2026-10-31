"""P3 + P3.5 on the Maid side. Sub-agents are real detached supervisor processes running fake_codex.py.

P3: answering a sub-agent's question and permission request by voice, follow-ups, capabilities.
P3.5: the "said it, didn't do it" correction, read_task, honest no-effect feedback, the 3-day window and search,
the daily budget.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest
from maid_harness.codex_runner import CodexRunner, Profile
from maid_voice.llm import ChatTurn, ToolCall

from maid_agent.agent import AgentConfig, MaidAgent
from maid_agent.arbiter import SpeechArbiter
from maid_agent.conversation import Conversation, ConversationLog
from maid_agent.tools import capability_block, state_block, tools_for
from maid_agent.turns import CLAIMS, MaidTurns
from test_p1 import build, say

FAKE = Path(__file__).resolve().parents[2] / "maid_harness" / "tests" / "fake_codex.py"


@pytest.fixture
def maid(tmp_path, monkeypatch):
    src = tmp_path / "user_codex"
    src.mkdir()
    (src / "config.toml").write_text('[model_providers.OpenAI]\nname = "OpenAI"\nbase_url = "https://a.example"\n')
    monkeypatch.setenv("MAID_USER_CODEX_HOME", str(src))
    db = tmp_path / "tasks.sqlite3"
    harness, ctl, asr, llm = build(db=db)
    profile = Profile("quick", timeout_s=5)
    harness.worker_factory = lambda t: CodexRunner(db, tmp_path / "tasks", tmp_path / "home", profile,
                                                   codex=f"{sys.executable} {FAKE}")
    return harness, ctl, asr, llm


async def settled(harness, task_id, *states):
    for _ in range(300):
        harness.pump()
        t = harness.registry.get(task_id)
        if (t.status in states) if states else t.status not in ("accepted", "running"):
            return t
        await asyncio.sleep(0.03)
    raise AssertionError(harness.registry.get(task_id).status)


# ---- P3 -------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_she_relays_the_question_and_her_answer_resumes_the_sub_agent(maid):
    harness, ctl, asr, llm = maid
    arbiter = SpeechArbiter(harness)
    llm.text, llm.calls = "わかった、調べてくる。", [ToolCall("delegate", {"goal": "SCENARIO:ask weather", "tier": "quick"})]
    await say(ctl, asr, "天気調べて")
    [task] = harness.registry.list()
    t = await settled(harness, task.task_id)
    assert t.status == "needs_input"
    assert [i.kind for i in arbiter.pending] == ["needs_input"] and "Which city?" in arbiter.pending[0].text
    llm.text, llm.calls = "大阪ね、伝えとく。", [ToolCall("answer_task", {"task_id": task.task_id, "answer": "Osaka"})]
    await say(ctl, asr, "大阪で")
    assert "あきらの答え待ち" in llm.requests[-1]["context"] and "Which city?" in llm.requests[-1]["context"]
    t = await settled(harness, task.task_id, "completed")
    assert "Osaka" in t.summary


@pytest.mark.asyncio
async def test_a_permission_request_is_asked_by_voice_and_a_bare_yes_grants_it(maid):
    harness, ctl, asr, llm = maid
    arbiter = SpeechArbiter(harness)
    task = await harness.submit("x", metadata={"goal": "SCENARIO:perm", "tier": "work"})
    t = await settled(harness, task.task_id)
    assert t.status == "needs_permission" and "asks permission to read /tmp/elsewhere" in arbiter.pending[0].text
    assert "頼まれている許可: read /tmp/elsewhere" in state_block(harness.registry.list())
    llm.text, llm.calls = "じゃあ、許可する。", []  # she forgot confirm_task: the yes/no backstop answers it
    await say(ctl, asr, "うん、いいよ")
    t = await settled(harness, task.task_id, "completed")
    assert "granted" in t.summary and t.metadata["grants"][0]["target"] == "/tmp/elsewhere"


@pytest.mark.asyncio
async def test_follow_up_continues_a_finished_task_instead_of_starting_a_new_one(maid):
    harness, ctl, asr, llm = maid
    task = await harness.submit("x", metadata={"goal": "SCENARIO:done", "tier": "quick"})
    await settled(harness, task.task_id)
    llm.text, llm.calls = "続けさせる。", [ToolCall("follow_up_task", {"task_id": "wrong", "message": "cheaper ones"})]
    await say(ctl, asr, "もっと安いのは？")
    t = await settled(harness, task.task_id, "completed")
    assert len(harness.registry.list()) == 1 and "cheaper ones" in t.summary and "cheaper ones" in t.goal


def test_with_codex_she_is_told_research_and_work_are_real_and_computer_use_is_practice():
    agent = MaidAgent(Path("/tmp"), AgentConfig())
    agent._register_runners()
    caps = agent.harness.capabilities.usable()
    block = capability_block(caps)
    assert "research（使える）" in block and "work（使える）" in block and "computer_use（練習用）" in block
    names = {t["function"]["name"] for t in tools_for(caps)}
    assert {"delegate", "answer_task", "follow_up_task", "cancel_task", "confirm_task", "read_task", "search_tasks"} <= names
    assert type(agent._worker(type("T", (), {"metadata": {"tier": "computer"}})())).__name__ == "FakeSubAgent"
    assert type(agent._worker(type("T", (), {"metadata": {"tier": "quick"}})())).__name__ == "CodexRunner"


# ---- P3.5 -----------------------------------------------------------------------------------------
def test_claim_words_catch_the_live_failure_and_leave_plain_chat_alone():
    for said in ("もっと安いの、探し直してくる。", "わかった、止める。", "続けさせる。", "調べてくるね", "やっておく。"):
        assert CLAIMS.search(said), said
    for said in ("ふーん。そう。", "まだ途中みたい。", "十八度でくもりだって。"):
        assert not CLAIMS.search(said), said


@pytest.mark.asyncio
async def test_said_it_but_called_nothing_gets_one_correction_call_that_does_it(maid):
    harness, ctl, asr, llm = maid
    task = await harness.submit("x", metadata={"goal": "SCENARIO:done"})
    await settled(harness, task.task_id)
    asked = []

    async def complete(messages, tools=None, max_tokens=300):
        asked.append(messages)
        return "", [ToolCall("follow_up_task", {"task_id": task.task_id, "message": "cheaper than 2000 yen"})]

    llm.complete = complete
    ctl.hooks.llm = llm
    llm.text, llm.calls = "もっと安いの、探し直してくる。", []  # the live failure: the claim, no tool call
    await say(ctl, asr, "それ高すぎ、もっと安いのはある？")
    t = await settled(harness, task.task_id, "completed")
    assert len(asked) == 1 and "探し直してくる" in asked[0][-1]["content"]
    assert "cheaper than 2000 yen" in t.summary

    llm.text = "ふーん、そう。"
    await say(ctl, asr, "今日は疲れた")
    assert len(asked) == 1, "no claim, no extra call"


@pytest.mark.asyncio
async def test_a_tool_call_that_changes_nothing_is_reported_back_to_her(maid):
    harness, ctl, asr, llm = maid
    arbiter = SpeechArbiter(harness)
    ctl.hooks.followup = arbiter.answer
    task = await harness.submit("x", metadata={"goal": "SCENARIO:sleep"})
    await settled(harness, task.task_id, "running")
    llm.text, llm.calls = "伝えておく。", [ToolCall("answer_task", {"task_id": task.task_id, "answer": "Osaka"})]
    await say(ctl, asr, "大阪で")
    assert any("[correction] answer_task did nothing" in i.text for i in arbiter.pending)
    await harness.cancel(task.task_id)


@pytest.mark.asyncio
async def test_read_task_gives_her_the_full_result_and_the_state_shows_the_newest_details(maid):
    harness, ctl, asr, llm = maid
    arbiter = SpeechArbiter(harness)
    ctl.hooks.followup = arbiter.answer
    task = await harness.submit("x", metadata={"goal": "SCENARIO:done find jackets"})
    await settled(harness, task.task_id)
    assert "1980 yen" in state_block(harness.registry.list()), "the newest result's details are in her context"
    llm.text, llm.calls = "", [ToolCall("read_task", {"task_id": task.task_id})]
    await say(ctl, asr, "さっきの全部教えて")
    assert ctl.completed[-1]["assistant_text"] == "ちょっと見てみる。"
    assert "full list: A 1980 yen" in arbiter.pending[0].text


def test_state_shows_three_days_and_older_tasks_are_found_by_search():
    harness, ctl, asr, llm = build()
    now = time.time()
    for i, age_days in enumerate((0.1, 0.2, 0.3, 1, 5)):
        t = harness.registry.create_task(f"task {i}", "task", metadata={"goal": f"jacket search {i}"})
        harness.registry.update(t.task_id, status="completed", summary=f"found {i}", now=now - age_days * 86400)
    block = state_block(harness.registry.list(), now=now)
    assert "jacket search 0" in block and "jacket search 3" not in block and "jacket search 4" not in block
    assert "ほかに三日以内の作業が1件" in block
    assert [t.goal for t in harness.registry.search("jacket 4")][0] == "jacket search 4"


@pytest.mark.asyncio
async def test_search_conversation_finds_older_talk(maid):
    harness, ctl, asr, llm = maid
    arbiter = SpeechArbiter(harness)
    log = ConversationLog()
    log.append("old", [ChatTurn("user", "来週の金曜は歯医者だよ"), ChatTurn("assistant", "ん、覚えとく。")], "ptt")
    ctl.hooks.conversation = Conversation(log, "new")
    ctl.hooks.followup = arbiter.answer
    llm.text, llm.calls = "", [ToolCall("search_conversation", {"query": "歯医者"})]
    await say(ctl, asr, "前に予定の話したよね")
    assert "来週の金曜は歯医者だよ" in arbiter.pending[0].text


@pytest.mark.asyncio
async def test_no_new_sub_agent_once_the_daily_budget_is_spent(maid):
    harness, ctl, asr, llm = maid
    arbiter = SpeechArbiter(harness)
    ctl.hooks.followup, ctl.hooks.daily_token_budget = arbiter.answer, 100
    old = harness.registry.create_task("x", "task", metadata={"usage": [{"input_tokens": 90, "output_tokens": 20,
                                                                           "t": time.time()}]})
    llm.text, llm.calls = "わかった。", [ToolCall("delegate", {"goal": "more", "tier": "quick"})]
    await say(ctl, asr, "これも調べて")
    assert [t.task_id for t in harness.registry.list()] == [old.task_id]
    assert any("budget is used up (110 of 100" in i.text for i in arbiter.pending)


def test_old_workspaces_are_pruned_but_their_logs_stay(tmp_path):
    from maid_harness.codex_runner import prune_workspaces

    harness, ctl, asr, llm = build()
    old = harness.registry.create_task("old", "task")
    new = harness.registry.create_task("new", "task")
    harness.registry.update(old.task_id, status="completed", now=time.time() - 10 * 86400)
    harness.registry.update(new.task_id, status="completed")
    for t in (old, new):
        ws = tmp_path / t.task_id
        (ws / "build").mkdir(parents=True)
        (ws / "build" / "big.bin").write_bytes(b"x" * 100)
        (ws / "events.jsonl").write_text("{}\n")
    assert prune_workspaces(tmp_path, harness.registry, retention_days=7) == 1
    assert sorted(p.name for p in (tmp_path / old.task_id).iterdir()) == ["events.jsonl"]
    assert (tmp_path / new.task_id / "build" / "big.bin").exists()
    assert prune_workspaces(tmp_path, harness.registry, retention_days=7) == 0, "pruned once"
