"""P3 on the Maid side: answering a sub-agent's question and permission request by voice, follow-ups,
and what she is told she can do once real sub-agents are wired in. The sub-agent is fake_codex.py."""

import asyncio
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from maid_harness.codex_runner import CodexRunner, Profile
from maid_harness.subagent_mcp import SubAgentTools
from maid_voice.llm import ToolCall

from maid_agent.agent import AgentConfig, MaidAgent
from maid_agent.arbiter import SpeechArbiter
from maid_agent.tools import capability_block, state_block, tools_for
from test_p1 import build, say

FAKE = Path(__file__).resolve().parents[2] / "maid_harness" / "tests" / "fake_codex.py"


class FakeCodexRunner(CodexRunner):
    def command(self, task, workspace):
        return [sys.executable, str(FAKE), *super().command(task, workspace)[1:]]


@pytest_asyncio.fixture
async def maid(tmp_path):
    harness, ctl, asr, llm = build()
    tools = SubAgentTools(harness)
    await tools.start()
    harness.worker_factory = lambda t: FakeCodexRunner(harness, tools, tmp_path, Profile("quick", timeout_s=5))
    yield harness, ctl, asr, llm
    await tools.stop()


async def settled(harness, task_id):
    for _ in range(250):
        t = harness.registry.get(task_id)
        if t.status not in ("accepted", "running"):
            return t
        await asyncio.sleep(0.02)
    raise AssertionError(t.status)


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
    t = await settled(harness, task.task_id)
    assert t.status == "completed" and "Osaka" in t.summary


@pytest.mark.asyncio
async def test_a_permission_request_is_asked_by_voice_and_a_bare_yes_grants_it(maid):
    harness, ctl, asr, llm = maid
    arbiter = SpeechArbiter(harness)
    task = await harness.submit("x", metadata={"goal": "SCENARIO:perm", "tier": "work"})
    t = await settled(harness, task.task_id)
    assert t.status == "needs_permission"
    assert "asks permission to write /tmp/elsewhere/x.txt" in arbiter.pending[0].text
    assert "頼まれている許可: write /tmp/elsewhere/x.txt" in state_block(harness.registry.list())
    llm.text, llm.calls = "じゃあ、許可する。", []  # she forgot confirm_task: the yes/no backstop answers it
    await say(ctl, asr, "うん、いいよ")
    t = await settled(harness, task.task_id)
    assert t.status == "completed" and "granted" in t.summary and t.metadata["grants"][0]["target"] == "/tmp/elsewhere"


@pytest.mark.asyncio
async def test_follow_up_continues_a_finished_task_instead_of_starting_a_new_one(maid):
    harness, ctl, asr, llm = maid
    task = await harness.submit("x", metadata={"goal": "SCENARIO:done", "tier": "quick"})
    await settled(harness, task.task_id)
    llm.text, llm.calls = "続けさせる。", [ToolCall("follow_up_task", {"task_id": "wrong", "message": "also Osaka"})]
    await say(ctl, asr, "ついでに大阪も")
    t = await settled(harness, task.task_id)
    assert len(harness.registry.list()) == 1 and t.status == "completed" and "also Osaka" in t.summary


def test_with_codex_she_is_told_research_and_work_are_real_and_computer_use_is_practice():
    agent = MaidAgent(Path("/tmp"), AgentConfig())
    agent.tools = object()  # stands in for a started SubAgentTools
    agent._register_runners()
    caps = agent.harness.capabilities.usable()
    block = capability_block(caps)
    assert "research（使える）" in block and "work（使える）" in block and "computer_use（練習用）" in block
    assert "sub_agents" not in block
    names = {t["function"]["name"] for t in tools_for(caps)}
    assert {"delegate", "answer_task", "follow_up_task", "cancel_task", "confirm_task"} <= names
    assert agent._worker(type("T", (), {"metadata": {"tier": "computer"}})()).__class__.__name__ == "FakeSubAgent"
