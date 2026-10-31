"""P3: the Codex runner and the sub-agent MCP tools, driven by fake_codex.py (no network, no model)."""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from maid_harness import Harness
from maid_harness.codex_runner import CodexRunner, Profile, config_overrides
from maid_harness.subagent_mcp import SubAgentTools

FAKE = Path(__file__).with_name("fake_codex.py")


def args_of(overrides: list[str]) -> dict:
    return {k: json.loads(v) for k, v in (overrides[i + 1].split("=", 1) for i in range(0, len(overrides), 2))}


def test_overrides_follow_the_profile_and_grants_widen_it_for_one_task():
    quick = args_of(config_overrides(Profile("quick", "read-only", "low", search=True), "http://x", []))
    assert quick["sandbox_mode"] == "read-only" and quick["web_search"] == "live"
    assert quick["approval_policy"] == "never" and quick["mcp_servers.maid.default_tools_approval_mode"] == "approve"
    assert quick["notify"] == []
    work = Profile("work", "workspace-write", "medium")
    granted = args_of(config_overrides(work, "http://x", [{"kind": "write_path", "target": "/repo"}, {"kind": "network"}]))
    assert granted["sandbox_workspace_write.writable_roots"] == ["/repo"]
    assert granted["sandbox_workspace_write.network_access"] is True
    assert args_of(config_overrides(work, "http://x", [{"kind": "full_access"}]))["sandbox_mode"] == "danger-full-access"


@pytest_asyncio.fixture
async def rig(tmp_path):
    harness = Harness()
    tools = SubAgentTools(harness)
    await tools.start()

    class FakeCodexRunner(CodexRunner):
        def command(self, task, workspace):
            return [sys.executable, str(FAKE), *super().command(task, workspace)[1:]]

    profile = Profile("quick", timeout_s=2)
    harness.worker_factory = lambda t: FakeCodexRunner(harness, tools, tmp_path, profile)
    yield harness, tools, tmp_path
    await tools.stop()


async def finish(harness, task_id, timeout=5.0):
    for _ in range(int(timeout / 0.02)):
        t = harness.registry.get(task_id)
        if t.status not in ("accepted", "running"):
            return t
        await asyncio.sleep(0.02)
    raise AssertionError(f"still {t.status}")


@pytest.mark.asyncio
async def test_a_submitted_result_completes_the_task_and_everything_is_recorded(rig):
    harness, _, root = rig
    task = await harness.submit("東京の天気", metadata={"goal": "SCENARIO:done weather in Tokyo"})
    t = await finish(harness, task.task_id)
    assert t.status == "completed" and t.summary == "It is 19 degrees in Tokyo."
    assert t.metadata["thread_id"] == "thread-123" and t.metadata["usage"]
    events = (root / task.task_id / "events.jsonl").read_text().splitlines()
    assert json.loads(events[0])["type"] == "thread.started"
    assert any(e.type == "task.progress" for e in harness.inbox.list(task.task_id))


@pytest.mark.asyncio
async def test_ask_user_pauses_and_the_answer_resumes_the_same_session(rig):
    harness, _, _ = rig
    task = await harness.submit("天気調べて", metadata={"goal": "SCENARIO:ask weather"})
    t = await finish(harness, task.task_id)
    assert t.status == "needs_input" and t.question == "Which city?"
    await harness.answer(task.task_id, "Osaka")
    t = await finish(harness, task.task_id)
    assert t.status == "completed" and "Which city?" in t.summary and "Osaka" in t.summary
    assert t.question is None and "pending_message" not in t.metadata


@pytest.mark.asyncio
async def test_permission_grant_resumes_with_a_wider_sandbox_and_a_denial_does_not(rig):
    harness, _, _ = rig
    for approve, roots in ((True, ["/tmp/elsewhere"]), (False, [])):
        task = await harness.submit("x", metadata={"goal": "SCENARIO:perm write elsewhere"})
        t = await finish(harness, task.task_id)
        assert t.status == "needs_permission" and t.metadata["permission"]["what"] == "write /tmp/elsewhere/x.txt"
        await harness.grant(task.task_id, approve)
        t = await finish(harness, task.task_id)
        assert t.status == "completed" and json.loads(t.metadata["result"]["details"]) == roots
        assert ("granted" if approve else "denied") in t.summary


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario,error", [("exit", "without submitting"), ("crash", "exited with 3: boom"),
                                            ("sleep", "longer than 2 s")])
async def test_no_result_crash_and_timeout_all_fail_never_complete(rig, scenario, error):
    harness, _, _ = rig
    task = await harness.submit("x", metadata={"goal": f"SCENARIO:{scenario}"})
    t = await finish(harness, task.task_id, timeout=6)
    assert t.status == "failed" and error in t.error


@pytest.mark.asyncio
async def test_cancel_kills_the_process_and_a_follow_up_resumes(rig):
    harness, _, _ = rig
    task = await harness.submit("x", metadata={"goal": "SCENARIO:sleep"})
    await asyncio.sleep(0.5)
    await harness.cancel(task.task_id)
    await asyncio.sleep(0.1)
    assert harness.registry.get(task.task_id).status == "cancelled"
    done = await harness.submit("x", metadata={"goal": "SCENARIO:done"})
    await finish(harness, done.task_id)
    await harness.follow_up(done.task_id, "also check Osaka")
    t = await finish(harness, done.task_id)
    assert t.status == "completed" and "also check Osaka" in t.summary


@pytest.mark.asyncio
async def test_a_sub_agent_can_only_reach_its_own_task(rig):
    harness, tools, _ = rig
    a = await harness.submit("a", metadata={"goal": "SCENARIO:sleep"})
    b = await harness.submit("b", metadata={"goal": "SCENARIO:sleep"})
    url_a, url_b = tools.url_for(a.task_id), tools.url_for(b.task_id)
    forged = url_a.rsplit("/", 2)[0] + "/" + b.task_id + "/" + url_a.rsplit("/", 1)[1]
    async with httpx.AsyncClient(trust_env=False) as c:
        assert (await c.post(forged, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})).status_code == 404
        assert (await c.post(url_b, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})).status_code == 200
    for t in (a, b):
        await harness.cancel(t.task_id)
