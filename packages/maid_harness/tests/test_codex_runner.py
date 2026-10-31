"""P3/P3.5: detached Codex sub-agents, driven by fake_codex.py through the real supervisor and stdio MCP
processes (no network, no model). Also the clean CODEX_HOME and how it follows ~/.codex."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

from maid_harness import Harness
from maid_harness.codex_home import build_codex_home, fingerprint
from maid_harness.codex_runner import CodexRunner, Profile, config_overrides, filesystem_table
from maid_harness.core import pid_alive

FAKE = Path(__file__).with_name("fake_codex.py")


@pytest.fixture
def user_codex(tmp_path, monkeypatch):
    """A stand-in for the user's ~/.codex, so tests never touch the real one."""
    src = tmp_path / "user_codex"
    src.mkdir()
    (src / "config.toml").write_text('model_provider = "OpenAI"\nmodel = "gpt-big"\nnotify = ["x"]\n\n'
                                     '[mcp_servers.heavy]\ncommand = "x"\n\n[model_providers.OpenAI]\nname = "OpenAI"\nbase_url = "https://a.example"\n')
    (src / "auth.json").write_text(json.dumps({"tokens": {"account_id": "acct-1"}}))
    monkeypatch.setenv("MAID_USER_CODEX_HOME", str(src))
    monkeypatch.setenv("API_KEY", "sk-secret-from-dotenv")  # what the Maid's .env puts in her environment
    return src


def make(tmp_path, max_running=2, timeout_s=3.0):
    db = tmp_path / "tasks.sqlite3"
    root, home = tmp_path / "tasks", tmp_path / "codex_home"
    profile = Profile("quick", timeout_s=timeout_s, model="gpt-5.6-luna")
    runner = lambda t: CodexRunner(db, root, home, profile, codex=f"{sys.executable} {FAKE}")
    return Harness(db, worker_factory=runner, max_running=max_running), root, home


async def until(harness, task_id, *states, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        harness.pump()
        t = harness.registry.get(task_id)
        if t.status in states:
            return t
        await asyncio.sleep(0.05)
    raise AssertionError(f"{task_id} is {harness.registry.get(task_id).status}, wanted {states}")


# ---- config ------------------------------------------------------------------------------
def test_reads_and_writes_outside_the_workspace_need_grants():
    fs = filesystem_table([])
    assert fs == {":minimal": "read", ":workspace_roots": "write", ":tmpdir": "write"}
    fs = filesystem_table([{"kind": "read_path", "target": "/repo"}, {"kind": "write_path", "target": "/out"},
                           {"kind": "read_path", "target": "/out"}])
    assert fs["/repo"] == "read" and fs["/out"] == "write", "a read grant never downgrades a write grant"
    assert filesystem_table([{"kind": "full_access"}])["/"] == "write"
    args = config_overrides(Profile("quick", search=True), ["py", "-m", "x"], [{"kind": "network"}])
    kv = dict(args[i + 1].split("=", 1) for i in range(0, len(args), 2))
    assert kv["web_search"] == '"live"' and kv["permissions.maid.network.enabled"] == "true"
    assert kv["mcp_servers.maid.default_tools_approval_mode"] == '"approve"'
    assert kv["permissions.maid.filesystem"].startswith('{":minimal"="read"')


def test_clean_home_keeps_only_provider_and_login_and_follows_a_switch(tmp_path, user_codex):
    home = tmp_path / "home"
    fp = build_codex_home(home, "gpt-5.6-luna", "low")
    cfg = (home / "config.toml").read_text()
    assert 'model = "gpt-5.6-luna"' in cfg and "a.example" in cfg
    assert "mcp_servers" not in cfg and "notify" not in cfg and "gpt-big" not in cfg
    assert (home / "auth.json").is_symlink() and fp["endpoint"] == "a.example"
    # cc-switch style: the user's files are rewritten in place -> the next build follows them
    (user_codex / "config.toml").write_text('model_provider = "OpenAI"\n[model_providers.OpenAI]\nname = "OpenAI"\nbase_url = "https://b.example"\n')
    (user_codex / "auth.json").write_text(json.dumps({"tokens": {"account_id": "acct-2"}}))
    fp2 = build_codex_home(home, "gpt-5.6-luna", "low")
    assert fp2["endpoint"] == "b.example" and fp2["account"] != fp["account"]
    assert "acct" not in json.dumps(fp2), "the fingerprint never contains the account itself"


# ---- runs ---------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_detached_run_completes_with_details_and_never_sees_the_maids_secrets(tmp_path, user_codex):
    harness, root, home = make(tmp_path)
    task = await harness.submit("東京の天気", metadata={"goal": "SCENARIO:done weather"})
    t = await until(harness, task.task_id, "completed")
    assert t.summary == "It is 19 degrees in Tokyo." and "1980 yen" in t.metadata["result"]["details"]
    assert t.metadata["thread_id"] == "thread-123" and t.metadata["usage"] and t.metadata["codex"]["model"] == "gpt-5.6-luna"
    env = json.loads((root / task.task_id / "env.json").read_text())
    assert "API_KEY" not in env and env["CODEX_HOME"] == str(home)
    assert json.loads((root / task.task_id / "events.jsonl").read_text().splitlines()[0])["type"] == "thread.started"
    runs = [json.loads(line) for line in (root / task.task_id / "runs.jsonl").read_text().splitlines()]
    assert runs[0]["codex"]["endpoint"] == "a.example"


@pytest.mark.asyncio
async def test_ask_user_pauses_and_the_answer_resumes_the_same_session(tmp_path, user_codex):
    harness, _, _ = make(tmp_path)
    task = await harness.submit("天気調べて", metadata={"goal": "SCENARIO:ask"})
    t = await until(harness, task.task_id, "needs_input")
    assert t.question == "Which city?"
    await harness.answer(task.task_id, "Osaka")
    t = await until(harness, task.task_id, "completed")
    assert "Which city?" in t.summary and "Osaka" in t.summary and "pending_message" not in t.metadata


@pytest.mark.asyncio
async def test_a_read_grant_reaches_the_sandbox_and_a_denial_does_not(tmp_path, user_codex):
    harness, _, _ = make(tmp_path)
    for approve in (True, False):
        task = await harness.submit("x", metadata={"goal": "SCENARIO:perm"})
        t = await until(harness, task.task_id, "needs_permission")
        assert t.metadata["permission"] == {"what": "read /tmp/elsewhere", "why": "you asked for it",
                                            "kind": "read_path", "target": "/tmp/elsewhere"}
        await harness.grant(task.task_id, approve)
        t = await until(harness, task.task_id, "completed")
        assert ('"/tmp/elsewhere"="read"' in t.metadata["result"]["details"]) is approve
        assert ("granted" if approve else "denied") in t.summary


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario,error", [("exit", "without submitting"), ("crash", "exited with 3: boom"),
                                            ("sleep", "longer than 3 s")])
async def test_no_result_crash_and_timeout_all_fail_never_complete(tmp_path, user_codex, scenario, error):
    harness, _, _ = make(tmp_path)
    task = await harness.submit("x", metadata={"goal": f"SCENARIO:{scenario}"})
    t = await until(harness, task.task_id, "failed", timeout=10)
    assert error in t.error


@pytest.mark.asyncio
async def test_cancel_kills_the_whole_process_group_and_a_cancelled_task_can_be_restarted(tmp_path, user_codex):
    harness, _, _ = make(tmp_path)
    task = await harness.submit("x", metadata={"goal": "SCENARIO:sleep"})
    t = await until(harness, task.task_id, "running")
    await asyncio.sleep(0.5)
    pid = harness.registry.get(task.task_id).metadata["pid"]
    await harness.cancel(task.task_id)
    for _ in range(50):
        harness.check_supervisors()
        if not pid_alive(pid):
            break
        await asyncio.sleep(0.1)
    assert not pid_alive(pid) and harness.registry.get(task.task_id).status == "cancelled"
    await harness.follow_up(task.task_id, "start again please")
    t = await until(harness, task.task_id, "completed")
    assert "start again please" in t.summary


# ---- the Maid going away -------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_sub_agent_outlives_the_maid_and_a_new_maid_picks_up_its_result(tmp_path, user_codex):
    first, _, _ = make(tmp_path)
    task = await first.submit("x", metadata={"goal": "SCENARIO:slow"})
    await until(first, task.task_id, "running")
    first.pump()  # the first Maid has seen everything up to here, then dies without cleaning up
    del first

    second, _, _ = make(tmp_path)
    heard = []
    second.inbox.subscribe(lambda e: heard.append(e.type))
    assert second.recover() == [] and second.registry.get(task.task_id).status == "running"
    t = await until(second, task.task_id, "completed")
    assert t.summary == "done while you were away" and "task.completed" in heard


@pytest.mark.asyncio
async def test_a_result_that_arrived_while_no_maid_was_running_is_still_told(tmp_path, user_codex):
    first, _, _ = make(tmp_path)
    first.pump()  # saves her cursor
    task = await first.submit("x", metadata={"goal": "SCENARIO:done"})
    del first
    for _ in range(100):  # nobody is watching while it finishes
        if Harness(tmp_path / "tasks.sqlite3").registry.get(task.task_id).status == "completed":
            break
        await asyncio.sleep(0.05)
    second, _, _ = make(tmp_path)
    heard = []
    second.inbox.subscribe(lambda e: heard.append(e.type))
    second.recover()
    second.pump()
    assert "task.completed" in heard


@pytest.mark.asyncio
async def test_at_most_max_running_sub_agents_the_rest_wait_in_line(tmp_path, user_codex):
    harness, _, _ = make(tmp_path, max_running=1)
    a = await harness.submit("a", metadata={"goal": "SCENARIO:sleep"})
    b = await harness.submit("b", metadata={"goal": "SCENARIO:done"})
    await until(harness, a.task_id, "running")
    assert harness.registry.get(b.task_id).status == "accepted" and harness.registry.get(b.task_id).metadata["queued"]
    await harness.cancel(a.task_id)
    t = await until(harness, b.task_id, "completed")
    assert "queued" not in t.metadata


@pytest.mark.asyncio
async def test_questions_survive_a_restart_and_are_asked_again(tmp_path, user_codex):
    first, _, _ = make(tmp_path)
    task = await first.submit("x", metadata={"goal": "SCENARIO:ask"})
    await until(first, task.task_id, "needs_input")
    del first
    second, _, _ = make(tmp_path)
    heard = []
    second.inbox.subscribe(lambda e: heard.append((e.type, e.payload.get("question"))))
    second.recover()
    assert ("task.needs_input", "Which city?") in heard and second.registry.get(task.task_id).status == "needs_input"
