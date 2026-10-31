"""Codex as the sub-agent runner (ARCHITECTURE §5.2, P3): one `codex exec` process per task run.

  first run   codex exec --json -C runs/tasks/<id> <brief + goal>          -> thread.started gives the session id
  resume      codex exec resume --json <thread_id> <answer / grant / follow-up>   (cwd = the task workspace)

The sub-agent reaches the harness only through its own MCP URL (subagent_mcp.py). Every JSON event is saved to
runs/tasks/<id>/events.jsonl. Cancel kills the process group, a timeout fails the task, and an exit without
submit_result fails it too (core.py decides; this only runs the process).
Flags checked against codex-cli 0.165.0 (2026-10-31): `resume` has no -s/-C, so the sandbox is set with -c.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

from .models import TaskRecord
from .subagent_mcp import SubAgentTools
from .workers import Progress, Worker

BRIEF = """You are a background sub-agent working for Natsume, a voice assistant. She hands you one task and
reads your result aloud to the user in Japanese. The user cannot see your output, only what you submit.

Rules:
- Work in English. Use the `maid` MCP tools to talk to the harness.
- Call report_progress at real milestones (a few times, not every step).
- If you cannot continue without the user, call ask_user with one short question and end your turn.
- If your sandbox blocks something you need (writing outside your workspace, network, full access), call
  request_permission and end your turn. Never try to get around the sandbox.
- Finish by calling submit_result. user_summary is spoken aloud: 1-3 plain sentences with the actual answer
  or outcome (numbers, names, what changed), no markdown, no URLs. Put longer material in details.
- If you could not do it, still call submit_result and say plainly what failed. Never claim what you did not do.
"""


@dataclass
class Profile:
    name: str
    sandbox: str = "read-only"         # read-only | workspace-write | danger-full-access
    effort: str = "low"                # model_reasoning_effort
    search: bool = False               # live web search
    network: bool = False              # network for shell commands (workspace-write only)
    timeout_s: float = 300.0
    model: str = ""                    # "" = the user's Codex default
    extra: list[str] = field(default_factory=list)  # more -c key=value overrides


PROFILES = {
    "quick": Profile("quick", "read-only", "low", search=True, timeout_s=300),
    "work": Profile("work", "workspace-write", "medium", timeout_s=1800),
}


def _toml(value) -> str:
    return json.dumps(value)  # JSON strings, lists and booleans are valid TOML literals


def config_overrides(profile: Profile, mcp_url: str, grants: list[dict]) -> list[str]:
    """-c overrides for one run. Grants from request_permission widen the sandbox for this task only."""
    sandbox, roots, network = profile.sandbox, [], profile.network
    for g in grants:
        kind = g.get("kind")
        if kind == "full_access":
            sandbox = "danger-full-access"
        elif kind in ("write_path", "network") and sandbox == "read-only":
            sandbox = "workspace-write"
        if kind == "write_path" and g.get("target"):
            roots.append(str(g["target"]))
        if kind == "network":
            network = True
    kv = {
        "mcp_servers.maid.url": mcp_url,
        # our four tools touch only this task; "prompt" (the default) hangs an unattended exec forever
        "mcp_servers.maid.default_tools_approval_mode": "approve",
        "sandbox_mode": sandbox,
        "approval_policy": "never",           # exec is unattended; escalation goes through request_permission
        "model_reasoning_effort": profile.effort,
        "notify": [],                         # don't fire the user's desktop turn-ended hook for sub-agents
        "web_search": "live" if profile.search else "disabled",
    }
    if sandbox == "workspace-write":
        kv["sandbox_workspace_write.network_access"] = network
        if roots:
            kv["sandbox_workspace_write.writable_roots"] = roots
    if profile.model:
        kv["model"] = profile.model
    out = []
    for k, v in kv.items():
        out += ["-c", f"{k}={_toml(v)}"]
    for extra in profile.extra:
        out += ["-c", extra]
    return out


def describe_item(item: dict) -> str | None:
    """A few words of progress from a Codex event, for 「どうなった？」 between report_progress calls."""
    kind = item.get("type")
    if kind == "command_execution":
        return f"running: {str(item.get('command', ''))[:80]}"
    if kind == "web_search":
        return f"searching the web: {str(item.get('query', ''))[:80]}"
    if kind == "file_change":
        return "editing files"
    return None


class CodexRunner(Worker):
    def __init__(self, harness, tools: SubAgentTools, root: Path, profile: Profile, codex: str = "codex") -> None:
        self.harness, self.tools, self.root, self.profile, self.codex = harness, tools, root, profile, codex

    def command(self, task: TaskRecord, workspace: Path) -> list[str]:
        meta = task.metadata
        cfg = config_overrides(self.profile, self.tools.url_for(task.task_id), meta.get("grants", []))
        thread, message = meta.get("thread_id"), meta.get("pending_message")
        if thread and message:
            return [self.codex, "exec", "resume", "--json", "--skip-git-repo-check", *cfg, thread, message]
        goal = meta.get("goal") or task.request
        prompt = (f"{BRIEF}\nTask: {goal}\n"
                  f"The user's own words (Japanese speech recognition, may contain mistakes): {task.request}\n"
                  f"Your workspace: {workspace} (sandbox: {self.profile.sandbox})")
        return [self.codex, "exec", "--json", "--skip-git-repo-check", "-C", str(workspace), *cfg, prompt]

    async def run(self, task: TaskRecord, report: Progress) -> None:
        workspace = self.root / task.task_id
        workspace.mkdir(parents=True, exist_ok=True)
        cmd = self.command(task, workspace)
        with open(workspace / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "profile": self.profile.name, "argv": cmd[:-1]}) + "\n")
        stderr = open(workspace / "stderr.log", "ab")
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=workspace, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=stderr, start_new_session=True, limit=16 * 1024 * 1024)
        try:
            await asyncio.wait_for(self._read(task.task_id, proc, workspace), self.profile.timeout_s)
            code = await proc.wait()
        except asyncio.TimeoutError:
            self._kill(proc)
            raise TimeoutError(f"the sub-agent took longer than {int(self.profile.timeout_s)} s") from None
        except asyncio.CancelledError:
            self._kill(proc)
            raise
        finally:
            stderr.close()
        current = self.harness.registry.get(task.task_id)
        meta = current.metadata if current else {}
        if code != 0 and not meta.get("result") and not meta.get("pause"):
            tail = (workspace / "stderr.log").read_bytes()[-400:].decode("utf-8", "replace").strip()
            raise RuntimeError(f"codex exited with {code}: {tail.splitlines()[-1] if tail else 'no output'}")
        return None  # the outcome is whatever the agent submitted through its tools

    async def _read(self, task_id: str, proc, workspace: Path) -> None:
        with open(workspace / "events.jsonl", "ab") as log:
            async for line in proc.stdout:
                log.write(line)
                log.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._on_event(task_id, event)

    def _on_event(self, task_id: str, event: dict) -> None:
        reg = self.harness.registry
        task = reg.get(task_id)
        if task is None:
            return
        kind = event.get("type")
        if kind == "thread.started" and event.get("thread_id"):
            reg.update(task_id, metadata={**task.metadata, "thread_id": event["thread_id"]})
        elif kind == "turn.completed" and event.get("usage"):
            usage = [*task.metadata.get("usage", []), event["usage"]]
            reg.update(task_id, metadata={**task.metadata, "usage": usage})
        elif kind == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message" and item.get("text"):
                reg.update(task_id, metadata={**task.metadata, "last_message": item["text"][-2000:]})
            elif (text := describe_item(item)) and task.status == "running":
                reg.update(task_id, progress_text=text)

    @staticmethod
    def _kill(proc) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # the whole tree: codex plus anything it started
        except ProcessLookupError:
            pass
