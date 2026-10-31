"""Codex as the sub-agent runner (ARCHITECTURE §5.2, P3 + P3.5).

  Maid ── CodexRunner.spawn() ──▶ python -m maid_harness.subagent run …   (detached: outlives the Maid)
                                     └─ codex exec --json …                 (CODEX_HOME = a clean home, codex_home.py)
                                          └─ python -m maid_harness.subagent mcp …   (its four tools, stdio)

  first run   codex exec --json -C runs/tasks/<id> <brief + goal>          -> thread.started gives the session id
  resume      codex exec resume --json <thread_id> <answer / grant / follow-up>   (cwd = the task workspace)

Sandbox (checked against codex-cli 0.165.0, 2026-10-31): a filesystem permission profile, not sandbox_mode.
The sub-agent can read the system basics and read/write its own workspace and temp dir; everything else,
reads included, needs a grant. Grants are passed per run as one inline TOML table (`-c` cannot spell a
path as a dotted key). Shell network is off unless granted. Web search (quick tier) is Codex's own tool.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .models import TaskRecord
from .workers import DetachedWorker

BRIEF = """You are a background sub-agent working for Natsume, a voice assistant. She hands you one task and
reads your result aloud to the user in Japanese. The user cannot see your output, only what you submit.

Rules:
- Work in English. Use the `maid` MCP tools to talk to the harness.
- Call report_progress at real milestones (a few times, not every step).
- If you cannot continue without the user, call ask_user with one short question and end your turn.
- You may read and write only inside your workspace. To read or write anything outside it, or to use the
  network from the shell, call request_permission first (kind read_path / write_path with the absolute path,
  or network) and end your turn. Never try to get around the sandbox. If something is blocked, ask.
- Finish by calling submit_result. user_summary is spoken aloud: 1-3 plain sentences with the actual answer
  or outcome (numbers, names, what changed), no markdown, no URLs. Put everything the user may ask about next
  (all options with their numbers, sources, details) in details.
- If you could not do it, still call submit_result and say plainly what failed. Never claim what you did not do.
"""

BASE_FILESYSTEM = {":minimal": "read", ":workspace_roots": "write", ":tmpdir": "write"}
# what a sub-agent's shell may see of the Maid's environment: never her .env secrets (API keys)
ENV_KEEP = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR",
            "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy", "SSL_CERT_FILE")


@dataclass
class Profile:
    name: str
    effort: str = "low"                # model_reasoning_effort
    search: bool = False               # Codex's live web search tool
    network: bool = False              # network for shell commands
    timeout_s: float = 300.0
    model: str = ""                    # "" = whatever the user's ~/.codex uses
    extra: list[str] = field(default_factory=list)  # more -c key=value overrides


PROFILES = {
    "quick": Profile("quick", "low", search=True, timeout_s=300),
    "work": Profile("work", "low", timeout_s=1800),
}


def _toml(value) -> str:
    return json.dumps(value)  # JSON strings, lists and booleans are valid TOML literals


def filesystem_table(grants: list[dict]) -> dict:
    fs = dict(BASE_FILESYSTEM)
    for g in grants:
        kind, target = g.get("kind"), g.get("target")
        if kind == "full_access":
            fs["/"] = "write"
        elif kind in ("read_path", "write_path") and target:
            path = str(Path(str(target)).expanduser())
            if fs.get(path) != "write":
                fs[path] = "write" if kind == "write_path" else "read"
    return fs


def config_overrides(profile: Profile, mcp_cmd: list[str], grants: list[dict]) -> list[str]:
    """-c overrides for one run. Grants from request_permission widen the sandbox for this task only."""
    network = profile.network or any(g.get("kind") in ("network", "full_access") for g in grants)
    fs = filesystem_table(grants)
    kv = {
        "mcp_servers.maid.command": mcp_cmd[0],
        "mcp_servers.maid.args": mcp_cmd[1:],
        # our four tools touch only this task; "prompt" (the default) hangs an unattended exec forever
        "mcp_servers.maid.default_tools_approval_mode": "approve",
        "approval_policy": "never",           # exec is unattended; escalation goes through request_permission
        "model_reasoning_effort": profile.effort,
        "web_search": "live" if profile.search else "disabled",
        "permissions.maid.network.enabled": network,
    }
    if profile.model:
        kv["model"] = profile.model
    out = []
    for k, v in kv.items():
        out += ["-c", f"{k}={_toml(v)}"]
    out += ["-c", "permissions.maid.filesystem={" + ", ".join(f"{_toml(k)}={_toml(v)}" for k, v in fs.items()) + "}"]
    for extra in profile.extra:
        out += ["-c", extra]
    return out


def command(task: TaskRecord, workspace: Path, profile: Profile, codex: list[str], mcp_cmd: list[str]) -> list[str]:
    meta = task.metadata
    cfg = config_overrides(profile, mcp_cmd, meta.get("grants", []))
    thread, message = meta.get("thread_id"), meta.get("pending_message")
    if thread and message:
        return [*codex, "exec", "resume", "--json", "--skip-git-repo-check", *cfg, thread, message]
    prompt = (f"{BRIEF}\nTask: {meta.get('goal') or task.request}\n"
              f"The user's own words (Japanese speech recognition, may contain mistakes): {task.request}\n"
              f"Your workspace: {workspace}")
    return [*codex, "exec", "--json", "--skip-git-repo-check", "-C", str(workspace), *cfg, prompt]


def subagent_env(home: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k in ENV_KEEP}
    env["CODEX_HOME"] = str(home)
    return env


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


class CodexRunner(DetachedWorker):
    """Starts the detached supervisor for one run of one task."""

    def __init__(self, db: Path, root: Path, home: Path, profile: Profile, codex: str = "codex") -> None:
        self.db, self.root, self.home, self.profile, self.codex = db, root, home, profile, codex

    def spawn(self, task: TaskRecord) -> subprocess.Popen:
        workspace = self.root / task.task_id
        workspace.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, "-m", "maid_harness.subagent", "run", "--db", str(self.db), "--task", task.task_id,
                "--root", str(self.root), "--home", str(self.home), "--codex", self.codex,
                "--profile", json.dumps(asdict(self.profile))]
        log = open(workspace / "supervisor.log", "ab")
        try:
            return subprocess.Popen(argv, cwd=workspace, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                    start_new_session=True)  # its own group: survives the Maid, killed as one
        finally:
            log.close()


def codex_argv(codex: str) -> list[str]:
    return shlex.split(codex)


def log_run(workspace: Path, record: dict) -> None:
    with open(workspace / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.time(), **record}, ensure_ascii=False) + "\n")


KEPT_FILES = {"events.jsonl", "runs.jsonl", "stderr.log", "supervisor.log"}  # the dataset; the rest is work


def prune_workspaces(root: Path, registry, retention_days: float) -> int:
    """Delete the files sub-agents made in workspaces of tasks finished more than `retention_days` ago.
    Their event logs stay (the task dataset). Returns how many workspaces were pruned."""
    import shutil

    cutoff, pruned = time.time() - retention_days * 86400, 0
    for task in registry.list():
        ws = root / task.task_id
        if task.status in ("accepted", "running", "needs_input", "needs_permission", "waiting_confirmation") \
                or task.updated_at > cutoff or task.metadata.get("pruned") or not ws.is_dir():
            continue
        for entry in ws.iterdir():
            if entry.name in KEPT_FILES:
                continue
            shutil.rmtree(entry) if entry.is_dir() and not entry.is_symlink() else entry.unlink()
        registry.update(task.task_id, meta={"pruned": time.time()})
        pruned += 1
    return pruned
