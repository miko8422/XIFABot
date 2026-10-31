"""The processes a sub-agent run is made of (P3.5). Neither needs the Maid to be running.

  python -m maid_harness.subagent run --db … --task … --root … --home … --codex … --profile '{…}'
      The per-run supervisor: builds the clean CODEX_HOME, starts Codex with a stripped environment, saves every
      event to runs/tasks/<id>/events.jsonl, heartbeats into the registry, enforces the timeout, and settles the
      run (result / pause / failure) in the shared SQLite file. SIGTERM from the Maid's cancel ends it quietly.

  python -m maid_harness.subagent mcp --db … --task …
      The sub-agent's four tools over stdio (subagent_mcp.py), started by Codex itself.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .codex_home import build_codex_home
from .codex_runner import BASE_FILESYSTEM, Profile, codex_argv, command, describe_item, log_run, subagent_env
from .core import Harness

HEARTBEAT_S = 10.0


def run(db: str, task_id: str, root: Path, home: Path, codex: str, profile: Profile) -> int:
    harness = Harness(db, event_source="subagent")
    reg = harness.registry
    task = reg.get(task_id)
    if task is None or task.status != "running":
        return 0
    workspace = root / task_id
    workspace.mkdir(parents=True, exist_ok=True)
    fingerprint = build_codex_home(home, profile.model, profile.effort, permissions={"filesystem": BASE_FILESYSTEM})
    mcp_cmd = [sys.executable, "-m", "maid_harness.subagent", "mcp", "--db", db, "--task", task_id]
    cmd = command(task, workspace, profile, codex_argv(codex), mcp_cmd)
    log_run(workspace, {"profile": profile.name, "codex": fingerprint, "resume": "resume" in cmd[:4],
                        "grants": task.metadata.get("grants", [])})
    reg.update(task_id, meta={"codex": fingerprint, "alive_at": time.time()})

    stopping = threading.Event()
    timed_out = threading.Event()
    with open(workspace / "stderr.log", "ab") as stderr:
        proc = subprocess.Popen(cmd, cwd=workspace, env=subagent_env(home), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=stderr)

    def on_term(signum, frame):  # the Maid cancelled: it already wrote `cancelled`
        stopping.set()
        proc.kill()

    signal.signal(signal.SIGTERM, on_term)

    def heartbeat():
        while not stopping.wait(HEARTBEAT_S):
            reg.update(task_id, meta={"alive_at": time.time()})

    def deadline():
        if not stopping.wait(profile.timeout_s):
            timed_out.set()
            proc.kill()

    threading.Thread(target=heartbeat, daemon=True).start()
    threading.Thread(target=deadline, daemon=True).start()
    with open(workspace / "events.jsonl", "ab") as events:
        for line in proc.stdout:
            events.write(line)
            events.flush()
            try:
                on_event(harness, task_id, json.loads(line))
            except (json.JSONDecodeError, KeyError):
                continue
    code = proc.wait()
    cancelled = stopping.is_set()
    stopping.set()
    if cancelled:
        return 0
    meta = reg.get(task_id).metadata
    if timed_out.is_set():
        harness.settle(task_id, error=f"the sub-agent took longer than {int(profile.timeout_s)} s")
    elif code != 0 and not meta.get("result") and not meta.get("pause"):
        tail = (workspace / "stderr.log").read_bytes()[-400:].decode("utf-8", "replace").strip()
        harness.settle(task_id, error=f"codex exited with {code}: {tail.splitlines()[-1] if tail else 'no output'}")
    else:
        harness.settle(task_id)
    return 0


def on_event(harness: Harness, task_id: str, event: dict) -> None:
    reg = harness.registry
    kind = event.get("type")
    if kind == "thread.started" and event.get("thread_id"):
        reg.update(task_id, meta={"thread_id": event["thread_id"]})
    elif kind == "turn.completed" and event.get("usage"):
        usage = [*reg.get(task_id).metadata.get("usage", []), {**event["usage"], "t": time.time()}]
        reg.update(task_id, meta={"usage": usage})
    elif kind == "item.completed":
        item = event.get("item") or {}
        if item.get("type") == "agent_message" and item.get("text"):
            reg.update(task_id, meta={"last_message": item["text"][-2000:]})
        elif text := describe_item(item):
            reg.update(task_id, progress_text=text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m maid_harness.subagent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--db", required=True)
    r.add_argument("--task", required=True)
    r.add_argument("--root", required=True, type=Path)
    r.add_argument("--home", required=True, type=Path)
    r.add_argument("--codex", default="codex")
    r.add_argument("--profile", required=True)
    m = sub.add_parser("mcp")
    m.add_argument("--db", required=True)
    m.add_argument("--task", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "mcp":
        from .subagent_mcp import serve_stdio
        serve_stdio(args.db, args.task)
        return 0
    return run(args.db, args.task, args.root, args.home, args.codex, Profile(**json.loads(args.profile)))


if __name__ == "__main__":
    sys.exit(main())
