"""The four tools every sub-agent gets (ARCHITECTURE §5.2), as a stdio MCP server Codex starts per task.

  codex ──stdio──▶ python -m maid_harness.subagent mcp --db <tasks.sqlite3> --task <id> ──▶ registry

It writes to the shared SQLite registry directly, so it works with no Maid process running (P3.5), and it is
bound to one task by its arguments: a sub-agent can only ever touch its own task. Every tool returns at once:
ask_user and request_permission only record the pause, and the agent is told to end its turn.
"""

from __future__ import annotations

import json
import sys
from typing import Any

PROTOCOL_VERSION = "2025-06-18"
PAUSE_REPLY = "Recorded. End your turn now without further work; you will be resumed with the user's reply."

TOOLS = [
    {"name": "report_progress",
     "description": "Tell the user how far along you are. Call at real milestones, not every step.",
     "inputSchema": {"type": "object", "required": ["text"], "properties": {
         "percent": {"type": "integer", "minimum": 0, "maximum": 100},
         "text": {"type": "string", "description": "A few plain English words, e.g. 'reading the test output'."}}}},
    {"name": "ask_user",
     "description": ("Ask the user one short question when you cannot continue without the answer. "
                     "After calling it, end your turn; you are resumed with the answer."),
     "inputSchema": {"type": "object", "required": ["question"], "properties": {
         "question": {"type": "string", "description": "One plain English question, answerable by voice."}}}},
    {"name": "request_permission",
     "description": ("Ask for something your sandbox does not allow: reading or writing outside your workspace, "
                     "network for shell commands, or full access. After calling it, end your turn."),
     "inputSchema": {"type": "object", "required": ["kind", "what", "why"], "properties": {
         "kind": {"type": "string", "enum": ["read_path", "write_path", "network", "full_access"]},
         "target": {"type": "string", "description": "The absolute path, for read_path and write_path."},
         "what": {"type": "string", "description": "What you want to do, in plain English."},
         "why": {"type": "string"}}}},
    {"name": "submit_result",
     "description": ("Finish the task. This is the only way a task completes. user_summary is read aloud to the "
                     "user: 1-3 plain English sentences with the actual answer or outcome, no markdown."),
     "inputSchema": {"type": "object", "required": ["user_summary"], "properties": {
         "user_summary": {"type": "string"},
         "details": {"type": "string", "description": "Everything the user may ask about next: the full findings, "
                                                       "options with their numbers, links, what changed."},
         "artifacts": {"type": "array", "items": {"type": "string"}, "description": "Paths of files you produced."}}}},
]


def _text(value: Any) -> dict:
    return {"content": [{"type": "text", "text": value if isinstance(value, str) else json.dumps(value)}]}


def call_tool(harness, task_id: str, name: str, a: dict) -> str:
    if name == "report_progress":
        harness.sub_progress(task_id, a.get("percent"), str(a.get("text", "")))
        return "ok"
    if name == "ask_user":
        harness.sub_pause(task_id, "needs_input", question=str(a["question"]))
        return PAUSE_REPLY
    if name == "request_permission":
        harness.sub_pause(task_id, "needs_permission", kind=a.get("kind") or "full_access", target=a.get("target"),
                          what=str(a["what"]), why=str(a.get("why", "")))
        return PAUSE_REPLY
    if name == "submit_result":
        harness.sub_result(task_id, str(a["user_summary"]), str(a.get("details", "")), a.get("artifacts"))
        return "Result recorded. You are done; end your turn."
    raise ValueError(f"unknown tool {name}")


def dispatch(harness, task_id: str, message: dict) -> dict | None:
    method, rid = message.get("method"), message.get("id")
    if rid is None:  # a notification
        return None
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": (message.get("params") or {}).get("protocolVersion", PROTOCOL_VERSION),
            "capabilities": {"tools": {}}, "serverInfo": {"name": "maid", "version": "0.3.5"},
            "instructions": "Report to the Maid, who speaks to the user. Always finish with submit_result."}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = message.get("params") or {}
        try:
            value = call_tool(harness, task_id, str(params.get("name")), dict(params.get("arguments") or {}))
            return {"jsonrpc": "2.0", "id": rid, "result": _text(value)}
        except Exception as e:  # noqa: BLE001 - a bad call is the agent's problem, reported back to it
            return {"jsonrpc": "2.0", "id": rid, "result": {**_text(f"error: {e}"), "isError": True}}
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"method not found: {method}"}}


def serve_stdio(db: str, task_id: str) -> None:
    from .core import Harness

    harness = Harness(db, event_source="subagent")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            reply = dispatch(harness, task_id, json.loads(line))
        except json.JSONDecodeError as e:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()
