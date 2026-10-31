"""The four tools every sub-agent gets (ARCHITECTURE §5.2), as a streamable-HTTP MCP server in the Maid's process.

Each task gets its own URL, /mcp/<task_id>/<token>, passed to Codex with `-c mcp_servers.maid.url=...`.
The token is random per task, so a sub-agent can only ever touch its own task. Every tool returns at once:
ask_user and request_permission only record the pause, and the agent is told to end its turn. The user's
answer arrives later as the next message of the same session (Harness.answer / grant).
"""

from __future__ import annotations

import json
import secrets
from typing import Any

from aiohttp import web

from .core import Harness

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
     "description": ("Ask for something your sandbox does not allow: writing outside your workspace, network "
                     "access, or a command that needs full access. After calling it, end your turn."),
     "inputSchema": {"type": "object", "required": ["what", "why"], "properties": {
         "kind": {"type": "string", "enum": ["write_path", "network", "full_access"]},
         "target": {"type": "string", "description": "The path for write_path."},
         "what": {"type": "string", "description": "What you want to do, in plain English."},
         "why": {"type": "string"}}}},
    {"name": "submit_result",
     "description": ("Finish the task. This is the only way a task completes. user_summary is read aloud to the "
                     "user: 1-3 plain English sentences with the actual answer or outcome, no markdown."),
     "inputSchema": {"type": "object", "required": ["user_summary"], "properties": {
         "user_summary": {"type": "string"},
         "details": {"type": "string", "description": "Anything longer: findings, links, what changed."},
         "artifacts": {"type": "array", "items": {"type": "string"}, "description": "Paths of files you produced."}}}},
]


def _text(value: Any) -> dict:
    return {"content": [{"type": "text", "text": value if isinstance(value, str) else json.dumps(value)}]}


class SubAgentTools:
    def __init__(self, harness: Harness, host: str = "127.0.0.1", port: int = 0) -> None:
        self.harness = harness
        self.host, self.port = host, port
        self._tokens: dict[str, str] = {}
        self._runner: web.AppRunner | None = None

    # ---- lifecycle --------------------------------------------------------------
    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/mcp/{task_id}/{token}", self._handle)
        app.router.add_get("/mcp/{task_id}/{token}", self._no_stream)
        app.router.add_delete("/mcp/{task_id}/{token}", self._end_session)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]  # port 0 -> the one the OS picked

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    def url_for(self, task_id: str) -> str:
        token = self._tokens.setdefault(task_id, secrets.token_urlsafe(16))
        return f"http://{self.host}:{self.port}/mcp/{task_id}/{token}"

    @staticmethod
    async def _no_stream(request: web.Request) -> web.Response:
        return web.Response(status=405)  # no server-sent event stream: every reply is a plain JSON response

    @staticmethod
    async def _end_session(request: web.Request) -> web.Response:
        return web.Response(status=200)

    # ---- JSON-RPC over HTTP -------------------------------------------------------------
    async def _handle(self, request: web.Request) -> web.Response:
        task_id, token = request.match_info["task_id"], request.match_info["token"]
        if not secrets.compare_digest(self._tokens.get(task_id, ""), token):
            return web.Response(status=404)
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
        messages = body if isinstance(body, list) else [body]
        replies = [r for m in messages if (r := await self.dispatch(task_id, m)) is not None]
        if not replies:
            return web.Response(status=202)  # notifications only
        return web.json_response(replies if isinstance(body, list) else replies[0])

    async def dispatch(self, task_id: str, message: dict) -> dict | None:
        method, rid = message.get("method"), message.get("id")
        if rid is None:  # a notification
            return None
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": (message.get("params") or {}).get("protocolVersion", PROTOCOL_VERSION),
                "capabilities": {"tools": {}}, "serverInfo": {"name": "maid", "version": "0.3.0"},
                "instructions": "Report to the Maid, who speaks to the user. Always finish with submit_result."}}
        if method == "ping":
            return {"jsonrpc": "2.0", "id": rid, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
        if method == "tools/call":
            params = message.get("params") or {}
            try:
                value = self.call(task_id, str(params.get("name")), dict(params.get("arguments") or {}))
                return {"jsonrpc": "2.0", "id": rid, "result": _text(value)}
            except Exception as e:  # noqa: BLE001 - a bad call is the agent's problem, reported back to it
                return {"jsonrpc": "2.0", "id": rid, "result": {**_text(f"error: {e}"), "isError": True}}
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"method not found: {method}"}}

    def call(self, task_id: str, name: str, a: dict) -> str:
        h = self.harness
        if name == "report_progress":
            h.sub_progress(task_id, a.get("percent"), str(a.get("text", "")))
            return "ok"
        if name == "ask_user":
            h.sub_pause(task_id, "needs_input", question=str(a["question"]))
            return PAUSE_REPLY
        if name == "request_permission":
            h.sub_pause(task_id, "needs_permission", kind=a.get("kind") or "full_access", target=a.get("target"),
                        what=str(a["what"]), why=str(a.get("why", "")))
            return PAUSE_REPLY
        if name == "submit_result":
            h.sub_result(task_id, str(a["user_summary"]), str(a.get("details", "")), a.get("artifacts"))
            return "Result recorded. You are done; end your turn."
        raise ValueError(f"unknown tool {name}")
