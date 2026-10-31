"""A dependency-free MCP stdio server for the v0.1 harness.

This implements the small MCP surface needed by the demo: initialize,
tools/list, and tools/call. It keeps the harness usable before we commit to a
particular MCP SDK version, while remaining straightforward to replace with
the official SDK later.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Iterable
from typing import Any

from .core import Harness
from .router import route_text

PROTOCOL_VERSION = "2025-06-18"

TOOLS = [
    {
        "name": "task_submit",
        "description": "Submit one heavy task to the harness.",
        "inputSchema": {
            "type": "object",
            "required": ["request"],
            "properties": {
                "request": {"type": "string"},
                "route": {"type": "string", "enum": ["task", "computer_use"]},
                "owner_id": {"type": "string"},
                "confirmed": {"type": "boolean"},
            },
        },
    },
    {
        "name": "task_get",
        "description": "Read the current state of one task.",
        "inputSchema": {"type": "object", "required": ["task_id"], "properties": {"task_id": {"type": "string"}}},
    },
    {
        "name": "task_list",
        "description": "List task records.",
        "inputSchema": {"type": "object", "properties": {"status": {"type": "string"}, "owner_id": {"type": "string"}}},
    },
    {
        "name": "task_cancel",
        "description": "Cancel a running or waiting task.",
        "inputSchema": {"type": "object", "required": ["task_id"], "properties": {"task_id": {"type": "string"}}},
    },
    {
        "name": "task_confirm",
        "description": "Confirm a computer-use task and start it.",
        "inputSchema": {"type": "object", "required": ["task_id"], "properties": {"task_id": {"type": "string"}}},
    },
    {
        "name": "event_publish",
        "description": "Publish a reporter or system event to the harness inbox.",
        "inputSchema": {
            "type": "object",
            "required": ["event_type", "source", "text"],
            "properties": {"event_type": {"type": "string"}, "source": {"type": "string"}, "text": {"type": "string"}},
        },
    },
    {
        "name": "event_list",
        "description": "Read recent events from the harness inbox.",
        "inputSchema": {"type": "object", "properties": {"task_id": {"type": "string"}, "limit": {"type": "integer"}}},
    },
]


def _result(value: Any) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}],
        "structuredContent": value,
    }


async def call_tool(name: str, args: dict[str, Any], harness: Harness) -> dict[str, Any]:
    if name == "task_submit":
        request = str(args.get("request", "")).strip()
        decision = route_text(request)
        route = str(args.get("route") or ("computer_use" if decision.route == "computer_use" else "task"))
        task = await harness.submit(
            request,
            route=route,  # type: ignore[arg-type]
            owner_id=str(args.get("owner_id", "default")),
            worker_kind="computer-use" if route == "computer_use" else "demo",
            requires_confirmation=route == "computer_use",
            confirmed=bool(args.get("confirmed", False)),
        )
        return task.to_dict()
    if name == "task_get":
        task = harness.registry.get(str(args["task_id"]))
        return task.to_dict() if task else {"error": "task not found"}
    if name == "task_list":
        return {"tasks": [task.to_dict() for task in harness.registry.list(args.get("status"), args.get("owner_id"))]}
    if name == "task_cancel":
        return (await harness.cancel(str(args["task_id"]))).to_dict()
    if name == "task_confirm":
        return (await harness.confirm(str(args["task_id"]))).to_dict()
    if name == "event_publish":
        event = harness.publish_event(
            str(args["event_type"]),
            str(args["source"]),
            {"text": str(args["text"])},
            dedupe_key=args.get("dedupe_key"),
            priority=str(args.get("priority", "normal")),
        )
        return event.to_dict()
    if name == "event_list":
        return {"events": [event.to_dict() for event in harness.inbox.list(args.get("task_id"), int(args.get("limit", 100)))]}
    raise ValueError(f"unknown tool {name}")


async def dispatch(message: dict[str, Any], harness: Harness) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "maid-harness", "version": "0.1.0"},
                "instructions": "The Maid owns conversation; use these tools for supervised task and event state.",
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        try:
            params = message.get("params") or {}
            value = await call_tool(str(params.get("name")), dict(params.get("arguments") or {}), harness)
            return {"jsonrpc": "2.0", "id": request_id, "result": _result(value)}
        except Exception as exc:  # noqa: BLE001 - return JSON-RPC errors to MCP clients
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32602, "message": f"{type(exc).__name__}: {exc}"},
            }
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"method not found: {method}"}}


async def run_stdio(harness: Harness | None = None) -> None:
    service = harness or Harness()
    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            break
        try:
            message = json.loads(line)
            response = await dispatch(message, service)
            if response is not None:
                sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                sys.stdout.flush()
        except json.JSONDecodeError as exc:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}) + "\n")
            sys.stdout.flush()
