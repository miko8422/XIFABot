import pytest

from maid_harness import EventInbox, Harness, TaskRegistry
from maid_harness.mcp_server import dispatch


def test_event_inbox_deduplicates_and_expires():
    clock = [100.0]
    registry = TaskRegistry()
    inbox = EventInbox(registry, lambda: clock[0])
    first = inbox.publish("system.changed", "test", {"text": "a"}, dedupe_key="same", ttl_seconds=10)
    duplicate = inbox.publish("system.changed", "test", {"text": "a"}, dedupe_key="same", ttl_seconds=10)
    assert duplicate.event_id == first.event_id
    assert len(inbox.list()) == 1
    clock[0] = 111.0
    assert inbox.list() == []


@pytest.mark.asyncio
async def test_mcp_initialize_tools_and_task_call():
    harness = Harness(worker_delay_s=0.001)
    init = await dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}, harness)
    assert init["result"]["serverInfo"]["name"] == "maid-harness"
    tools = await dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, harness)
    assert {tool["name"] for tool in tools["result"]["tools"]} >= {"task_submit", "task_get", "event_publish"}
    result = await dispatch(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "task_submit", "arguments": {"request": "构建这个项目"}},
        },
        harness,
    )
    payload = result["result"]["structuredContent"]
    assert payload["status"] in {"accepted", "running"}
    await harness.wait(payload["task_id"], timeout=1)


@pytest.mark.asyncio
async def test_mcp_computer_use_route_is_explicit_and_waits_for_confirmation():
    harness = Harness(worker_delay_s=0.001)
    result = await dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "task_submit", "arguments": {"request": "Steamを開いて", "route": "computer_use"}},
        },
        harness,
    )
    assert result["result"]["structuredContent"]["status"] == "waiting_confirmation"
