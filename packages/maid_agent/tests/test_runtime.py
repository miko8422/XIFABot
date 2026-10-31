import asyncio
import time

import pytest
from maid_harness import Harness

from maid_agent.router import benchmark, route_text
from maid_agent.runtime import MaidRuntime


def test_router_is_fast_and_has_the_expected_boundaries():
    assert route_text("你好，今天怎么样").route == "direct"
    assert route_text("搜索一下本地报告").route == "task"
    assert route_text("打开 Steam").route == "computer_use"
    assert route_text("このプロジェクトを検索して").route == "task"
    assert route_text("Steamを開いて").route == "computer_use"
    assert route_text("现在几点").route == "reporter_query"
    measured = benchmark(500)
    assert measured["p95_ms"] < 5.0


@pytest.mark.asyncio
async def test_maid_submits_without_waiting_for_worker_and_reports_completion():
    harness = Harness(worker_delay_s=0.08)
    maid = MaidRuntime(harness)
    started = time.perf_counter()
    reply = await maid.handle_text("搜索一下这个项目")
    elapsed = time.perf_counter() - started
    assert reply.task_id and reply.route.route == "task"
    assert elapsed < 0.1
    task = await harness.wait(reply.task_id, timeout=2)
    assert task.status == "completed" and task.summary
    notification = maid.poll_notification()
    assert notification and "終わった" in notification.text


@pytest.mark.asyncio
async def test_computer_use_waits_for_confirmation():
    harness = Harness(worker_delay_s=0.001)
    maid = MaidRuntime(harness)
    reply = await maid.handle_text("打开 Steam")
    assert reply.needs_confirmation and reply.task_id
    assert harness.registry.get(reply.task_id).status == "waiting_confirmation"
    await maid.confirm(reply.task_id)
    task = await harness.wait(reply.task_id, timeout=1)
    assert task.status == "completed"


@pytest.mark.asyncio
async def test_multiple_tasks_merge_into_one_maid_notification():
    harness = Harness(worker_delay_s=0.001)
    maid = MaidRuntime(harness)
    first = await maid.handle_text("搜索第一个东西")
    second = await maid.handle_text("总结第二个东西")
    await asyncio.gather(harness.wait(first.task_id, 1), harness.wait(second.task_id, 1))
    notification = maid.poll_notification()
    assert notification and notification.text.count("終わった") == 2
    assert "2件の作業" not in notification.text  # completed summaries remain individually attributable
