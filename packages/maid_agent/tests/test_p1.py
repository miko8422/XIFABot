"""P1: delegate routing, T1 state, speech arbiter, voice cancel, restart. Real controller + harness, fake models."""

import asyncio
import time
from pathlib import Path

import numpy as np
import pytest
from maid_harness import Harness
from maid_harness.workers import FakeSubAgent
from maid_voice.asr import FinalTranscript
from maid_voice.config import VoiceProfile
from maid_voice.controller import TurnController
from maid_voice.llm import LLMDelta, ToolCall
from maid_voice.metrics import now_ns

from maid_agent.arbiter import ArbiterConfig, SpeechArbiter
from maid_agent.router import route_text, wants_cancel, yes_no
from maid_agent.tools import state_block
from maid_agent.turns import MaidTurns


# ---- fakes ---------------------------------------------------------------------
class ASR:
    model = "fake-asr"

    def __init__(self):
        self.text = ""

    async def transcribe(self, pcm, sr):
        return FinalTranscript(self.text, "ja")


class LLM:
    """Scripted: next reply text and tool calls are set by the test."""
    model = "fake-llm"

    def __init__(self):
        self.text, self.calls, self.requests = "ん。", [], []

    async def stream_response(self, history, user_text, **kw):
        self.requests.append({"user_text": user_text, **kw})
        await asyncio.sleep(0.005)
        if self.text:
            yield LLMDelta(text=self.text)
        for c in self.calls:
            yield LLMDelta(tool_call=c)
        yield LLMDelta(done=True)


class TTS:
    model = "fake-tts"

    async def stream_audio(self, text, voice):
        await asyncio.sleep(0.002)
        yield np.ones(240, dtype=np.int16)


class Player:
    """Plays each generation instantly once it is finished."""

    def __init__(self):
        self.ctl = None
        self.held: set[int] = set()

    def begin_generation(self, gen, held=False):
        if held:
            self.held.add(gen)

    def release(self, gen):
        self.held.discard(gen)
        return True

    def enqueue(self, gen, pcm):
        return gen == self.ctl.active_gen

    def finish_generation(self, gen):
        asyncio.get_running_loop().call_later(0.01, self._play, gen)

    def _play(self, gen):
        self.ctl.player_event("start", gen, now_ns())
        self.ctl.player_event("end", gen, now_ns())

    def stop_now(self, gen):
        asyncio.get_running_loop().call_soon(self.ctl.player_event, "halted", gen, now_ns())


class Ctx:
    def __init__(self, ctl):
        self.ctl, self.speaking = ctl, False

    def user_speaking(self):
        return self.speaking


def build(routing="tool", duration=0.05, db=":memory:"):
    harness = Harness(db, worker_factory=lambda t: FakeSubAgent(duration))
    asr, llm, player = ASR(), LLM(), Player()
    ctl = TurnController(asr, llm, TTS(), player, VoiceProfile("test"))
    player.ctl = ctl
    ctl.hooks = MaidTurns(harness, routing)
    return harness, ctl, asr, llm


async def say(ctl, asr, text):
    asr.text = text
    t = now_ns()
    await ctl.submit(np.zeros(1600, dtype="float32"), 16000, t, t)
    await asyncio.sleep(0.02)  # committed tool calls run in their own task


async def until(cond, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


# ---- router fallback ---------------------------------------------------------------
def test_japanese_patterns_in_the_fallback_router():
    assert route_text("Python 3.14の新機能を調べて").route == "task"
    assert route_text("テストを走らせて直しておいて").route == "task"
    assert route_text("Steamを立ち上げて").route == "computer_use"
    assert route_text("キーボードのレビューを探して").route == "task", "a keyboard is not computer use"
    assert route_text("今日ちょっと疲れた").route == "direct"
    assert wants_cancel("やっぱりやめて") and not wants_cancel("今日はいい天気")
    assert yes_no("うん、いいよ") is True and yes_no("やめて") is False and yes_no("今日は疲れた") is None


# ---- delegate / T1 / cancel ------------------------------------------------------------
@pytest.mark.asyncio
async def test_delegate_submits_a_task_and_the_next_turn_sees_its_state():
    harness, ctl, asr, llm = build(duration=5)
    llm.text, llm.calls = "わかった、調べてくる。", [ToolCall("delegate", {"goal": "Search Python 3.14 news", "tier": "quick", "needs": ["web"]})]
    await say(ctl, asr, "Python 3.14の新機能を調べて")
    [task] = harness.registry.list()
    assert task.metadata["goal"] == "Search Python 3.14 news" and task.request == "Python 3.14の新機能を調べて"
    assert task.status == "running" and task.route == "task"

    llm.text, llm.calls = "まだ途中。", []
    await say(ctl, asr, "さっきのどうなった？")
    req = llm.requests[-1]
    assert req["tools"] and "Search Python 3.14 news" in req["context"] and "実行中" in req["context"]
    assert len(harness.registry.list()) == 1, "asking about a task does not start another"
    await harness.cancel(task.task_id)


@pytest.mark.asyncio
async def test_computer_tier_waits_for_confirmation_and_a_yes_starts_it():
    harness, ctl, asr, llm = build()
    llm.text, llm.calls = "始めていい？", [ToolCall("delegate", {"goal": "Open Steam", "tier": "computer"})]
    await say(ctl, asr, "Steamを開いて")
    [task] = harness.registry.list()
    assert task.status == "waiting_confirmation" and task.route == "computer_use"
    llm.text, llm.calls = "じゃあ、始める。", []  # she forgot confirm_task: the keyword backstop answers it
    await say(ctl, asr, "うん、お願い")
    assert (await harness.wait(task.task_id, 2)).status == "completed"


@pytest.mark.asyncio
async def test_cancel_by_voice_with_and_without_the_tool_call():
    harness, ctl, asr, llm = build(duration=5)
    for calls in ([ToolCall("cancel_task", {"task_id": "t-wrong-id"})], []):  # mangled id, then no call at all
        task = await harness.submit("x", metadata={"goal": "Search keyboards"})
        llm.text, llm.calls = "わかった、止める。", calls
        await say(ctl, asr, "やっぱりやめて")
        assert harness.registry.get(task.task_id).status == "cancelled"


@pytest.mark.asyncio
async def test_a_spoken_cancel_does_not_guess_between_two_tasks():
    harness, ctl, asr, llm = build(duration=5)
    a = await harness.submit("a", metadata={"goal": "A"})
    b = await harness.submit("b", metadata={"goal": "B"})
    llm.text, llm.calls = "どっちを止める？", []
    await say(ctl, asr, "やめて")
    assert {harness.registry.get(t.task_id).status for t in (a, b)} == {"running"}
    for t in (a, b):
        await harness.cancel(t.task_id)


@pytest.mark.asyncio
async def test_keyword_routing_delegates_without_tools():
    harness, ctl, asr, llm = build(routing="keyword")
    llm.text = "わかった。"
    await say(ctl, asr, "このプロジェクトを検索して")
    assert llm.requests[-1]["tools"] is None and "作業として受け付けた" in llm.requests[-1]["context"]
    [task] = harness.registry.list()
    assert task.metadata["routing"] == "keyword"
    await say(ctl, asr, "今日は疲れた")
    assert len(harness.registry.list()) == 1


# ---- arbiter ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_completion_is_spoken_at_the_next_pause_never_over_the_user(tmp_path):
    harness, ctl, asr, llm = build(duration=0.05)
    ctx = Ctx(ctl)
    arbiter = SpeechArbiter(harness, ArbiterConfig(quiet_gap_s=0.1, tick_s=0.02), log_dir=tmp_path)
    loop = asyncio.create_task(arbiter.run(ctx))
    try:
        ctx.speaking = True
        task = await harness.submit("x", metadata={"goal": "Search keyboards"})
        await harness.wait(task.task_id, 2)
        await asyncio.sleep(0.3)
        assert not any(r["mode"] == "notice" for r in ctl.completed), "spoke while the user was speaking"
        ctx.speaking = False
        llm.text = "キーボードの調べもの、終わったよ。"
        await until(lambda: any(r["mode"] == "notice" for r in ctl.completed))
        notice = [r for r in ctl.completed if r["mode"] == "notice"][-1]
        assert "[completed] Search keyboards: Simulated sub-agent" in notice["notice"]
        assert llm.requests[-1]["tools"] is None and "知らせ方" in llm.requests[-1]["context"]
        assert (tmp_path / time.strftime("%Y-%m-%d") / "arbiter.jsonl").exists()
    finally:
        loop.cancel()


@pytest.mark.asyncio
async def test_pending_news_merges_into_one_line_and_mute_holds_it():
    harness, ctl, asr, llm = build(duration=0.02)
    ctx = Ctx(ctl)
    arbiter = SpeechArbiter(harness, ArbiterConfig(quiet_gap_s=0.05, tick_s=0.02, muted=True))
    loop = asyncio.create_task(arbiter.run(ctx))
    try:
        tasks = [await harness.submit(g, metadata={"goal": g}) for g in ("A", "B")]
        await asyncio.gather(*(harness.wait(t.task_id, 2) for t in tasks))
        await asyncio.sleep(0.2)
        assert not any(r["mode"] == "notice" for r in ctl.completed) and len(arbiter.pending) == 2
        arbiter.cfg.muted = False
        await until(lambda: any(r["mode"] == "notice" for r in ctl.completed))
        notices = [r for r in ctl.completed if r["mode"] == "notice"]
        assert len(notices) == 1 and "] A:" in notices[0]["notice"] and "] B:" in notices[0]["notice"]
    finally:
        loop.cancel()


@pytest.mark.asyncio
async def test_heartbeat_for_a_long_task_and_nothing_for_a_cancelled_one():
    harness, ctl, asr, llm = build(duration=5)
    arbiter = SpeechArbiter(harness, ArbiterConfig(heartbeat_s=60))
    task = await harness.submit("x", metadata={"goal": "Long build"})
    await asyncio.sleep(0.01)  # let it start running
    assert arbiter._due_heartbeats(time.time()) == []
    [beat] = arbiter._due_heartbeats(time.time() + 61)
    assert beat.kind == "progress" and "Long build" in beat.text
    assert arbiter._due_heartbeats(time.time() + 62) == [], "at most once per heartbeat_s"
    await harness.cancel(task.task_id)
    assert arbiter.pending == [], "a cancel she already acknowledged is not announced again"


# ---- restart ---------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_restart_marks_running_tasks_interrupted_and_she_mentions_them(tmp_path):
    db = tmp_path / "tasks.sqlite3"
    first = Harness(db, worker_factory=lambda t: FakeSubAgent(30))
    task = await first.submit("x", metadata={"goal": "Long build"})
    first._running[task.task_id].cancel()  # the process stops mid-task
    await asyncio.sleep(0.01)
    second = Harness(db)
    arbiter = SpeechArbiter(second)
    [again] = second.recover()
    assert again.status == "interrupted" and second.registry.get(task.task_id).status == "interrupted"
    assert [i.kind for i in arbiter.pending] == ["interrupted"]
    assert "再起動で中断" in state_block([again])


# ---- capability list (what she knows she can do) ------------------------------------------------
def test_capability_block_and_tools_follow_the_harness_registry():
    from maid_harness import Capability
    from maid_agent.tools import capability_block, tools_for

    harness = Harness()
    caps = harness.capabilities.usable()
    block = capability_block(caps)
    assert "sub_agents（練習用）" in block and "weather" in block and "screen" not in block
    names = {t["function"]["name"] for t in tools_for(caps)}
    assert names == {"delegate", "cancel_task", "confirm_task"}, "no look_at_screen without a screen"

    harness.capabilities.register(Capability("screen", "live", "can look at the screen", ("look_at_screen",)))
    assert "look_at_screen" in {t["function"]["name"] for t in tools_for(harness.capabilities.usable())}
    harness.capabilities.set_status("screen", "off")
    assert "screen" not in capability_block(harness.capabilities.usable())


def test_role_card_v2_no_longer_says_she_can_never_check_the_weather():
    from maid_voice.persona import RoleCard

    persona = Path(__file__).resolve().parents[1] / "persona"
    v1, v2 = RoleCard.load(persona / "natsume.v1.toml"), RoleCard.load(persona / "natsume.v2.toml")
    assert "まだ確かめる手段がない" in v1.boundaries
    assert "まだ確かめる手段がない" not in v2.boundaries and "できること" in v2.boundaries
    assert (v1.identity, v1.speech, v1.voice_rules) == (v2.identity, v2.speech, v2.voice_rules)


@pytest.mark.asyncio
async def test_look_at_screen_answers_after_looking_through_the_arbiter():
    from maid_harness import Capability

    harness, ctl, asr, llm = build()
    arbiter = SpeechArbiter(harness, ArbiterConfig(quiet_gap_s=0.05, tick_s=0.02))
    questions = []

    async def look(q):
        questions.append(q)
        return "front app Safari; reading a weather page; answer: Tokyo 18°C, cloudy"

    harness.capabilities.register(Capability("screen", "live", "can look at the screen", ("look_at_screen",)))
    ctl.hooks = MaidTurns(harness, look=look, followup=arbiter.answer,
                          facts=lambda: ["画面（30秒前に見た）: Safari / reading"])
    llm.text, llm.calls = "", [ToolCall("look_at_screen", {"question": "what is the weather shown"})]
    await say(ctl, asr, "画面見て、天気どう？")
    req = llm.requests[-1]
    assert "look_at_screen" in {t["function"]["name"] for t in req["tools"]} and "できること" in req["context"]
    assert "Safari / reading" in req["context"], "reporter facts reach the state block"
    assert ctl.completed[-1]["assistant_text"] == "ちょっと見る。"
    assert questions == ["what is the weather shown"]
    assert [i.kind for i in arbiter.pending] == ["lookup"] and "Tokyo 18°C" in arbiter.pending[0].text

    loop = asyncio.create_task(arbiter.run(Ctx(ctl)))
    try:
        llm.text, llm.calls = "東京は十八度で曇りだって。", []
        await until(lambda: any(r["mode"] == "notice" for r in ctl.completed))
        assert "[screen, asked: what is the weather shown]" in ctl.completed[-1]["notice"]
    finally:
        loop.cancel()


@pytest.mark.asyncio
async def test_screen_look_waits_until_her_acknowledgement_has_played():
    from maid_agent.agent import AgentConfig, MaidAgent

    class Screen:
        def __init__(self):
            self.states = []

        async def look(self, q):
            self.states.append((ctl.active_gen, ctl.state))
            return "front app Warp"

        def facts(self):
            return []

    harness, ctl, asr, llm = build()
    agent = MaidAgent(Path("/tmp"), AgentConfig(), screen=Screen())
    agent.ctx = Ctx(ctl)
    ctl.active_gen, ctl.state = 7, "responding"  # she is still saying 「ちょっと見る」
    look = asyncio.create_task(agent._look_when_quiet("what is on screen"))
    await asyncio.sleep(0.2)
    assert agent.screen.states == [], "vision must not start while she speaks"
    ctl.active_gen, ctl.state = 0, "listening"
    assert await look == "front app Warp" and agent.screen.states == [(0, "listening")]
