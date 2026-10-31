"""P4: activity reporter, screen reporter, per-activity speech rules, proactive gate. No screen, no models."""

import asyncio
import time
from pathlib import Path

import httpx
import pytest
from maid_harness import Harness
from maid_harness.reporters.activity import (ActivityProfiles, ActivityReporter, ActivityWatchSource, Sample,
                                             classify)
from maid_harness.reporters.screen.observer import Settled
from maid_harness.reporters.screen.reporter import ScreenReporter
from maid_harness.reporters.screen.vision import Observation

from maid_agent.arbiter import ArbiterConfig, Item, SpeechArbiter, SpeechRules
from maid_agent.proactive import ProactiveConfig, ProactiveSpeaker
from test_p1 import Ctx, build, until


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        return self.t


# ---- activity reporter ------------------------------------------------------------------------------
def test_classify_away_call_game_working_and_any_app_can_be_a_game():
    p = ActivityProfiles()
    assert classify(Sample("Steam"), p) == "gaming"
    assert classify(Sample("Google Chrome", title="Meet - weekly sync"), p) == "call"
    assert classify(Sample("zoom.us", idle_s=500), p) == "away", "away wins over everything"
    assert classify(Sample("Warp", locked=True), p) == "away"
    assert classify(Sample("Warp"), p) == "working"
    demo = ActivityProfiles(game=("Google Chrome",))
    assert classify(Sample("Google Chrome"), demo) == "gaming"


def test_changes_are_debounced_and_reported_with_how_long_the_last_one_lasted():
    clock, events = Clock(), []
    rep = ActivityReporter(lambda t, src, p: events.append((t, p)), ActivityProfiles(debounce_s=5), clock=clock)
    rep.observe(Sample("Warp"))
    clock.t += 60
    assert rep.observe(Sample("Steam")) is None and events == [], "a quick alt-tab is not a change"
    clock.t += 2
    rep.observe(Sample("Warp"))
    clock.t += 1
    rep.observe(Sample("Steam"))
    clock.t += 6
    state = rep.observe(Sample("Steam"))
    assert state.activity == "gaming" and events == [("activity.changed", {"activity": "gaming", "app": "Steam",
                                                                           "previous": "working", "previous_s": 63})]
    clock.t += 30
    rep.observe(Sample("Minecraft"))  # still gaming: the app is a fact, not a change
    assert rep.state.app == "Minecraft" and len(events) == 1
    assert rep.facts() == ["あきらの今: ゲーム中（Minecraft）、さっきから"]


def test_away_is_immediate_and_coming_back_says_how_long():
    clock, events = Clock(), []
    rep = ActivityReporter(lambda t, src, p: events.append((t, p)), clock=clock)
    rep.observe(Sample("Warp"))
    clock.t += 10
    rep.observe(Sample("Warp", idle_s=400))
    assert rep.state.activity == "away" and events[-1][0] == "activity.changed"
    clock.t += 1200
    rep.observe(Sample("Warp"))
    clock.t += 6
    rep.observe(Sample("Warp"))
    assert events[-1] == ("user.back", {"away_s": 1200, "activity": "working", "app": "Warp"})


@pytest.mark.asyncio
async def test_activitywatch_source_reads_the_newest_window_and_afk_events():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/0/buckets/":
            return httpx.Response(200, json={"aw-watcher-window_mac": {}, "aw-watcher-afk_mac": {}})
        if "window" in path:
            return httpx.Response(200, json=[{"data": {"app": "Google Chrome", "title": "Meet - standup"}}])
        return httpx.Response(200, json=[{"data": {"status": "not-afk"}}])

    src = ActivityWatchSource()
    src._client = httpx.AsyncClient(base_url="http://aw", transport=httpx.MockTransport(handler))
    assert await src.available()
    s = await src.sample()
    assert (s.app, s.title, s.idle_s) == ("Google Chrome", "Meet - standup", 0.0)
    assert classify(s, ActivityProfiles()) == "call"


# ---- speech arbiter per activity ----------------------------------------------------------------------
def test_speech_rules_from_config_override_only_what_is_given():
    rules = SpeechRules.from_raw({"gaming": {"cooldown_s": 120}, "greet_after_s": 60})
    assert rules.get("gaming").cooldown_s == 120 and rules.get("gaming").style
    assert rules.get("call").task_news == "hold" and rules.greet_after_s == 60
    assert rules.get("something-new").task_news == "speak"


@pytest.mark.asyncio
async def test_task_news_waits_out_a_call_but_answers_to_her_own_questions_do_not():
    harness, ctl, asr, llm = build(duration=0.02)
    activity = ["call"]
    arbiter = SpeechArbiter(harness, ArbiterConfig(quiet_gap_s=0.05, tick_s=0.02), activity=lambda: activity[0])
    loop = asyncio.create_task(arbiter.run(Ctx(ctl)))
    try:
        task = await harness.submit("x", metadata={"goal": "Search keyboards"})
        await harness.wait(task.task_id, 2)
        arbiter.answer("screen", "[screen, asked: what is open] front app Zoom")
        await until(lambda: any(r["mode"] == "notice" for r in ctl.completed))
        first = [r for r in ctl.completed if r["mode"] == "notice"]
        assert len(first) == 1 and "front app Zoom" in first[0]["notice"] and "keyboards" not in first[0]["notice"]
        await asyncio.sleep(0.3)
        assert sum(r["mode"] == "notice" for r in ctl.completed) == 1, "held while the call goes on"
        activity[0] = "working"
        await until(lambda: sum(r["mode"] == "notice" for r in ctl.completed) == 2)
        assert "Search keyboards" in [r for r in ctl.completed if r["mode"] == "notice"][-1]["notice"]
    finally:
        loop.cancel()


@pytest.mark.asyncio
async def test_back_after_a_while_she_greets_and_tells_what_came_in_while_gaming_short():
    harness, ctl, asr, llm = build(duration=0.02)
    activity = ["away"]
    arbiter = SpeechArbiter(harness, ArbiterConfig(quiet_gap_s=0.05, tick_s=0.02), activity=lambda: activity[0],
                            rules=SpeechRules(greet_after_s=600))
    loop = asyncio.create_task(arbiter.run(Ctx(ctl)))
    try:
        task = await harness.submit("x", metadata={"goal": "Build the report"})
        await harness.wait(task.task_id, 2)
        await asyncio.sleep(0.2)
        assert not any(r["mode"] == "notice" for r in ctl.completed)
        activity[0] = "gaming"
        harness.publish_event("user.back", "reporter.activity", {"away_s": 1500, "activity": "gaming", "app": "Steam"})
        await until(lambda: any(r["mode"] == "notice" for r in ctl.completed))
        notice = [r for r in ctl.completed if r["mode"] == "notice"][0]["notice"]
        assert notice.index("back after 25 min") < notice.index("Build the report"), "greeting first, then the news"
        assert "ゲーム中" in notice, "the gaming style rides along"
    finally:
        loop.cancel()


def test_a_short_absence_gets_no_greeting():
    harness, ctl, asr, llm = build()
    arbiter = SpeechArbiter(harness, rules=SpeechRules(greet_after_s=600))
    harness.publish_event("user.back", "reporter.activity", {"away_s": 200})
    assert arbiter.pending == []


# ---- proactive gate --------------------------------------------------------------------------------------
def test_proactive_gate_per_activity_cooldown_and_manual_looks():
    activity = ["working"]
    sp = ProactiveSpeaker(Path("/tmp"), ProactiveConfig(), SpeechRules(), lambda: activity[0])
    obs = {"worth_mentioning": True, "sensitive": False}
    now = time.time()
    assert sp.gate(obs, now) is None
    sp._last_spoke = now - 400
    assert "cooldown while working" in sp.gate(obs, now)[1], "working: 10 min"
    activity[0] = "gaming"
    assert sp.gate(obs, now) is None, "gaming: 5 min, so 400 s is enough"
    activity[0] = "call"
    assert "no remarks while call" in sp.gate(obs, now)[1]
    assert sp.gate({**obs, "manual": True}, now) is None, "the test button ignores rules and cooldown"
    activity[0] = "working"
    sp._last_spoke = None
    assert sp.gate({**obs, "sensitive": True}, now)[1] == "sensitive screen"
    assert "no LLM call" in sp.gate({**obs, "worth_mentioning": False}, now)[1]
    sp.cfg.speak = False
    assert "switched off" in sp.gate(obs, now)[1]


# ---- screen reporter ---------------------------------------------------------------------------------------
class FakeObserver:
    enabled = True

    class plat:
        @staticmethod
        def foreground():
            return type("F", (), {"app": "Steam", "bundle_id": "com.valvesoftware.steam"})()

    def grab_main(self):
        from PIL import Image
        return Image.new("RGB", (64, 64))


class FakeVision:
    model = "fake-vision"

    def __init__(self):
        self.calls = 0

    async def describe(self, image, app, question=""):
        self.calls += 1
        return Observation("a boss fight", "the boss is at low HP", True, False, "{}", 5.0)


def settled():
    from PIL import Image
    return Settled(time.time(), "Steam", "com.valvesoftware.steam", 0.3, 0.5, 0.4, False, None, 0.1, True,
                   ["meaning_changed"], Image.new("RGB", (64, 64)))


@pytest.mark.asyncio
async def test_screen_reporter_waits_while_anyone_speaks_and_drops_stale_changes():
    events, busy = [], [True]
    rep = ScreenReporter(lambda t, src, p: events.append((t, p)), Path("/tmp"), busy=lambda: busy[0], busy_wait_s=0.3)
    rep.observer, rep.vision = FakeObserver(), FakeVision()
    worker = asyncio.create_task(rep._worker())
    try:
        rep._latest = settled()
        rep._wake.set()
        await asyncio.sleep(0.6)
        assert rep.vision.calls == 0 and events[-1][0] == "screen.skipped", "never looks while someone speaks"
        busy[0] = False
        rep._latest = settled()
        rep._wake.set()
        for _ in range(50):
            if events[-1][0] == "screen.changed":
                break
            await asyncio.sleep(0.02)
        t, p = events[-1]
        assert t == "screen.changed" and p["app"] == "Steam" and p["notable"] == "the boss is at low HP"
        assert "boss fight" in rep.facts()[0]
    finally:
        worker.cancel()


@pytest.mark.asyncio
async def test_screen_changes_reach_the_proactive_speaker_through_the_inbox():
    harness, ctl, asr, llm = build()
    sp = ProactiveSpeaker(Path("/tmp"), ProactiveConfig(), SpeechRules(), lambda: "gaming")
    sp.ctx = Ctx(ctl)

    async def broadcast(e):
        pass

    sp.ctx.broadcast = broadcast
    harness.inbox.subscribe(sp._on_event)
    harness.publish_event("screen.changed", "reporter.screen", {"app": "Steam", "worth_mentioning": True})
    assert sp._latest is not None and sp._latest.payload["app"] == "Steam" and sp._wake.is_set()
