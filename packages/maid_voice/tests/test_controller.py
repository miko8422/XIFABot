import asyncio
from pathlib import Path

import numpy as np
import pytest

from maid_voice.asr import FinalTranscript
from maid_voice.audio import PlaybackQueue
from maid_voice.config import VoiceProfile
from maid_voice.controller import TurnController
from maid_voice.llm import LLMDelta
from maid_voice.metrics import now_ns
from maid_voice.segmenter import ClauseSegmenter

VOICE = VoiceProfile("test", Path("/dev/null"), "テスト")


class FakeASR:
    model = "fake-asr"

    def __init__(self, text="hello", fail=False):
        self.text, self.fail = text, fail

    async def transcribe(self, pcm, sr):
        await asyncio.sleep(0.005)
        if self.fail:
            raise ConnectionError("asr down")
        return FinalTranscript(self.text, "en")


class FakeLLM:
    model = "fake-llm"

    def __init__(self, pieces=("はい、", "わかった。", "すぐやるわ。"), delay=0.01):
        self.pieces, self.delay = pieces, delay

    async def stream_response(self, history, user_text):
        for p in self.pieces:
            await asyncio.sleep(self.delay)
            yield LLMDelta(text=p)
        yield LLMDelta(done=True)


class FakeTTS:
    model = "fake-tts"
    sample_rate = 24000
    genuinely_streams = True

    def __init__(self, chunks=5, delay=0.01, fail=False):
        self.chunks, self.delay, self.fail = chunks, delay, fail

    async def stream_audio(self, text, voice):
        if self.fail:
            raise RuntimeError("tts crashed")
        for _ in range(self.chunks):
            await asyncio.sleep(self.delay)
            yield np.ones(240, dtype=np.int16)


class FakePlayer:
    """Drains a real PlaybackQueue on a timer, like the audio callback would."""

    def __init__(self):
        self.queue = PlaybackQueue()
        self.controller: TurnController | None = None
        self.played: list[int] = []  # generation id of every non-silent block
        self._halt = None
        self._task = None

    def attach(self, controller):
        self.controller = controller
        self._task = asyncio.create_task(self._pump())

    async def _pump(self):
        while True:
            await asyncio.sleep(0.002)
            samples, events = self.queue.read(240)
            if samples.any():
                self.played.append(self.queue._current_gen)
            for kind, gen, _ in events:
                self.controller.player_event(kind, gen, now_ns())
            if self._halt is not None:
                self.controller.player_event("halted", self._halt, now_ns())
                self._halt = None

    def begin_generation(self, gen, held=False):
        self.queue.set_active(gen, held)

    def release(self, gen):
        return self.queue.release(gen)

    def enqueue(self, gen, pcm):
        return self.queue.push(gen, pcm)

    def finish_generation(self, gen):
        self.queue.push(gen, PlaybackQueue.END)

    def stop_now(self, gen):
        self._halt = gen
        self.queue.flush()


def make(asr=None, llm=None, tts=None):
    player = FakePlayer()
    events = []
    ctl = TurnController(asr or FakeASR(), llm or FakeLLM(), tts or FakeTTS(), player, VOICE,
                         emit=events.append)
    player.attach(ctl)
    return ctl, player, events


def submit(ctl):
    t = now_ns()
    return ctl.submit(np.zeros(1600, dtype="float32"), 16000, t, t, source="test")


async def settle(ctl, timeout=2.0):
    for _ in range(int(timeout / 0.01)):
        if ctl.state == "listening" and (not ctl._task or ctl._task.done()):
            await asyncio.sleep(0.05)  # let halted events land
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"controller stuck in {ctl.state}")


@pytest.mark.asyncio
async def test_happy_path_records_all_stages():
    ctl, player, events = make()
    await submit(ctl)
    await settle(ctl)
    rec = ctl.completed[-1]
    assert rec["error"] is None and not rec.get("cancelled")
    for key in ("asr_ms", "llm_ttft_ms", "tts_first_audio_ms", "first_sound_ms"):
        assert rec[key] is not None and rec[key] >= 0, key
    clauses = [e["text"] for e in events if e["type"] == "clause"]
    assert clauses == ["はい、わかった。", "すぐやるわ。"]
    assert len(ctl.history) == 2


@pytest.mark.asyncio
async def test_new_turn_cancels_old_and_no_stale_audio_plays():
    ctl, player, _ = make(tts=FakeTTS(chunks=40, delay=0.01))
    submit(ctl)
    await asyncio.sleep(0.15)  # first turn is mid-playback
    first_gen = ctl.active_gen
    assert first_gen in player.played
    submit(ctl)
    second_gen = ctl.active_gen
    played_before = len(player.played)
    await settle(ctl, timeout=5)
    after = player.played[played_before:]
    assert first_gen not in after, "stale generation audio played after interruption"
    assert second_gen in after
    cancelled = [r for r in ctl.completed if r["cancelled"]]
    assert len(cancelled) == 1 and cancelled[0]["stop_ms"] is not None


@pytest.mark.asyncio
async def test_late_packets_for_old_generation_are_rejected():
    q = PlaybackQueue()
    q.set_active(1)
    assert q.push(1, np.ones(10, dtype=np.int16))
    q.set_active(2)
    assert not q.push(1, np.ones(10, dtype=np.int16))  # late packet from gen 1
    samples, events = q.read(10)
    assert not samples.any() and q.dropped_chunks == 2


@pytest.mark.asyncio
async def test_explicit_stop_returns_to_listening():
    ctl, player, _ = make(tts=FakeTTS(chunks=50))
    submit(ctl)
    await asyncio.sleep(0.1)
    await ctl.stop()
    await settle(ctl)
    assert ctl.completed[-1]["cancelled"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asr,tts", [(FakeASR(fail=True), None), (None, FakeTTS(fail=True))])
async def test_stage_failure_recovers(asr, tts):
    ctl, player, events = make(asr=asr, tts=tts)
    await submit(ctl)
    await settle(ctl)
    assert ctl.completed[-1]["error"]
    assert any(e["type"] == "error" for e in events)
    # the app still works afterwards
    ctl.asr, ctl.tts = FakeASR(), FakeTTS()
    await submit(ctl)
    await settle(ctl)
    assert ctl.completed[-1]["error"] is None


@pytest.mark.asyncio
async def test_empty_transcript_does_not_call_llm():
    ctl, _, events = make(asr=FakeASR(text=""))
    await submit(ctl)
    await settle(ctl)
    assert ctl.completed[-1]["error"] == "empty_transcript"
    assert not any(e["type"] == "llm.delta" for e in events)


def test_segmenter_flushes_first_clause_early():
    seg = ClauseSegmenter()
    out = []
    for piece in ["うーん、", "今日は和食", "がいいんじゃない、お魚とか。", "どうかしら？"]:
        out += seg.push(piece)
    out += seg.flush()
    # first clause breaks at 、 once long enough; later ones only at sentence ends
    assert out == ["うーん、今日は和食がいいんじゃない、", "お魚とか。", "どうかしら？"]


def test_segmenter_does_not_emit_bare_punctuation():
    seg = ClauseSegmenter()
    assert seg.push("。") == []
    assert seg.push("はい。") == ["。はい。"]


@pytest.mark.asyncio
async def test_say_speaks_a_fixed_line_and_is_refused_during_a_turn():
    ctl, player, _ = make(tts=FakeTTS(chunks=20, delay=0.01))
    task = ctl.say("Codexの作業、終わったみたい。")
    assert task is not None
    assert ctl.say("二つ目") is None, "must not start a second line while speaking"
    await settle(ctl, timeout=5)
    rec = ctl.completed[-1]
    assert rec["mode"] == "proactive" and rec["error"] is None and rec["assistant_text"].startswith("Codex")
    assert ctl.history[-1].role == "assistant" and ctl.history[-1].content.startswith("Codex")

    submit(ctl)
    await asyncio.sleep(0.05)
    assert ctl.say("割り込み") is None, "must not talk over a turn in progress"
    await settle(ctl, timeout=5)
