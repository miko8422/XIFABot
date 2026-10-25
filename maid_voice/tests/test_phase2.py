import asyncio

import numpy as np
import pytest

from maid_voice.audio import PlaybackQueue
from maid_voice.vad import EnergyEndpointer, VADConfig
from test_controller import FakeTTS, make, settle

SR = 16000
FRAME = 320  # 20 ms


def frames(kind: str, ms: int):
    n = ms // 20
    rng = np.random.default_rng(0)
    for _ in range(n):
        if kind == "speech":
            yield (0.1 * np.sin(np.arange(FRAME) / 5) + 0.01 * rng.standard_normal(FRAME)).astype("float32")
        else:
            yield (0.001 * rng.standard_normal(FRAME)).astype("float32")


def run_vad(script, cfg=None):
    ep = EnergyEndpointer(SR, cfg or VADConfig())
    t = 0
    events = []
    for kind, ms in script:
        for f in frames(kind, ms):
            t += 20_000_000
            events += [(e.kind, round(e.t_ns / 1e6)) for e in ep.push(f, t)]
    return events


def test_vad_basic_utterance():
    ev = run_vad([("silence", 500), ("speech", 800), ("silence", 800)])
    kinds = [k for k, _ in ev]
    assert kinds == ["start", "speculative", "end"]
    # speech end is the last voiced frame (1300 ms), not when silence was detected
    assert ev[1][1] == ev[2][1] == 1300


def test_vad_pause_shorter_than_endpoint_is_one_utterance():
    ev = run_vad([("silence", 400), ("speech", 600), ("silence", 400), ("speech", 600), ("silence", 800)])
    assert [k for k, _ in ev] == ["start", "speculative", "resume", "speculative", "end"]


def test_vad_ignores_short_blips():
    ev = run_vad([("silence", 400), ("speech", 60), ("silence", 1000)])
    assert ev == []


def test_held_queue_plays_silence_until_released():
    q = PlaybackQueue()
    q.set_active(5, held=True)
    q.push(5, np.ones(100, dtype=np.int16))
    samples, events = q.read(100)
    assert not samples.any() and events == []
    assert q.release(5)
    samples, events = q.read(100)
    assert samples.all() and events[0][0] == "start"


@pytest.mark.asyncio
async def test_speculation_dropped_on_resume_never_plays():
    ctl, player, _ = make(tts=FakeTTS(chunks=20))
    t = 0
    ctl.submit(np.zeros(1600, dtype="float32"), 16000, t, None, held=True, mode="handsfree")
    spec_gen = ctl.active_gen
    await asyncio.sleep(0.2)  # LLM + TTS done, audio sitting in the held queue
    assert spec_gen not in player.played
    # user keeps talking -> new speculative turn replaces it
    ctl.submit(np.zeros(3200, dtype="float32"), 16000, t, None, held=True, mode="handsfree")
    new_gen = ctl.active_gen
    await asyncio.sleep(0.05)
    assert ctl.commit(new_gen, t)
    await settle(ctl, timeout=5)
    assert spec_gen not in player.played
    assert new_gen in player.played
    dropped = [r for r in ctl.completed if r["generation_id"] == spec_gen][0]
    assert dropped["error"] == "cancelled:speculation_dropped"
    assert len(ctl.history) == 2  # only the committed turn enters the dialogue


@pytest.mark.asyncio
async def test_commit_of_stale_generation_is_refused():
    ctl, _, _ = make()
    ctl.submit(np.zeros(1600, dtype="float32"), 16000, 0, None, held=True)
    old = ctl.active_gen
    ctl.submit(np.zeros(1600, dtype="float32"), 16000, 0, None, held=True)
    assert not ctl.commit(old, 0)
    await ctl.stop()
