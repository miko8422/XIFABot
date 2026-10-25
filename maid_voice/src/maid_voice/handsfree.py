"""Hands-free turn taking: VAD events -> speculative / committed turns + interim transcripts."""

import asyncio

import numpy as np

from .controller import Emit, TurnController
from .metrics import now_ns
from .vad import EnergyEndpointer, VADConfig


class HandsFree:
    def __init__(self, ctl: TurnController, sample_rate: int, cfg: VADConfig, emit: Emit | None = None,
                 speculative: bool = True, barge_in: bool = False, partial_interval_ms: int = 400):
        self.ctl = ctl
        self.sample_rate = sample_rate
        self.endpointer = EnergyEndpointer(sample_rate, cfg)
        self.emit = emit
        self.speculative = speculative
        self.barge_in = barge_in  # voice interrupts playback; headphones only until AEC lands
        self.partial_interval_ms = partial_interval_ms
        self.enabled = True
        self.frames: asyncio.Queue[tuple[np.ndarray, int]] = asyncio.Queue(maxsize=500)
        self._spec_gen: int | None = None
        self._partial_task: asyncio.Task | None = None
        self._last_partial_ns = 0
        self._loop: asyncio.AbstractEventLoop | None = None

    async def _emit(self, event: dict) -> None:
        if self.emit:
            res = self.emit(event)
            if asyncio.iscoroutine(res):
                await res

    def feed_threadsafe(self, frame: np.ndarray, t_end_ns: int) -> None:
        """Capture callback hook (audio thread). Drops frames rather than blocking."""
        if self._loop and self.enabled:
            self._loop.call_soon_threadsafe(self._put, frame, t_end_ns)

    def _put(self, frame, t):
        if not self.frames.full():
            self.frames.put_nowait((frame, t))

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        while True:
            frame, t = await self.frames.get()
            await self.process(frame, t)

    async def process(self, frame: np.ndarray, t_end_ns: int) -> None:
        for ev in self.endpointer.push(frame, t_end_ns):
            await self._on_event(ev)
        self._maybe_partial()

    async def _on_event(self, ev) -> None:
        ctl = self.ctl
        if ev.kind == "start":
            if ctl.state == "responding" and self._spec_gen is None:
                if not self.barge_in:
                    self.endpointer.reset()  # her own voice or a backchannel; ignore while speaking
                    return
                await ctl.stop("voice_barge_in")
            await self._emit({"type": "state", "state": "capturing"})
        elif ev.kind == "speculative" and self.speculative:
            ctl.submit(ev.pcm, self.sample_rate, ev.t_ns, None, t_speech_start_ns=ev.t_speech_start_ns,
                       held=True, mode="handsfree")
            self._spec_gen = ctl.active_gen
        elif ev.kind == "resume" and self._spec_gen is not None:
            if ctl.active_gen == self._spec_gen:
                ctl.cancel("speculation_dropped")
            self._spec_gen = None
            await self._emit({"type": "state", "state": "capturing"})
        elif ev.kind == "end":
            t_endpoint = now_ns()
            if self._spec_gen is not None and ctl.commit(self._spec_gen, t_endpoint):
                pass  # speculative work becomes the real turn; held audio starts playing
            else:
                ctl.submit(ev.pcm, self.sample_rate, ev.t_ns, None, t_speech_start_ns=ev.t_speech_start_ns,
                           mode="handsfree")
                ctl.commit(ctl.active_gen, t_endpoint)
            self._spec_gen = None

    def _maybe_partial(self) -> None:
        if not self.endpointer.in_speech or self._spec_gen is not None:
            return
        if self._partial_task and not self._partial_task.done():
            return  # one interim request in flight at a time; never queue them up
        now = now_ns()
        if now - self._last_partial_ns < self.partial_interval_ms * 1e6:
            return
        self._last_partial_ns = now
        pcm = self.endpointer.utterance_audio()
        if len(pcm) < self.sample_rate * 0.3:
            return
        self._partial_task = asyncio.create_task(self._partial(pcm))

    async def _partial(self, pcm: np.ndarray) -> None:
        try:
            res = await self.ctl.asr.transcribe(pcm, self.sample_rate)
            if self.endpointer.in_speech:  # interim only; never acted upon
                await self._emit({"type": "asr.partial", "text": res.text, "language": res.language})
        except Exception:
            pass  # interim hypotheses are best-effort
