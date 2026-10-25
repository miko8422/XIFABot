"""Microphone capture and cancellable playback.

Both run in PortAudio callback threads and never block on network or model work.
Timestamps are converted from the PortAudio stream clock to time.monotonic_ns so that
"first sample played" means the estimated DAC time, not the time we queued the chunk.
"""

import collections
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import sounddevice as sd


def _resolve_device(name: str, kind: str):
    if not name:
        return None
    for i, d in enumerate(sd.query_devices()):
        if name.lower() in d["name"].lower() and d[f"max_{kind}_channels"] > 0:
            return i
    raise RuntimeError(f"no {kind} device matching {name!r}")


@dataclass
class CapturedUtterance:
    pcm: np.ndarray  # float32 mono
    sample_rate: int
    t_start_ns: int
    t_speech_start_ns: int | None
    t_speech_end_ns: int | None


class Capture:
    """Always-open input stream; push-to-talk selects which frames form the utterance."""

    def __init__(self, sample_rate: int = 16000, device: str = "", block_ms: int = 20,
                 voice_threshold: float = 0.015):
        self.sample_rate = sample_rate
        self.voice_threshold = voice_threshold
        self._lock = threading.Lock()
        self._recording = False
        self._blocks: list[np.ndarray] = []
        self._t_start = 0
        self._t_voice_first: int | None = None
        self._t_voice_last: int | None = None
        self.level = 0.0
        self.on_frame: Callable[[np.ndarray, int], None] | None = None  # called on the audio thread
        self.device_name = sd.query_devices(_resolve_device(device, "input") or sd.default.device[0])["name"]
        self._stream = sd.InputStream(
            samplerate=sample_rate, channels=1, dtype="float32",
            blocksize=int(sample_rate * block_ms / 1000), latency="low",
            device=_resolve_device(device, "input"), callback=self._callback,
        )

    def start(self) -> None:
        self._stream.start()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()

    def _callback(self, indata, frames, time_info, status):
        block = indata[:, 0].copy()
        # monotonic time at which the *end* of this block hit the ADC
        adc_lag = time_info.currentTime - time_info.inputBufferAdcTime
        t_end = time.monotonic_ns() - int(adc_lag * 1e9) + int(frames / self.sample_rate * 1e9)
        rms = float(np.sqrt(np.mean(block * block)))
        self.level = rms
        if self.on_frame:
            self.on_frame(block, t_end)
        with self._lock:
            if not self._recording:
                return
            self._blocks.append(block)
            if rms > self.voice_threshold:
                if self._t_voice_first is None:
                    self._t_voice_first = t_end - int(frames / self.sample_rate * 1e9)
                self._t_voice_last = t_end

    @property
    def recording(self) -> bool:
        """Push-to-talk is held."""
        return self._recording

    def begin(self) -> None:
        with self._lock:
            self._recording = True
            self._blocks = []
            self._t_start = time.monotonic_ns()
            self._t_voice_first = self._t_voice_last = None

    def end(self) -> CapturedUtterance:
        with self._lock:
            self._recording = False
            pcm = np.concatenate(self._blocks) if self._blocks else np.zeros(0, dtype="float32")
            return CapturedUtterance(pcm, self.sample_rate, self._t_start,
                                     self._t_voice_first, self._t_voice_last)


class PlaybackQueue:
    """Generation-tagged chunk queue. Pure logic, no audio device, so it is unit-testable.

    Only chunks whose generation_id equals the active generation are ever returned;
    anything older is dropped on arrival or on read (late-packet rejection).
    """

    END = None  # sentinel chunk marking end of a generation's audio

    def __init__(self):
        self._lock = threading.Lock()
        self._chunks: collections.deque[tuple[int, np.ndarray | None]] = collections.deque()
        self._current: np.ndarray | None = None
        self._current_gen = -1
        self._pos = 0
        self.active_gen = -1
        self.held = False  # speculative generation: buffer audio but play silence
        self.dropped_chunks = 0

    def set_active(self, gen: int, held: bool = False) -> None:
        with self._lock:
            self.active_gen = gen
            self.held = held
            self._purge()

    def release(self, gen: int) -> bool:
        """Open the gate for a held (speculative) generation once its turn is committed."""
        with self._lock:
            if gen != self.active_gen:
                return False
            self.held = False
            return True

    def push(self, gen: int, chunk: np.ndarray | None) -> bool:
        with self._lock:
            if gen != self.active_gen:
                self.dropped_chunks += 1
                return False
            self._chunks.append((gen, chunk))
            return True

    def flush(self) -> None:
        with self._lock:
            self.active_gen = -1
            self._purge()

    def _purge(self) -> None:
        kept = [(g, c) for g, c in self._chunks if g == self.active_gen]
        self.dropped_chunks += len(self._chunks) - len(kept)
        self._chunks = collections.deque(kept)
        if self._current_gen != self.active_gen:
            self._current, self._pos = None, 0

    def read(self, n: int) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
        """Fill n samples. Returns (samples, events) with events (kind, gen, sample_offset)."""
        out = np.zeros(n, dtype=np.int16)
        events = []
        filled = 0
        with self._lock:
            while filled < n and not self.held:
                if self._current is None:
                    if not self._chunks:
                        break
                    gen, chunk = self._chunks.popleft()
                    if gen != self.active_gen:
                        self.dropped_chunks += 1
                        continue
                    if chunk is None:
                        events.append(("end", gen, filled))
                        continue
                    if self._current_gen != gen:
                        events.append(("start", gen, filled))
                    self._current, self._current_gen, self._pos = chunk, gen, 0
                take = min(n - filled, len(self._current) - self._pos)
                out[filled:filled + take] = self._current[self._pos:self._pos + take]
                filled += take
                self._pos += take
                if self._pos >= len(self._current):
                    self._current = None
        return out, events

    def idle(self) -> bool:
        with self._lock:
            return self._current is None and not self._chunks


class Player:
    """Output stream fed from a PlaybackQueue. Reports start/end/halt with DAC-time estimates."""

    def __init__(self, sample_rate: int = 24000, device: str = "", block_ms: int = 10,
                 on_event: Callable[[str, int, int], None] | None = None):
        self.sample_rate = sample_rate
        self.queue = PlaybackQueue()
        self.on_event = on_event or (lambda kind, gen, t_ns: None)
        self._halt_pending: int | None = None
        self._was_playing = False
        dev = _resolve_device(device, "output")
        self.device_name = sd.query_devices(dev if dev is not None else sd.default.device[1])["name"]
        self._stream = sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype="int16",
            blocksize=int(sample_rate * block_ms / 1000), latency="low",
            device=dev, callback=self._callback,
        )

    def start(self) -> None:
        self._stream.start()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()

    def begin_generation(self, gen: int, held: bool = False) -> None:
        self.queue.set_active(gen, held)

    def release(self, gen: int) -> bool:
        return self.queue.release(gen)

    def enqueue(self, gen: int, pcm: np.ndarray) -> bool:
        return self.queue.push(gen, pcm)

    def finish_generation(self, gen: int) -> None:
        self.queue.push(gen, PlaybackQueue.END)

    def stop_now(self, gen: int) -> None:
        """Flush everything; the next callback emits silence and reports 'halted'."""
        self._halt_pending = gen
        self.queue.flush()

    def _callback(self, outdata, frames, time_info, status):
        dac_lag = time_info.outputBufferDacTime - time_info.currentTime
        t_dac = time.monotonic_ns() + int(dac_lag * 1e9)
        samples, events = self.queue.read(frames)
        outdata[:, 0] = samples
        for kind, gen, offset in events:
            self.on_event(kind, gen, t_dac + int(offset / self.sample_rate * 1e9))
        if self._halt_pending is not None:
            self.on_event("halted", self._halt_pending, t_dac)
            self._halt_pending = None
