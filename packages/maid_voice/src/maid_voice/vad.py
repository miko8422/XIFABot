"""Hands-free endpointing: frame-level energy VAD with an adaptive noise floor.

Emits, per utterance:
  start        speech began (after min_speech_ms of voiced frames)
  speculative  silence reached spec_silence_ms: the controller may start work, playback held
  resume       speech came back after a speculative event: speculative work must be dropped
  end          silence reached end_silence_ms: the turn is committed
The detector is pure (no audio device, no asyncio) so it can be unit-tested with synthetic frames.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VADConfig:
    min_speech_ms: int = 120  # voiced time before declaring a speech start
    spec_silence_ms: int = 250  # silence before speculative generation starts
    end_silence_ms: int = 600  # silence that commits the turn
    preroll_ms: int = 300  # audio kept from before the detected start
    max_utterance_s: float = 30.0
    threshold_ratio: float = 3.0  # voiced = rms above noise_floor * ratio ...
    min_threshold: float = 0.008  # ... and above this absolute floor
    noise_adapt: float = 0.05  # EMA rate for the noise floor on unvoiced frames


@dataclass
class VADEvent:
    kind: str  # start | speculative | resume | end
    t_ns: int  # monotonic time the event refers to (speech start / last voiced frame)
    pcm: np.ndarray | None = None  # utterance audio for speculative / end
    t_speech_start_ns: int | None = None


@dataclass
class EnergyEndpointer:
    sample_rate: int = 16000
    cfg: VADConfig = field(default_factory=VADConfig)
    noise_floor: float = 0.003

    def __post_init__(self):
        self._preroll: list[np.ndarray] = []
        self._utt: list[np.ndarray] = []
        self._in_speech = False
        self._voiced_ms = 0.0
        self._silence_ms = 0.0
        self._spec_sent = False
        self._t_start = 0
        self._t_last_voiced = 0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def utterance_audio(self) -> np.ndarray:
        return np.concatenate(self._utt) if self._utt else np.zeros(0, dtype="float32")

    def threshold(self) -> float:
        return max(self.cfg.min_threshold, self.noise_floor * self.cfg.threshold_ratio)

    def push(self, frame: np.ndarray, t_end_ns: int) -> list[VADEvent]:
        frame_ms = 1000 * len(frame) / self.sample_rate
        rms = float(np.sqrt(np.mean(frame * frame))) if len(frame) else 0.0
        voiced = rms > self.threshold()
        if not voiced and not self._in_speech:
            self.noise_floor += self.cfg.noise_adapt * (rms - self.noise_floor)
        events: list[VADEvent] = []

        if not self._in_speech:
            self._preroll.append(frame)
            keep = max(1, int(self.cfg.preroll_ms / frame_ms))
            self._preroll = self._preroll[-keep:]
            self._voiced_ms = self._voiced_ms + frame_ms if voiced else 0.0
            if self._voiced_ms >= self.cfg.min_speech_ms:
                self._in_speech = True
                self._utt = list(self._preroll)
                self._preroll = []
                self._t_start = t_end_ns - int(self._voiced_ms * 1e6)
                self._t_last_voiced = t_end_ns
                self._silence_ms = 0.0
                self._spec_sent = False
                events.append(VADEvent("start", self._t_start))
            return events

        self._utt.append(frame)
        if voiced:
            self._t_last_voiced = t_end_ns
            if self._spec_sent:
                self._spec_sent = False
                events.append(VADEvent("resume", t_end_ns))
            self._silence_ms = 0.0
        else:
            self._silence_ms += frame_ms

        too_long = len(self._utt) * frame_ms >= self.cfg.max_utterance_s * 1000
        if not self._spec_sent and self._silence_ms >= self.cfg.spec_silence_ms:
            self._spec_sent = True
            events.append(VADEvent("speculative", self._t_last_voiced, self.utterance_audio(), self._t_start))
        if self._silence_ms >= self.cfg.end_silence_ms or too_long:
            events.append(VADEvent("end", self._t_last_voiced, self.utterance_audio(), self._t_start))
            self._in_speech = False
            self._utt = []
            self._voiced_ms = 0.0
        return events

    def reset(self) -> None:
        self.__post_init__()
