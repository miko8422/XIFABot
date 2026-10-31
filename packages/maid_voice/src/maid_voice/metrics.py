"""Per-turn trace with monotonic timestamps, exported as JSONL + CSV summary."""

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def now_ns() -> int:
    return time.monotonic_ns()


@dataclass
class TurnTrace:
    session_id: str
    utterance_id: str
    generation_id: int
    wall_time_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "mic"  # mic | fixture
    mode: str = "ptt"  # ptt | handsfree | proactive
    speculative: bool = False  # generation started before the turn was committed
    # monotonic ns timestamps
    t_speech_start_ns: int | None = None
    t_speech_end_ns: int | None = None  # last voiced frame (energy based)
    t_endpoint_ns: int | None = None  # hands-free: VAD committed the turn
    t_ptt_release_ns: int | None = None  # push-to-talk release / end of fixture
    t_asr_request_ns: int | None = None
    t_asr_final_ns: int | None = None
    t_llm_request_ns: int | None = None
    t_llm_first_token_ns: int | None = None
    t_first_clause_ns: int | None = None
    t_tts_request_ns: int | None = None
    t_tts_first_audio_ns: int | None = None
    t_playback_start_ns: int | None = None  # estimated DAC time of first sample
    t_playback_end_ns: int | None = None
    t_cancel_ns: int | None = None
    t_playback_halted_ns: int | None = None
    # content / config
    user_text: str = ""
    user_language: str | None = None
    assistant_text: str = ""
    clauses: list[str] = field(default_factory=list)
    asr_model: str = ""
    llm_model: str = ""
    tts_model: str = ""
    voice_profile: str = ""
    output_device: str = ""
    usage: dict | None = None
    est_cost_usd: float | None = None
    cancelled: bool = False
    error: str | None = None

    def mark(self, name: str) -> int:
        t = now_ns()
        setattr(self, f"t_{name}_ns", t)
        return t

    def ms(self, start: str, end: str) -> float | None:
        a, b = getattr(self, f"t_{start}_ns"), getattr(self, f"t_{end}_ns")
        return None if a is None or b is None else round((b - a) / 1e6, 1)

    def summary(self) -> dict:
        """Derived latencies in ms. 'first_sound' is the headline metric."""
        return {
            "first_sound_ms": self.ms("speech_end", "playback_start"),
            "release_to_sound_ms": self.ms("ptt_release", "playback_start"),
            "endpoint_ms": self.ms("speech_end", "endpoint"),
            "endpoint_to_sound_ms": self.ms("endpoint", "playback_start"),
            "asr_ms": self.ms("asr_request", "asr_final"),
            "llm_ttft_ms": self.ms("llm_request", "llm_first_token"),
            "llm_first_clause_ms": self.ms("llm_request", "first_clause"),
            "tts_first_audio_ms": self.ms("tts_request", "tts_first_audio"),
            "playback_lag_ms": self.ms("tts_first_audio", "playback_start"),
            "stop_ms": self.ms("cancel", "playback_halted"),
        }


SUMMARY_FIELDS = [
    "wall_time_utc", "utterance_id", "generation_id", "source", "mode", "speculative",
    "user_language", "user_text", "assistant_text", "first_sound_ms", "release_to_sound_ms",
    "endpoint_ms", "endpoint_to_sound_ms", "asr_ms", "llm_ttft_ms",
    "llm_first_clause_ms", "tts_first_audio_ms", "playback_lag_ms", "stop_ms", "cancelled",
    "error", "asr_model", "llm_model", "tts_model", "voice_profile", "est_cost_usd",
]


class MetricsSink:
    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = run_dir / "turns.jsonl"
        self.csv = run_dir / "summary.csv"

    def write(self, trace: TurnTrace) -> dict:
        record = {**asdict(trace), **trace.summary()}
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        new = not self.csv.exists()
        with self.csv.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(record)
        return record
