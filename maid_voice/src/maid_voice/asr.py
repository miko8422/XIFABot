"""ASR adapters. Phase 1: utterance-level transcription (push-to-talk)."""

import io
from dataclasses import dataclass
from typing import Protocol

import httpx
import numpy as np
import soundfile as sf

LANG_CODES = {"japanese": "ja", "english": "en", "chinese": "zh"}


@dataclass
class FinalTranscript:
    text: str
    language: str | None  # ISO-ish code when known


class ASRAdapter(Protocol):
    model: str

    async def warmup(self) -> None: ...
    async def transcribe(self, pcm: np.ndarray, sample_rate: int) -> FinalTranscript: ...


class OmlxASR:
    """oMLX /v1/audio/transcriptions. Accepts a whole utterance, so it is not incremental."""

    def __init__(self, base_url: str, model: str, language: str = "", prompt: str = ""):
        self.model = model
        self.language = language
        self.prompt = prompt
        self._client = httpx.AsyncClient(base_url=base_url, timeout=30)

    async def warmup(self) -> None:
        await self.transcribe(np.zeros(16000, dtype="float32"), 16000)

    async def transcribe(self, pcm: np.ndarray, sample_rate: int) -> FinalTranscript:
        buf = io.BytesIO()
        sf.write(buf, pcm, sample_rate, format="WAV", subtype="PCM_16")
        data = {"model": self.model, "response_format": "verbose_json"}
        if self.language:
            data["language"] = self.language
        if self.prompt:
            data["prompt"] = self.prompt
        r = await self._client.post("/v1/audio/transcriptions", data=data,
                                    files={"file": ("utt.wav", buf.getvalue(), "audio/wav")})
        r.raise_for_status()
        body = r.json()
        lang = (body.get("language") or "").lower()
        return FinalTranscript(text=body.get("text", "").strip(), language=LANG_CODES.get(lang, lang or None))

    async def aclose(self) -> None:
        await self._client.aclose()
