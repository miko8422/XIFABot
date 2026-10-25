"""Streaming TTS adapters."""

import base64
from typing import AsyncIterator, Protocol

import httpx
import numpy as np

from .config import VoiceProfile

WAV_HEADER_BYTES = 44
EDGE_MS = 8  # fade length at clause start/end, so clause joins cannot click


class TTSAdapter(Protocol):
    model: str
    sample_rate: int
    genuinely_streams: bool

    async def warmup(self, voice: VoiceProfile) -> None: ...
    def stream_audio(self, text: str, voice: VoiceProfile) -> AsyncIterator[np.ndarray]: ...


class OmlxTTS:
    """oMLX /v1/audio/speech with stream=true.

    Phase 0 measured genuine model-native streaming (first PCM ~200 ms, dozens of chunks per
    sentence) rather than whole-sentence buffering, so genuinely_streams=True.
    The reference voice is re-sent per request; oMLX has no conditioning cache endpoint.
    """

    genuinely_streams = True

    def __init__(self, base_url: str, model: str, streaming_interval: float = 0.2,
                 sample_rate: int = 24000, temperature: float | None = None):
        self.model = model
        # Codec-token sampling temperature. The model default (0.9) occasionally samples a bad
        # 80 ms token in smooth vowel glides (heard as a "dropped mic" burst); 0.3-0.5 avoids it.
        self.temperature = temperature
        self.streaming_interval = streaming_interval
        self.sample_rate = sample_rate
        self._client = httpx.AsyncClient(base_url=base_url, timeout=60)
        self._ref_cache: dict[str, str] = {}

    def _ref_b64(self, voice: VoiceProfile) -> str:
        if voice.name not in self._ref_cache:
            self._ref_cache[voice.name] = base64.b64encode(voice.ref_audio.read_bytes()).decode()
        return self._ref_cache[voice.name]

    async def warmup(self, voice: VoiceProfile) -> None:
        async for _ in self.stream_audio("はい。", voice):
            pass

    def model_for(self, voice: VoiceProfile) -> str:
        return voice.model or self.model

    async def stream_audio(self, text: str, voice: VoiceProfile) -> AsyncIterator[np.ndarray]:
        body = {
            "model": self.model_for(voice), "input": text, "language": voice.language,
            "response_format": "wav", "stream": True,
            "streaming_interval": self.streaming_interval,
        }
        if voice.speaker:  # fine-tuned CustomVoice: the speaker is baked into the model
            body["voice"] = voice.speaker
        else:
            body.update(ref_audio=self._ref_b64(voice), ref_text=voice.ref_text)
        if self.temperature is not None:
            body["temperature"] = self.temperature
        header = bytearray()
        carry = b""
        edge = int(self.sample_rate * EDGE_MS / 1000)
        tail = np.zeros(0, dtype=np.int16)  # last `edge` samples, held back for the fade-out
        first = True
        async with self._client.stream("POST", "/v1/audio/speech", json=body) as r:
            if r.status_code != 200:
                raise RuntimeError(f"TTS HTTP {r.status_code}: {(await r.aread())[:300]!r}")
            async for chunk in r.aiter_bytes():
                if len(header) < WAV_HEADER_BYTES:
                    need = WAV_HEADER_BYTES - len(header)
                    header += chunk[:need]
                    chunk = chunk[need:]
                    if len(header) == WAV_HEADER_BYTES:
                        sr = int.from_bytes(header[24:28], "little")
                        if sr != self.sample_rate:
                            raise RuntimeError(f"TTS sample rate {sr} != playback {self.sample_rate}")
                data = carry + chunk
                usable = len(data) - len(data) % 2  # int16 alignment across HTTP chunks
                carry = data[usable:]
                if not usable:
                    continue
                pcm = np.concatenate([tail, np.frombuffer(data[:usable], dtype=np.int16)])
                if first and len(pcm) >= edge:
                    pcm = pcm.copy()
                    pcm[:edge] = (pcm[:edge] * np.linspace(0, 1, edge)).astype(np.int16)
                    first = False
                if len(pcm) > edge:
                    out, tail = pcm[:-edge], pcm[-edge:]
                    yield out.copy()
                else:
                    tail = pcm
        if len(tail):
            yield (tail * np.linspace(1, 0, len(tail))).astype(np.int16)

    async def aclose(self) -> None:
        await self._client.aclose()
