"""Streaming LLM adapters (OpenAI-compatible chat completions)."""

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator, Protocol

import httpx

if TYPE_CHECKING:
    from .persona import Persona

SYSTEM_PROMPT_JA = (
    "あなたは音声アシスタント「ナツメ」です。口調は少しそっけないけれど、根は優しく面倒見のいい女の子です。"
    "相手を突き放さず、親しい相手に話すくだけた日本語で、気遣いを一言添えて答えてください。"
    "ユーザーが英語で話しても、返答は必ず日本語にしてください。"
    "返答は音声で読み上げられます。一、二文で短く答え、最初の一文は特に短くしてください。"
    "記号、絵文字、箇条書き、英字の略語は使わないでください。"
    "天気、時刻、タスクの状態など、まだ確認する手段がない情報は作らないでください。"
    "その場合は「まだ見られないの」のように柔らかく伝え、どうすればいいかを短く提案してください。"
)


@dataclass
class LLMDelta:
    text: str = ""
    usage: dict | None = None
    done: bool = False


@dataclass
class ChatTurn:
    role: str
    content: str


class LLMAdapter(Protocol):
    model: str

    def stream_response(self, history: list[ChatTurn], user_text: str) -> AsyncIterator[LLMDelta]: ...


@dataclass
class OpenAICompatLLM:
    base_url: str
    api_key: str
    model: str
    max_tokens: int = 120
    extra_body: dict = field(default_factory=dict)
    system_prompt: str = SYSTEM_PROMPT_JA  # used when no persona is configured
    persona: "Persona | None" = None
    # USD per 1M tokens, for rough cost estimates only; verify against provider pricing
    price_in: float = 0.0
    price_out: float = 0.0

    def __post_init__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=httpx.Timeout(30, connect=5),
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    async def warmup(self) -> None:
        # opens and keeps the TLS connection alive; the reply is discarded
        async for _ in self.stream_response([], "hi"):
            pass

    async def stream_response(self, history: list[ChatTurn], user_text: str) -> AsyncIterator[LLMDelta]:
        system = await asyncio.to_thread(self.persona.system_prompt, user_text) if self.persona else self.system_prompt
        messages = [{"role": "system", "content": system}]
        messages += [{"role": t.role, "content": t.content} for t in history]
        messages.append({"role": "user", "content": user_text})
        body = {
            "model": self.model, "messages": messages, "stream": True,
            "max_tokens": self.max_tokens, "stream_options": {"include_usage": True},
            **self.extra_body,
        }
        async with self._client.stream("POST", "/chat/completions", json=body) as r:
            if r.status_code != 200:
                raise RuntimeError(f"LLM HTTP {r.status_code}: {(await r.aread())[:300]!r}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    yield LLMDelta(usage=chunk["usage"])
                for choice in chunk.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        yield LLMDelta(text=piece)
        yield LLMDelta(done=True)

    def estimate_cost(self, usage: dict | None) -> float | None:
        if not usage or not (self.price_in or self.price_out):
            return None
        return (usage.get("prompt_tokens", 0) * self.price_in
                + usage.get("completion_tokens", 0) * self.price_out) / 1e6

    async def aclose(self) -> None:
        await self._client.aclose()
