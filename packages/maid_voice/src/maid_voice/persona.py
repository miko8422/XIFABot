"""Role card + expression bank -> the replyer's system prompt.

The role card (TOML, versioned, never rewritten by conversation) holds who she is, how she speaks
and her rules. The expression bank holds "when <situation>, she says it like <style>" pairs mined
from her real lines (scripts/build_expressions.py); the few closest to the current turn are added
to each prompt. Persona data lives in ../maid_agent/persona: it belongs to the agent, not the voice.
"""

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

E5_MODEL = "intfloat/multilingual-e5-small"


class Embedder:
    """multilingual-e5-small on CPU: about 10 ms per short sentence, normalized vectors."""

    def __init__(self, model: str = E5_MODEL):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(model)
        self._model = AutoModel.from_pretrained(model).eval()

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = []
        with self._torch.inference_mode():
            for i in range(0, len(texts), 64):
                batch = self._tok(texts[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors="pt")
                hidden = self._model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).float()
                vec = (hidden * mask).sum(1) / mask.sum(1)
                out.append(self._torch.nn.functional.normalize(vec, dim=-1).numpy())
        return np.concatenate(out) if out else np.zeros((0, 384), dtype=np.float32)

    def query(self, text: str) -> np.ndarray:
        return self._embed([f"query: {text}"])[0]

    def passages(self, texts: list[str]) -> np.ndarray:
        return self._embed([f"passage: {t}" for t in texts])


@dataclass
class Expression:
    situation: str
    style: str
    example: str


@dataclass
class ExpressionBank:
    items: list[Expression]
    vectors: np.ndarray
    embedder: Embedder

    @classmethod
    def load(cls, path: Path, embedder: Embedder) -> "ExpressionBank":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        items = [Expression(r["situation"], r["style"], r["example"]) for r in rows]
        return cls(items, np.load(path.with_suffix(".npy")), embedder)

    def top(self, context: str, k: int = 4) -> list[Expression]:
        if not self.items:
            return []
        scores = self.vectors @ self.embedder.query(context)
        return [self.items[i] for i in np.argsort(-scores)[:k]]


@dataclass
class RoleCard:
    version: str
    user_name: str
    identity: str
    speech: str
    voice_rules: str
    boundaries: str
    behavior: str = ""
    examples: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "RoleCard":
        d = tomllib.loads(path.read_text(encoding="utf-8"))
        return cls(version=d["version"], user_name=d["user_name"], identity=d["identity"].strip(),
                   speech=d["speech"].strip(), voice_rules=d["voice_rules"].strip(),
                   boundaries=d["boundaries"].strip(), behavior=d.get("behavior", "").strip(),
                   examples=d.get("examples", []))

    def replyer_prompt(self, expressions: list[Expression] = ()) -> str:
        """System prompt for answering あきら directly (the planner gets `behavior` instead)."""
        parts = [f"# あなたについて\n{self.identity}", f"# 話し方\n{self.speech}",
                 f"# 音声で話すときの決まり\n{self.voice_rules}", f"# 守ること\n{self.boundaries}"]
        if self.examples:
            parts.append("# 普段のセリフの例（口調の参考。内容はそのまま使わない）\n"
                         + "\n".join(f"「{e}」" for e in self.examples))
        if expressions:
            parts.append("# 今の場面に合いそうな言い回し（自然に合うときだけ使う）\n"
                         + "\n".join(f"- {x.situation}：{x.style}（例「{x.example}」）" for x in expressions))
        return "\n\n".join(parts)


class Persona:
    """What the LLM adapter calls per turn: role card + the expressions closest to what あきら said."""

    def __init__(self, card_path: Path, expressions_path: Path | None = None, top_k: int = 4):
        self.card = RoleCard.load(card_path)
        self.top_k = top_k
        self.bank = ExpressionBank.load(expressions_path, Embedder()) if expressions_path else None
        self.last_expressions: list[Expression] = []

    def system_prompt(self, user_text: str) -> str:
        """Blocking (about 10 ms of CPU for the embedding); call it off the event loop."""
        self.last_expressions = self.bank.top(user_text, self.top_k) if self.bank and user_text else []
        return self.card.replyer_prompt(self.last_expressions)
