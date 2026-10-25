"""Phase 0 probe: streaming time-to-first-token for an OpenAI-compatible chat endpoint.

Usage: uv run scripts/probe_llm.py [--runs N] [--effort low] [--thinking off]
Reads BASE_URL / API_KEY from ../.env. Prints per-run TTFT, first-clause time and total.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

SYSTEM = (
    "あなたは音声アシスタント「ナツメ」です。ユーザーが英語で話しても必ず日本語で答えてください。"
    "返答は短く、話し言葉で、最初の一文は特に短くしてください。記号や絵文字は使わないでください。"
)
PROMPTS = [
    "Did my background task finish?",
    "今日はちょっと疲れたよ。",
    "What should I eat for dinner tonight?",
    "明日の天気ってどうかな？",
    "Can you remind me what we were talking about?",
]
CLAUSE_END = set("。！？!?、")


def run_once(client: httpx.Client, model: str, prompt: str, extra: dict) -> dict:
    body = {
        "model": model,
        "stream": True,
        "max_tokens": 120,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        **extra,
    }
    if "max_completion_tokens" in body:
        body.pop("max_tokens")
    t0 = time.monotonic()
    t_first = t_clause = None
    text = ""
    reasoning_chars = 0
    usage = None
    with client.stream("POST", "/chat/completions", json=body) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.read()[:400]!r}")
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices", []):
                delta = ch.get("delta", {})
                reasoning_chars += len(delta.get("reasoning_content") or "")
                piece = delta.get("content") or ""
                if piece and t_first is None:
                    t_first = time.monotonic()
                text += piece
                if t_clause is None and any(c in CLAUSE_END for c in piece):
                    t_clause = time.monotonic()
    t_end = time.monotonic()
    ms = lambda t: None if t is None else round((t - t0) * 1000)
    return {
        "prompt": prompt,
        "ttft_ms": ms(t_first),
        "first_clause_ms": ms(t_clause),
        "total_ms": ms(t_end),
        "reasoning_chars": reasoning_chars,
        "text": text,
        "usage": usage,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--runs", type=int, default=1, help="passes over the prompt list")
    ap.add_argument("--extra", default="{}", help="JSON merged into the request body")
    ap.add_argument("--base-url", default=None, help="default: BASE_URL from .env")
    ap.add_argument("--key-env", default="API_KEY", help="env var holding the key (first of a comma list)")
    args = ap.parse_args()

    client = httpx.Client(
        base_url=args.base_url or os.environ["BASE_URL"],
        headers={"Authorization": f"Bearer {os.environ[args.key_env].split(',')[0].strip()}"},
        timeout=30,
        http2=False,
    )
    extra = json.loads(args.extra)
    results = []
    for _ in range(args.runs):
        for p in PROMPTS:
            res = run_once(client, args.model, p, extra)
            results.append(res)
            print(json.dumps(res, ensure_ascii=False))
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    clauses = [r["first_clause_ms"] for r in results if r["first_clause_ms"] is not None]
    if ttfts:
        print(f"TTFT ms  p50={statistics.median(ttfts):.0f} min={min(ttfts)} max={max(ttfts)} n={len(ttfts)}")
    if clauses:
        print(f"clause ms p50={statistics.median(clauses):.0f} min={min(clauses)} max={max(clauses)}")


if __name__ == "__main__":
    main()
