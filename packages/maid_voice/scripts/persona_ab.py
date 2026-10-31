"""Blind A/B: built-in prompt (old) vs role card + expressions (new), same prompts, same voice.

Per prompt, A/B order is random; key in KEY_dont_peek.txt. Writes the replies as text and as
speech (natsume_ft), plus timing: persona prompt build and LLM first token for both.

  uv run scripts/persona_ab.py
"""

import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maid_voice.app import load_persona  # noqa: E402
from maid_voice.config import load_config  # noqa: E402
from maid_voice.llm import OpenAICompatLLM  # noqa: E402
from maid_voice.tts import OmlxTTS  # noqa: E402

PROMPTS = [
    "I'm so tired today.", "今日はちょっと疲れたよ。", "What should I eat for dinner?",
    "ねえ、聞いてる？", "I finally fixed that bug!", "ごめん、待たせちゃった。",
    "Do you think I'm working too much?", "今日のご飯、自分で作ってみたんだ。",
    "Can you remind me to sleep early tonight?", "暇だなあ。",
    "Thanks for keeping me company.", "ナツメって可愛いよね。",
    "I lost three games in a row.", "明日の朝、早いんだよね。",
    "What do you think of my new keyboard?", "ちょっと相談してもいい？",
    "Did the download finish?", "今日はもう寝るね。", "You're kind of cold sometimes.",
    "なんか元気出ないなあ。",
]


async def run(llm: OpenAICompatLLM, text: str) -> tuple[str, float, float]:
    t0 = time.monotonic()
    first = None
    out = ""
    async for d in llm.stream_response([], text):
        if d.text:
            first = first or time.monotonic()
            out += d.text
    return out, (first - t0) * 1000 if first else float("nan"), 0.0


async def main() -> None:
    cfg = load_config()
    l, t = cfg["llm"], cfg["tts"]
    common = dict(base_url=cfg.secret(l["base_url_env"]), api_key=cfg.secret(l["api_key_env"]), model=l["model"],
                  max_tokens=l["max_tokens"], extra_body=l.get("extra_body", {}))
    old = OpenAICompatLLM(**common)
    persona = load_persona(cfg)
    new = OpenAICompatLLM(**common, persona=persona)
    tts = OmlxTTS(t["base_url"], t["model"], t.get("streaming_interval", 0.2), 24000, t.get("temperature"))
    voice = cfg.voices["natsume_ft"]
    out = ROOT / "runs" / "persona_ab"
    out.mkdir(parents=True, exist_ok=True)

    build_ms = []
    for p in PROMPTS[:3]:  # warm the embedder
        persona.system_prompt(p)
    for p in PROMPTS:
        t0 = time.perf_counter()
        persona.system_prompt(p)
        build_ms.append((time.perf_counter() - t0) * 1000)

    rng = random.Random(1025)
    rows, key = [], []
    ttft = {"old": [], "new": []}
    for i, p in enumerate(PROMPTS):
        replies = {}
        for name, llm in (("old", old), ("new", new)):
            text, ms, _ = await run(llm, p)
            replies[name] = text
            ttft[name].append(ms)
        expr = [f"{x.situation}：{x.example}" for x in persona.last_expressions]
        order = ["old", "new"]
        rng.shuffle(order)
        key.append(f"{i:02d}: A={order[0]} B={order[1]}")
        for letter, name in zip("AB", order):
            pcm = np.concatenate([c async for c in tts.stream_audio(replies[name], voice)])
            sf.write(out / f"{i:02d}_{letter}.wav", pcm, 24000)
        rows.append({"i": i, "prompt": p, "A": replies[order[0]], "B": replies[order[1]], "new_expressions": expr})
        print(f"{i:02d} {p}\n   A: {replies[order[0]]}\n   B: {replies[order[1]]}", flush=True)

    (out / "replies.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "replies_blind.txt").write_text(
        "\n\n".join(f"{r['i']:02d} {r['prompt']}\n  A: {r['A']}\n  B: {r['B']}" for r in rows) + "\n", encoding="utf-8")
    (out / "KEY_dont_peek.txt").write_text("\n".join(key) + "\n")
    med = statistics.median
    print(f"\npersona prompt build: p50 {med(build_ms):.1f} ms, max {max(build_ms):.1f} ms (embedding + retrieval)")
    print(f"LLM first token p50: old {med(ttft['old']):.0f} ms, new {med(ttft['new']):.0f} ms "
          f"(new includes the prompt build; longer system prompt)")
    print(f"wrote {out}")
    await old.aclose()
    await new.aclose()
    await tts.aclose()


if __name__ == "__main__":
    asyncio.run(main())
