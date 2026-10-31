"""Mine ナツメ's game lines into an expression bank: "when <situation>, she says it like <style>".

Idea from MaiBot's expression learner (learn_style.prompt), applied to one character's own lines
instead of group chat. Uses the same line filter as the voice fine-tune (no adult scenes, no
non-verbal lines), walks each scene in order in windows, and asks the LLM for situation/style/example.
Then embeds each situation with multilingual-e5-small for retrieval at reply time.

  uv run --group eval scripts/build_expressions.py [--window 14] [--limit 0]
Writes ../maid_agent/persona/natsume_expressions.jsonl and .npy (embeddings); the raw mined set is kept
in natsume_expressions_raw.jsonl, so re-deduplicating does not call the LLM again (--remine forces it).
"""

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from sft_prep import NAT, PUNCT, clean, reject  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from maid_voice.config import load_config  # noqa: E402
from maid_voice.persona import Embedder  # noqa: E402

OUT = ROOT.parent / "maid_agent" / "persona"
# the game's cast and café: she only ever talks to あきら, so lines naming them are never used
GAME_NAMES = re.compile(r"高嶺|昂晴|火打谷|愛衣|明月|栞那|墨染|汐山|涼音|四季|ナツメ|ステラ")

PROMPT = """以下は、あるキャラクター（ナツメ）が会話の中で実際に話したセリフを、場面の順に並べたものです。相手のセリフは含まれていません。
ナツメの話し方の癖を、「どんな状況で」「どう言うか」の組として抽出してください。

{lines}

ルール:
1. situation は会話の状況を一般的に書く（20字以内）。例:「相手が疲れていると言ったとき」「褒められて照れたとき」
2. style はナツメの言い回し・語尾・口調の特徴（25字以内）。例:「『別に』で始めて素っ気なく返す」「語尾に『でしょ』を付けて念を押す」
3. example は上のセリフから一つをそのまま引用（40字以内に収まるもの）
4. 人名・店名・作品固有の名詞は situation と style に入れない。固有名詞を含むセリフは example に選ばない
5. 性的な内容のセリフは使わない
6. 日常会話（雑談、気遣い、からかい、照れ、呆れ、相槌、頼みごと等）で再利用できる状況を優先する。物語特有の状況（脅し、戦い、事件など）や、咳・悲鳴などの声だけのセリフは選ばない
7. 3〜6組。似た組を重複させない

JSONの配列だけを出力してください:
[{{"situation": "...", "style": "...", "example": "...", "id": "セリフ番号"}}]"""


REVIEW = """次の言い回しリストを確認し、日常の話し相手（ユーザー「あきら」と話すキャラクター）の言い回しとして使ってよいか判定してください。

{items}

次のどれかに当てはまるものは keep=false:
- 性的・恋愛の親密な場面の内容（キス、ベッド、身体的な行為、告白など）
- 例文に人名・店名・作品固有の名詞が含まれる（例: 昂晴、高嶺君、火打谷さん、ステラ）
- 物語特有で日常会話に使えない状況（死神、記憶、事件など）

JSONの配列だけを出力: [{{"id": 番号, "keep": true/false}}]"""


async def review(client: httpx.AsyncClient, model: str, extra: dict, batch: list[tuple[int, dict]],
                 sem: asyncio.Semaphore) -> dict[int, bool]:
    listing = "\n".join(f"[{i}] 状況:{it['situation']} / 言い方:{it['style']} / 例:「{it['example']}」" for i, it in batch)
    body = {"model": model, "messages": [{"role": "user", "content": REVIEW.format(items=listing)}],
            "max_tokens": 1500, "temperature": 0, **extra}
    async with sem:
        for attempt in range(3):
            try:
                r = await client.post("/chat/completions", json=body)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
                verdict = {int(v["id"]): bool(v["keep"]) for v in json.loads(re.search(r"\[.*\]", text, re.S).group(0))}
                # anything the reviewer skipped is dropped: unreviewed lines don't go into the persona
                return {i: verdict.get(i, False) for i, _ in batch}
            except Exception as e:  # noqa: BLE001 - retry any API/parse failure
                if attempt == 2:
                    print(f"  review batch at {batch[0][0]}: failed ({e}), dropping it", flush=True)
                    return {i: False for i, _ in batch}


def load_lines() -> dict[str, list[tuple[str, str]]]:
    """Clean, filtered lines grouped by scene (nat013_...), in order."""
    scenes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for lab in sorted(NAT.glob("*.lab")):
        text = clean(lab.read_text(encoding="utf-8"))
        dur = sf.info(lab.with_suffix(".ogg")).duration
        if reject(text, PUNCT.sub("", text), dur) is None:
            scenes[lab.stem.split("_")[0]].append((lab.stem, text))
    return scenes


async def mine(client: httpx.AsyncClient, model: str, extra: dict, window: list[tuple[str, str]],
               sem: asyncio.Semaphore) -> list[dict]:
    lines = "\n".join(f"[{i}] {t}" for i, (_, t) in enumerate(window))
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT.format(lines=lines)}],
            "max_tokens": 900, "temperature": 0.3, **extra}
    async with sem:
        for attempt in range(3):
            try:
                r = await client.post("/chat/completions", json=body)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
                items = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
                break
            except Exception as e:  # noqa: BLE001 - retry any API/parse failure
                if attempt == 2:
                    print(f"  window {window[0][0]}: failed ({e})", flush=True)
                    return []
    out = []
    for it in items:
        try:
            src = window[int(it.get("id", -1))]
        except (ValueError, IndexError):
            src = None
        ex = str(it.get("example", "")).strip()
        if not (it.get("situation") and it.get("style") and ex):
            continue
        # keep the example only if it really is one of her lines (no paraphrases)
        if len(ex) > 40 or not any(ex.rstrip("。") in t for _, t in window):
            continue
        out.append({"situation": it["situation"].strip(), "style": it["style"].strip(), "example": ex,
                    "source": src[0] if src else None})
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="only the first N windows (trial)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dedupe", type=float, default=0.97, help="cosine above which two pairs count as the same")
    ap.add_argument("--remine", action="store_true", help="call the LLM again even if the raw mined set exists")
    args = ap.parse_args()

    raw_path = OUT / "natsume_expressions_raw.jsonl"
    if raw_path.exists() and not args.remine and not args.limit:
        items = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
        print(f"{len(items)} mined pairs from {raw_path.name}", flush=True)
    else:
        items = await mine_all(args)
        OUT.mkdir(parents=True, exist_ok=True)
        raw_path.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")
    mined = len(items)

    if any("keep" not in it for it in items):  # content review, cached in the raw file
        verdicts = await review_all(items)
        for i, it in enumerate(items):
            it["keep"] = verdicts[i]
        raw_path.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")
    items = [it for it in items if it["keep"] and not GAME_NAMES.search(it["example"] + it["style"])]
    print(f"{len(items)} of {mined} pass the content review (no intimate scenes, no game names)", flush=True)

    # drop near-duplicate situations+styles (same meaning mined from different scenes)
    emb = Embedder()
    vecs = emb.passages([f"{it['situation']} {it['style']}" for it in items])
    keep: list[int] = []
    for i in range(len(items)):
        if all(float(vecs[i] @ vecs[j]) < args.dedupe for j in keep):
            keep.append(i)
    items = [items[i] for i in keep]
    sit_vecs = emb.passages([it["situation"] for it in items])

    with open(OUT / "natsume_expressions.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            it = {k: v for k, v in it.items() if k != "keep"}
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    np.save(OUT / "natsume_expressions.npy", sit_vecs.astype(np.float32))
    print(f"{len(items)} expressions (from {mined} mined, near-duplicates above {args.dedupe} removed) -> {OUT}")


async def review_all(items: list[dict], batch_size: int = 40) -> dict[int, bool]:
    cfg = load_config()
    llm = cfg["llm"]
    numbered = list(enumerate(items))
    batches = [numbered[i:i + batch_size] for i in range(0, len(numbered), batch_size)]
    sem = asyncio.Semaphore(8)
    async with httpx.AsyncClient(base_url=cfg.secret(llm["base_url_env"]), timeout=120,
                                 headers={"Authorization": f"Bearer {cfg.secret(llm['api_key_env'])}"}) as client:
        results = await asyncio.gather(*(review(client, llm["model"], llm.get("extra_body", {}), b, sem)
                                         for b in batches))
    return {i: keep for r in results for i, keep in r.items()}


async def mine_all(args) -> list[dict]:
    cfg = load_config()
    llm = cfg["llm"]
    scenes = load_lines()
    windows = [lines[i:i + args.window] for lines in scenes.values() for i in range(0, len(lines), args.window)]
    windows = [w for w in windows if len(w) >= 4]
    if args.limit:
        windows = windows[: args.limit]
    print(f"{sum(len(v) for v in scenes.values())} lines in {len(scenes)} scenes -> {len(windows)} windows", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(base_url=cfg.secret(llm["base_url_env"]), timeout=120,
                                 headers={"Authorization": f"Bearer {cfg.secret(llm['api_key_env'])}"}) as client:
        results = await asyncio.gather(*(mine(client, llm["model"], llm.get("extra_body", {}), w, sem)
                                         for w in windows))
    return [it for r in results for it in r]


if __name__ == "__main__":
    asyncio.run(main())
