"""Build the Qwen3-TTS fine-tuning set from ../nat (ogg + lab) -> runs/sft/data.

Keeps clean spoken lines only: 1-15 s, sane speaking rate, no non-verbal vocalisation
(moans, panting, voiced-mark kana like あ゛), which would teach the model breathy noise.
Writes 24 kHz wavs, train_raw.jsonl / val_raw.jsonl (relative paths, one shared ref.wav),
and dropped.tsv with the reason for every rejected line.

  uv run scripts/sft_prep.py [--val 80]
"""

import argparse
import ast
import json
import random
import re
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from build_reference import resample_half, trim

ROOT = Path(__file__).resolve().parents[1]
NAT = ROOT.parent / "nat"
OUT = ROOT / "runs" / "sft" / "data"
SR = 24000

RUBY = re.compile(r"\[([^\]]+)\](.)")  # [reading]kanji -> reading
PUNCT = re.compile(r"[、。！？…―\s　・「」『』“”～♪!?.,ー]")
NONVERBAL = [
    (re.compile(r"゛"), "voiced-mark"),
    (re.compile(r"[●♡❤]"), "censor/heart"),
    (re.compile(r"[ぁぃぅぇぉ]{2,}|[ァィゥェォ]{2,}"), "small-kana run"),
    (re.compile(r"(あっ、?){3,}|(んっ、?){2,}|(はぁ[…、]*){2,}|ッッ"), "panting/moan"),
    (re.compile(r"[んぁあはひふぅうおぉ]{5,}"), "drawn-out vowel"),
]


def clean(text: str) -> str:
    text = text.replace("\\\\n", "").replace("\\n", "").strip()
    text = RUBY.sub(r"\1", text)
    return text if text[-1:] in "。！？!?…" else text + "。"


def reject(text: str, core: str, dur: float) -> str | None:
    if not 1.0 <= dur <= 15.0:
        return f"duration {dur:.1f}s"
    if len(core) < 3:
        return "too short"
    for pat, why in NONVERBAL:
        if pat.search(text):
            return why
    rate = len(core) / dur
    if not 1.2 <= rate <= 14.0:
        return f"rate {rate:.1f} chars/s"
    # interjection-heavy lines (mostly あ/ん/は/ふ sounds) are breath, not speech
    if sum(c in "あんはひふぅうおぉぁっッ" for c in core) / len(core) > 0.5:
        return "interjection-heavy"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=int, default=80, help="held-out lines for validation loss")
    args = ap.parse_args()

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "wav").mkdir(parents=True)
    shutil.copy(ROOT / "voices/natsume/ref_a.wav", OUT / "ref.wav")
    shutil.copy(ROOT / "voices/natsume/ref_a.txt", OUT / "ref.txt")
    # the tts_eval sentences, so checkpoints are judged on the same text as earlier experiments
    tree = ast.parse((ROOT / "scripts/tts_eval.py").read_text(encoding="utf-8"))
    sentences = next(ast.literal_eval(n.value) for n in tree.body
                     if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SENTENCES")
    (OUT / "eval_sentences.json").write_text(json.dumps(sentences, ensure_ascii=False, indent=1), encoding="utf-8")

    kept, dropped = [], []
    for lab in sorted(NAT.glob("*.lab")):
        text = clean(lab.read_text(encoding="utf-8"))
        audio, sr = sf.read(lab.with_suffix(".ogg"), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr == 2 * SR:
            audio = resample_half(audio)
        elif sr != SR:
            dropped.append((lab.stem, f"sample rate {sr}", text))
            continue
        audio = trim(audio, 0.003)
        dur = len(audio) / SR
        why = reject(text, PUNCT.sub("", text), dur)
        if why:
            dropped.append((lab.stem, why, text))
            continue
        peak = float(np.abs(audio).max())
        if peak > 0.99:
            audio *= 0.97 / peak
        # 100 ms of silence on both ends, so the model learns clean starts and stops
        pad = np.zeros(SR // 10, dtype="float32")
        sf.write(OUT / "wav" / f"{lab.stem}.wav", np.concatenate([pad, audio, pad]), SR, subtype="PCM_16")
        kept.append({"audio": f"wav/{lab.stem}.wav", "text": text, "ref_audio": "ref.wav", "dur": round(dur + 0.2, 2)})

    random.Random(0).shuffle(kept)
    val, train = kept[: args.val], kept[args.val :]
    for name, rows in (("train_raw.jsonl", train), ("val_raw.jsonl", val)):
        with open(OUT / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT / "dropped.tsv", "w", encoding="utf-8") as f:
        f.writelines(f"{s}\t{w}\t{t}\n" for s, w, t in dropped)

    reasons: dict[str, int] = {}
    for _, w, _ in dropped:
        key = w.split(" ")[0]
        reasons[key] = reasons.get(key, 0) + 1
    hours = sum(r["dur"] for r in kept) / 3600
    print(f"kept {len(kept)} lines ({hours:.2f} h): train {len(train)}, val {len(val)}")
    print(f"dropped {len(dropped)}: {reasons}")


if __name__ == "__main__":
    main()
