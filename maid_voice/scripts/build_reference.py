"""Join reference clips into one cloning reference (24 kHz wav + transcript).

Clip ids are either system-voice numbers (01..11 -> voices/natsume/clips/NA_sys_XX.wav)
or game-line stems from the labeled dataset (nat013_034 -> ../nat/nat013_034.ogg + .lab).

Usage: uv run scripts/build_reference.py NAME ID [ID ...] [--gap-ms 250]
Writes voices/natsume/ref_NAME.wav and ref_NAME.txt.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = ROOT / "voices" / "natsume"
NAT_DIR = ROOT.parent / "nat"
TARGET_SR = 24000


def load_sys_transcripts() -> dict[str, str]:
    lines = (VOICE_DIR / "natsume_system_voice_transcript.txt").read_text(encoding="utf-8").splitlines()
    return dict(line.split("\t", 1) for line in lines if "\t" in line)


def resample_half(audio: np.ndarray) -> np.ndarray:
    """48 kHz -> 24 kHz with a windowed-sinc low-pass (no scipy dependency)."""
    taps = 63
    n = np.arange(taps) - taps // 2
    h = np.sinc(n / 2) * np.hamming(taps) / 2
    return np.convolve(audio, h, mode="same")[::2].astype("float32")


def load_clip(clip_id: str, sys_texts: dict[str, str]) -> tuple[np.ndarray, str]:
    if clip_id.startswith("nat"):
        audio, sr = sf.read(NAT_DIR / f"{clip_id}.ogg", dtype="float32")
        text = (NAT_DIR / f"{clip_id}.lab").read_text(encoding="utf-8").strip()
    else:
        fname = f"NA_sys_{int(clip_id):02d}.wav"
        audio, sr = sf.read(VOICE_DIR / "clips" / fname, dtype="float32")
        text = sys_texts[fname]
    if sr == 2 * TARGET_SR:
        audio = resample_half(audio)
    elif sr != TARGET_SR:
        raise ValueError(f"{clip_id}: unsupported sample rate {sr}")
    if text[-1] not in "。！？!?…":
        text += "。"
    return audio, text


def trim(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    loud = np.flatnonzero(np.abs(audio) > threshold)
    return audio[loud[0] : loud[-1] + 1] if loud.size else audio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--gap-ms", type=int, default=250)
    args = ap.parse_args()

    sys_texts = load_sys_transcripts()
    parts, texts = [], []
    for clip_id in args.ids:
        audio, text = load_clip(clip_id, sys_texts)
        parts += [trim(audio), np.zeros(int(TARGET_SR * args.gap_ms / 1000), dtype="float32")]
        texts.append(text)
    joined = np.concatenate(parts[:-1])
    sf.write(VOICE_DIR / f"ref_{args.name}.wav", joined, TARGET_SR, subtype="PCM_16")
    (VOICE_DIR / f"ref_{args.name}.txt").write_text("".join(texts), encoding="utf-8")
    print(f"ref_{args.name}: {len(joined) / TARGET_SR:.2f}s  {''.join(texts)}")


if __name__ == "__main__":
    main()
