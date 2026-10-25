"""Reproduce a reported artifact: synthesize one phrase N times per config and save the takes.

  uv run scripts/artifact_probe.py "そういう生き方" --n 8 --configs q17:a q17:e q17:a:ns q17bf:a q06:a
"""

import argparse
import base64
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {
    "q06": "mlx-community--Qwen3-TTS-12Hz-0.6B-Base-8bit",
    "q17": "mlx-community--Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "q17bf": "mlx-community--Qwen3-TTS-12Hz-1.7B-Base-bf16",
}


def synth(client, model, text, ref, interval, temperature):
    body = {"model": model, "input": text, "language": "japanese", "response_format": "wav",
            "ref_audio": base64.b64encode((ROOT / f"voices/natsume/ref_{ref}.wav").read_bytes()).decode(),
            "ref_text": (ROOT / f"voices/natsume/ref_{ref}.txt").read_text(encoding="utf-8")}
    if temperature is not None:
        body["temperature"] = temperature
    if interval != "ns":
        body.update(stream=True, streaming_interval=float(interval))
    r = client.post("/v1/audio/speech", json=body)
    r.raise_for_status()
    sr = int.from_bytes(r.content[24:28], "little")
    return np.frombuffer(r.content[44:], dtype=np.int16).astype("float32") / 32768, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--configs", nargs="+", default=["q17:a"], help="MODEL:REF[:INTERVAL][:TEMP]")
    ap.add_argument("--out", default="runs/artifact")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(base_url="http://127.0.0.1:9800", timeout=300)
    for spec in args.configs:
        model, ref, *rest = spec.split(":")
        interval = rest[0] if rest else "0.2"
        temp = float(rest[1]) if len(rest) > 1 else None
        for k in range(args.n):
            pcm, sr = synth(client, ALIASES.get(model, model), args.text, ref, interval, temp)
            sf.write(out / f"{spec.replace(':', '_')}_{k}.wav", pcm, sr)
        print(f"{spec}: {args.n} takes", flush=True)


if __name__ == "__main__":
    main()
