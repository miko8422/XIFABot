"""Phase 0 probe: does oMLX TTS really stream? Measures first-PCM time vs full synthesis.

Usage: uv run scripts/probe_tts.py --model M --ref a [--interval 0.2] [--no-stream] [--runs 2]
Saves outputs to runs/tts_probe/ for listening.
"""

import argparse
import base64
import json
import statistics
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OMLX = "http://127.0.0.1:9800"
SENTENCES = [
    "お疲れさま。ゆっくり休んでね。",
    "まだ確認していないわ。どのタスクか教えてくれる？",
    "うーん、今日は和食がいいんじゃない？お魚とかどうかしら。",
    "ちょっと待って、今調べてみるわね。",
]
WAV_HEADER = 44


def synth(client, model, text, ref, stream, interval):
    body = {"model": model, "input": text, "language": "japanese", "response_format": "wav"}
    if ref:
        body["ref_audio"] = base64.b64encode((ROOT / f"voices/natsume/ref_{ref}.wav").read_bytes()).decode()
        body["ref_text"] = (ROOT / f"voices/natsume/ref_{ref}.txt").read_text(encoding="utf-8")
    if stream:
        body["stream"] = True
        if interval:
            body["streaming_interval"] = interval
    t0 = time.monotonic()
    t_first = None
    buf = bytearray()
    arrivals = []  # (ms since t0, cumulative pcm bytes)
    with client.stream("POST", "/v1/audio/speech", json=body) as r:
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code}: {r.read()[:400]!r}")
        for chunk in r.iter_bytes():
            buf += chunk
            if len(buf) > WAV_HEADER and t_first is None:
                t_first = time.monotonic()
            arrivals.append((round((time.monotonic() - t0) * 1000), len(buf) - WAV_HEADER))
    t_end = time.monotonic()
    sr = int.from_bytes(buf[24:28], "little")
    pcm = np.frombuffer(bytes(buf[WAV_HEADER:]), dtype=np.int16)
    return {
        "first_audio_ms": round((t_first - t0) * 1000),
        "total_ms": round((t_end - t0) * 1000),
        "audio_s": round(len(pcm) / sr, 2),
        "sr": sr,
        "n_chunks": len(arrivals),
        "rtf": round((t_end - t0) / (len(pcm) / sr), 2),
    }, pcm, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", default=None, help="a / b / all, or omit for no clone")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--interval", type=float, default=None)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    out = ROOT / "runs" / "tts_probe"
    out.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(base_url=OMLX, timeout=120)
    # warmup: loads the model and its codec so we measure warm latency
    synth(client, args.model, "はい。", args.ref, not args.no_stream, args.interval)
    firsts = []
    for run in range(args.runs):
        for i, s in enumerate(SENTENCES):
            res, pcm, sr = synth(client, args.model, s, args.ref, not args.no_stream, args.interval)
            firsts.append(res["first_audio_ms"])
            print(json.dumps({"text": s, **res}, ensure_ascii=False))
            if run == 0:
                short = args.model.split("--")[-1]
                sf.write(out / f"{short}_{args.ref or 'noref'}{args.tag}_{i}.wav", pcm, sr)
    print(f"first audio ms p50={statistics.median(firsts):.0f} min={min(firsts)} max={max(firsts)}")


if __name__ == "__main__":
    main()
