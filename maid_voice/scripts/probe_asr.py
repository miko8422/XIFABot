"""Phase 0 probe: oMLX transcription latency on short utterances (push-to-talk style upload).

Usage: uv run scripts/probe_asr.py --model M [--runs 3] [--stream]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "runs" / "asr_fixtures"


def transcribe(client, model, path, stream):
    data = {"model": model}
    if stream:
        data["stream"] = "true"
    t0 = time.monotonic()
    t_first = None
    text = ""
    with client.stream("POST", "/v1/audio/transcriptions", data=data,
                       files={"file": (path.name, path.read_bytes(), "audio/wav")}) as r:
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code}: {r.read()[:300]!r}")
        if stream:
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                ev = json.loads(line[5:])
                if ev.get("type") == "transcript.text.delta":
                    t_first = t_first or time.monotonic()
                    text += ev.get("delta", "")
                elif ev.get("type") == "transcript.text.done":
                    text = ev.get("text", text)
        else:
            text = json.loads(r.read())["text"]
    t_end = time.monotonic()
    return {
        "file": path.name,
        "audio_s": round(sf.info(path).duration, 2),
        "first_ms": round(((t_first or t_end) - t0) * 1000),
        "total_ms": round((t_end - t0) * 1000),
        "text": text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--stream", action="store_true")
    args = ap.parse_args()
    client = httpx.Client(base_url="http://127.0.0.1:9800", timeout=120)
    files = sorted(FIXTURES.glob("*.wav"))
    transcribe(client, args.model, files[0], args.stream)  # warmup / model load
    totals = []
    for run in range(args.runs):
        for f in files:
            res = transcribe(client, args.model, f, args.stream)
            totals.append(res["total_ms"])
            if run == 0:
                print(json.dumps(res, ensure_ascii=False))
    print(f"total ms p50={statistics.median(totals):.0f} min={min(totals)} max={max(totals)} n={len(totals)}")


if __name__ == "__main__":
    main()
