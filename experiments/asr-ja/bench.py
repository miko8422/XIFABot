"""Run one recognizer per process, sequentially; no changes to the live app."""
import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import random
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache/numba"))
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / ".cache/tiktoken"))

import numpy as np
import soundfile as sf

CHOICES = ["omlx_auto", "omlx_ja", "qwen06", "qwen17", "reazon", "kotoba"]


def build(name, assets, threads):
    sync = lambda: None
    config = {"model": name}
    if name.startswith("omlx"):
        import httpx
        client = httpx.Client(base_url="http://127.0.0.1:9800", timeout=45)
        config.update(endpoint=str(client.base_url), language="" if name == "omlx_auto" else "ja",
                      model="mlx-community--Qwen3-ASR-0.6B-8bit")
        def transcribe(pcm):
            buf = io.BytesIO()
            sf.write(buf, pcm, 16000, format="WAV", subtype="PCM_16")
            form = {"model": config["model"], "response_format": "verbose_json"}
            if config["language"]:
                form["language"] = config["language"]
            response = client.post("/v1/audio/transcriptions", data=form,
                                   files={"file": ("clip.wav", buf.getvalue(), "audio/wav")})
            response.raise_for_status()
            data = response.json()
            return data.get("text", "").strip(), {"language": data.get("language")}
    elif name.startswith("qwen"):
        import mlx.core as mx
        from mlx_audio.stt.utils import load_model
        config.update(assets[name], language="Japanese", max_tokens=256, temperature=0)
        model = load_model(Path(assets[name]["path"]))
        sync = mx.synchronize
        def transcribe(pcm):
            out = model.generate(pcm, language="Japanese", max_tokens=256, temperature=0, verbose=False)
            return out.text.strip(), {"generation_tokens": getattr(out, "generation_tokens", None)}
    elif name == "reazon":
        import sherpa_onnx
        path = Path(assets[name]["path"])
        config.update(assets[name], precision="int8-fp32", threads=threads, padding_seconds_each_side=.9,
                      decoding="greedy_search", device="cpu")
        model = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=str(path / "tokens.txt"), encoder=str(path / "encoder-epoch-99-avg-1.int8.onnx"),
            decoder=str(path / "decoder-epoch-99-avg-1.onnx"),
            joiner=str(path / "joiner-epoch-99-avg-1.int8.onnx"),
            num_threads=threads, sample_rate=16000, feature_dim=80, decoding_method="greedy_search", provider="cpu")
        def transcribe(pcm):
            stream = model.create_stream()
            stream.accept_waveform(16000, np.pad(pcm, 14400))  # official ReazonSpeech preprocessing
            model.decode_stream(stream)
            return stream.result.text.strip(), {}
    else:
        import mlx.core as mx
        import mlx_whisper
        config.update(assets[name], language="ja", temperature=0, condition_on_previous_text=False)
        sync = mx.synchronize
        def transcribe(pcm):
            out = mlx_whisper.transcribe(pcm, path_or_hf_repo=assets[name]["path"], language="ja",
                task="transcribe", temperature=0, condition_on_previous_text=False, verbose=None)
            return out["text"].strip(), {"segments": len(out.get("segments", []))}
    return transcribe, sync, config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=CHOICES)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.jsonl")
    args = ap.parse_args()
    rows = [json.loads(s) for s in args.manifest.read_text().splitlines()]
    if args.limit:
        rows = [r for r in rows if r["group"] != "control"][:args.limit]
    assets = json.loads((ROOT / "data/assets.json").read_text())
    output = ROOT / "runs" / args.tag
    output.mkdir(parents=True, exist_ok=True)
    dest = output / (args.model + ".jsonl")
    if dest.exists():
        raise SystemExit(f"Refusing to overwrite {dest}; choose a different --tag")
    start = time.perf_counter()
    transcribe, sync, config = build(args.model, assets, args.threads)
    load_ms = (time.perf_counter() - start) * 1000
    speech = [r for r in rows if r["group"] in ["fleurs", "dialogue"]]
    warmups = []
    for row in (speech[:3] or rows[:3]):
        pcm, _ = sf.read(ROOT / row["audio"], dtype="float32")
        sync(); t0 = time.perf_counter()
        text, _ = transcribe(pcm)
        sync()
        warmups.append({"id": row["id"], "ms": (time.perf_counter() - t0) * 1000})
    schedule = [(r, 0) for r in rows]
    # Accuracy uses one frozen pass; latency uses 30 short clips, each repeated three times.
    for repeat in range(1, args.repeats):
        extra = [r for r in rows if r["latency_probe"]]
        random.Random(20261027 + repeat).shuffle(extra)
        schedule.extend((r, repeat) for r in extra)
    metadata = {"model": args.model, "config": config, "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "chip": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "load_ms": load_ms, "warmups": warmups, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "versions": {k: importlib.metadata.version(k) for k in ["mlx", "mlx-audio", "mlx-whisper", "sherpa-onnx", "numpy", "soundfile"]}}
    (output / (args.model + ".meta.json")).write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{args.model}: loaded in {load_ms:.0f} ms; running {len(schedule)} calls", flush=True)
    with dest.open("w") as f:
        for i, (row, repeat) in enumerate(schedule):
            pcm, _ = sf.read(ROOT / row["audio"], dtype="float32")
            sync(); t0 = time.perf_counter()
            error = None
            try:
                text, detail = transcribe(pcm)
                sync()
            except Exception as exc:
                text, detail, error = "", {}, f"{type(exc).__name__}: {exc}"
            ms = (time.perf_counter() - t0) * 1000
            rec = {**row, "model": args.model, "repeat": repeat, "hypothesis": text,
                   "latency_ms": round(ms, 3), "error": error, **detail}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if i % 10 == 0 or error or i == len(schedule) - 1:
                print(f"{i+1}/{len(schedule)} {row['id']} {ms:.0f} ms" + (f" ERROR {error}" if error else ""), flush=True)
    metadata.update(finished_utc=datetime.now(timezone.utc).isoformat(), peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    (output / (args.model + ".meta.json")).write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
