"""Recover large public HF downloads with resumable ranges and SHA-256 verification."""
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
JOBS = [
    ("dataset", "google/fleurs", "data/ja_jp/audio/test.tar.gz", ROOT / "data/source/data/ja_jp/audio/test.tar.gz"),
    ("model", "mlx-community/Qwen3-ASR-1.7B-8bit", "model.safetensors", ROOT / "models/qwen17/model.safetensors"),
    ("model", "kaiinui/kotoba-whisper-v2.0-mlx", "weights.npz", ROOT / "models/kotoba/weights.npz"),
]


def fetch(job):
    kind, repo, filename, dest = job
    assets = json.loads((ROOT / "data/assets.json").read_text())
    revision = next(v["revision"] for v in assets.values() if v["repo"] == repo)
    client = httpx.Client(timeout=httpx.Timeout(120, connect=30), follow_redirects=True,
                         limits=httpx.Limits(max_connections=8, max_keepalive_connections=8))
    api_kind = "datasets" if kind == "dataset" else "models"
    parent = str(Path(filename).parent)
    url = f"https://huggingface.co/api/{api_kind}/{repo}/tree/{revision}" + ("/" + parent if parent != "." else "")
    response = client.get(url); response.raise_for_status()
    meta = next(f for f in response.json() if f["path"] == filename)
    size, digest = meta["size"], meta["lfs"]["oid"]
    def valid(path):
        if not path.exists() or path.stat().st_size != size:
            return False
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest() == digest
    if valid(dest):
        print(f"Already verified: {dest.name}", flush=True); return
    resolve = f"https://huggingface.co/" + ("datasets/" if kind == "dataset" else "") + f"{repo}/resolve/{revision}/{filename}"
    redirect = client.head(resolve, follow_redirects=False)
    if redirect.status_code >= 400:
        redirect.raise_for_status()
    direct = redirect.headers.get("location", resolve)
    parts = dest.parent / ("." + dest.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    chunk = 8 * 1024 * 1024
    ranges = [(i, start, min(start + chunk, size) - 1) for i, start in enumerate(range(0, size, chunk))]
    def part(item):
        i, start, end = item
        p = parts / f"{i:05d}"
        if p.exists() and p.stat().st_size == end - start + 1:
            return
        for attempt in range(4):
            try:
                with client.stream("GET", direct, headers={"Range":f"bytes={start}-{end}"}) as r:
                    r.raise_for_status()
                    expected = f"bytes {start}-{end}/{size}"
                    if r.status_code != 206 or r.headers.get("content-range") != expected:
                        raise RuntimeError(f"Unexpected range response: {r.status_code}, {r.headers.get('content-range')}")
                    tmp = p.with_suffix(".partial")
                    with tmp.open("wb") as f:
                        for data in r.iter_bytes(1024 * 1024):
                            f.write(data)
                if tmp.stat().st_size != end - start + 1:
                    raise RuntimeError("Incomplete range")
                tmp.replace(p)
                return
            except Exception as exc:
                if attempt == 3:
                    raise RuntimeError(f"{dest.name} part {i}: {type(exc).__name__}") from exc
                time.sleep(attempt + 1)
    print(f"Recovering {dest.name}: {size / 2**20:.0f} MiB in {len(ranges)} ranges", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for n, future in enumerate(concurrent.futures.as_completed([pool.submit(part, r) for r in ranges]), 1):
            future.result()
            if n % 10 == 0 or n == len(ranges):
                print(f"{dest.name}: {n}/{len(ranges)} ranges", flush=True)
    assembled = dest.with_suffix(dest.suffix + ".verified-partial")
    with assembled.open("wb") as out:
        for i, _, _ in ranges:
            with (parts / f"{i:05d}").open("rb") as src:
                while data := src.read(8 * 1024 * 1024):
                    out.write(data)
    if not valid(assembled):
        raise RuntimeError(f"SHA-256 mismatch for {dest.name}")
    assembled.replace(dest)
    print(f"Verified {dest.name}: {digest}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", choices=["all", "fleurs", "qwen17", "kotoba"], default="all")
    args = ap.parse_args()
    # Finish the evaluation audio first so existing models can be tested while weights download.
    jobs = JOBS if args.job == "all" else [JOBS[["fleurs", "qwen17", "kotoba"].index(args.job)]]
    for job in jobs:
        fetch(job)
