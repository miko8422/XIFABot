"""Download isolated model assets, then freeze a Japanese ASR evaluation set."""
import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
SEED = 20261027


def inventory_from_metadata():
    """Record already-resolved revisions even while large downloads are in progress."""
    inventory = {}
    for name, repo, meta in [
        ("qwen17", "mlx-community/Qwen3-ASR-1.7B-8bit", "config.json"),
        ("kotoba", "kaiinui/kotoba-whisper-v2.0-mlx", "config.json"),
        ("reazon", "reazon-research/reazonspeech-k2-v2", "tokens.txt")]:
        path = ROOT / "models" / name
        revision = (path / ".cache/huggingface/download" / (meta + ".metadata")).read_text().splitlines()[0]
        inventory[name] = {"repo": repo, "revision": revision, "path": str(path)}
    base = ROOT / "data/source"
    revision = (base / ".cache/huggingface/download/data/ja_jp/test.tsv.metadata").read_text().splitlines()[0]
    inventory["fleurs"] = {"repo": "google/fleurs", "revision": revision,
        "files": {f: str(base / "data/ja_jp" / f) for f in ["test.tsv", "audio/test.tar.gz"]}}
    cached = Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3-ASR-0.6B-8bit"
    revision = (cached / "refs/main").read_text().strip()
    inventory["qwen06"] = {"repo": "mlx-community/Qwen3-ASR-0.6B-8bit", "revision": revision,
                            "path": str(cached / "snapshots" / revision)}
    (ROOT / "data/assets.json").write_text(json.dumps(inventory, indent=2) + "\n")


def smoke_data():
    """Two existing synthetic utterances for adapter checks; excluded from final clean scores."""
    import soundfile as sf
    inventory_from_metadata()
    rows = []
    for name, text in [("ja1", "今日はちょっと疲れたよ。"), ("ja2", "明日の天気ってどうかな？")]:
        path = PROJECT / "maid_voice/runs/asr_fixtures" / (name + ".wav")
        rows.append(dict(id=name, group="synthetic", audio=str(path), text=text,
                         seconds=sf.info(path).duration, latency_probe=False))
    (ROOT / "data/smoke.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def assets():
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi(token=False)
    jobs = [
        ("qwen17", "mlx-community/Qwen3-ASR-1.7B-8bit", None),
        ("kotoba", "kaiinui/kotoba-whisper-v2.0-mlx", None),
        ("reazon", "reazon-research/reazonspeech-k2-v2", [
            "*.int8.onnx", "decoder-epoch-99-avg-1.onnx", "tokens.txt", "README.md"]),
    ]
    def download(job):
        name, repo, patterns = job
        revision = api.model_info(repo).sha
        print(f"Downloading {name} at {revision}", flush=True)
        path = snapshot_download(repo, revision=revision, local_dir=ROOT / "models" / name,
                                 allow_patterns=patterns, token=False, max_workers=3)
        print(f"Ready: {name}", flush=True)
        return name, {"repo": repo, "revision": revision, "path": str(path)}
    def dataset():
        repo = "google/fleurs"
        revision = api.dataset_info(repo).sha
        files = {}
        for name in ["test.tsv", "audio/test.tar.gz"]:
            files[name] = hf_hub_download(repo, "data/ja_jp/" + name, repo_type="dataset",
                revision=revision, local_dir=ROOT / "data/source", token=False)
        print("Ready: FLEURS Japanese test", flush=True)
        return "fleurs", {"repo": repo, "revision": revision, "files": files}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, j) for j in jobs] + [pool.submit(dataset)]
        inventory = dict(f.result() for f in futures)
    cached = Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3-ASR-0.6B-8bit"
    revision = (cached / "refs/main").read_text().strip()
    inventory["qwen06"] = {"repo": "mlx-community/Qwen3-ASR-0.6B-8bit",
                            "revision": revision, "path": str(cached / "snapshots" / revision)}
    (ROOT / "data/assets.json").write_text(json.dumps(inventory, indent=2) + "\n")


def prepare():
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import math
    inventory = json.loads((ROOT / "data/assets.json").read_text())
    out = ROOT / "data/audio"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    manifest = []
    def add(identifier, group, pcm, sr, text, **meta):
        if pcm.ndim == 2:
            pcm = pcm.mean(axis=1)
        if sr != 16000:
            g = math.gcd(sr, 16000)
            pcm = resample_poly(pcm, 16000 // g, sr // g)
        path = out / (identifier + ".wav")
        sf.write(path, pcm, 16000, subtype="PCM_16")
        row = dict(id=identifier, group=group, audio=str(path.relative_to(ROOT)), text=text,
                   seconds=round(len(pcm) / 16000, 5), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **meta)
        manifest.append(row)
    with open(inventory["fleurs"]["files"]["test.tsv"]) as f:
        rows = list(csv.reader(f, delimiter="\t"))
    # Balanced by gender and duration; unique sentence IDs. No model output affects selection.
    selected, seen = [], set()
    for gender in ["MALE", "FEMALE"]:
        bins = [(6, 8, 10), (8, 12, 5), (12, 20, 15)] if gender == "MALE" else [(6, 8, 5), (8, 12, 10), (12, 20, 15)]
        for lo, hi, count in bins:
            eligible = [r for r in rows if lo <= int(r[5]) / 16000 < hi and r[6] == gender]
            rng.shuffle(eligible)
            take = []
            for r in eligible:
                if r[0] not in seen:
                    seen.add(r[0]); take.append(r)
                    if len(take) == count:
                        break
            assert len(take) == count, (gender, lo, len(take))
            selected.extend(take)
    wanted = {r[1]: r for r in selected}
    with tarfile.open(inventory["fleurs"]["files"]["audio/test.tar.gz"], "r|gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if name not in wanted:
                continue
            r = wanted.pop(name)
            pcm, sr = sf.read(io.BytesIO(tar.extractfile(member).read()), dtype="float32")
            add("fleurs_" + Path(name).stem, "fleurs", pcm, sr, r[2], sentence_id=r[0], gender=r[6])
    assert not wanted, wanted.keys()
    # Reuse only the project's text-cleaning rules, not its destructive data-generation entrypoint.
    import sys
    sys.path.insert(0, str(PROJECT / "maid_voice/scripts"))
    from sft_prep import clean, reject, PUNCT
    candidates = []
    for lab in sorted((PROJECT / "nat").glob("*.lab")):
        text = clean(lab.read_text())
        info = sf.info(lab.with_suffix(".ogg"))
        if 1 <= info.duration <= 8 and reject(text, PUNCT.sub("", text), info.duration) is None:
            candidates.append((lab, text, info.duration))
    for lo, hi in [(1, 4), (4, 8.001)]:
        bucket = [r for r in candidates if lo <= r[2] < hi]
        rng.shuffle(bucket)
        for lab, text, _ in bucket[:20]:
            pcm, sr = sf.read(lab.with_suffix(".ogg"), dtype="float32")
            add(lab.stem, "dialogue", pcm, sr, text, source=str(lab.relative_to(PROJECT)))
    clean_rows = list(manifest)
    # Paired 15 dB white-noise stress test; synthetic stress, not claimed to be real microphone noise.
    for group in ["fleurs", "dialogue"]:
        selected = rng.sample([r for r in clean_rows if r["group"] == group], 10)
        for r in selected:
            pcm, sr = sf.read(ROOT / r["audio"], dtype="float32")
            noise = np.random.default_rng(SEED + len(manifest)).standard_normal(pcm.shape).astype("float32")
            noise *= np.sqrt(np.mean(pcm ** 2)) / (10 ** (15 / 20) * np.sqrt(np.mean(noise ** 2)))
            noisy = pcm + noise
            noisy /= max(1, float(np.max(np.abs(noisy))) / .99)
            add(r["id"] + "_noise15", group + "_noise15", noisy, sr, r["text"], paired_id=r["id"])
    for name, text in [("ja1", "今日はちょっと疲れたよ。"), ("ja2", "明日の天気ってどうかな？")]:
        pcm, sr = sf.read(PROJECT / "maid_voice/runs/asr_fixtures" / (name + ".wav"), dtype="float32")
        add("fixture_" + name, "synthetic", pcm, sr, text)
    add("silence", "control", np.zeros(16000 * 2, dtype="float32"), 16000, "")
    add("quiet_noise", "control", np.random.default_rng(SEED).normal(0, .001, 32000).astype("float32"), 16000, "")
    latency_ids = set()
    for group in ["fleurs", "dialogue"]:
        eligible = [r for r in clean_rows if r["group"] == group and 2 <= r["seconds"] < (8 if group == "fleurs" else 6)]
        latency_ids.update(r["id"] for r in rng.sample(eligible, 15))
    for r in manifest:
        r["latency_probe"] = r["id"] in latency_ids
    rng.shuffle(manifest)
    (ROOT / "data/manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest))
    summary = {"seed": SEED, "clips": len(manifest), "seconds": sum(r["seconds"] for r in manifest),
        "groups": {g: sum(r["group"] == g for r in manifest) for g in sorted({r['group'] for r in manifest})},
        "latency_probe_ids": sorted(latency_ids), "fleurs_revision": inventory['fleurs']['revision']}
    (ROOT / "data/selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["assets", "data", "smoke"])
    args = ap.parse_args()
    {"assets": assets, "data": prepare, "smoke": smoke_data}[args.command]()
