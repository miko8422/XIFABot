"""TTS quality vs latency experiment.

  uv run --group eval scripts/tts_eval.py MODEL:REF[:INTERVAL] [...] [--name NAME]

INTERVAL is oMLX streaming_interval in seconds (default 0.2); "ns" = non-streaming request.
For each config: warm first-audio latency, real-time factor, ASR round-trip CER
(Qwen3-ASR via oMLX) and WavLM speaker similarity to held-out real ナツメ lines.
Writes runs/tts_eval/<name>/results.json, a summary table, and a blind listening pack.
"""

import argparse
import base64
import json
import random
import re
import statistics
import time
import unicodedata
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
NAT = ROOT.parent / "nat"
OMLX = "http://127.0.0.1:9800"
ASR_MODEL = "mlx-community--Qwen3-ASR-0.6B-8bit"
ALIASES = {
    "q06": "mlx-community--Qwen3-TTS-12Hz-0.6B-Base-8bit",
    "q17": "mlx-community--Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "q17bf": "mlx-community--Qwen3-TTS-12Hz-1.7B-Base-bf16",
    "vox": "mlx-community--VoxCPM2-8bit",
    "fish": "mlx-community--fish-audio-s2-pro-8bit",
    "iro": "mlx-community--Irodori-TTS-v4.1-Small-fp16",
}
# assistant-style lines with the usual trouble spots: loanwords, English, numbers, っ/ん/ー
SENTENCES = [
    "お疲れさま。今日はもう休んだら？",
    "バックグラウンドのタスクは、まだ終わってないみたい。",
    "ダウンロードは九十八パーセントまで進んでるわよ。",
    "ちょっと待って、今確認してみるから。",
    "Windowsのアップデートが来てるけど、あとでいいわよね。",
    "ふーん、コーヒーより紅茶のほうがいいんじゃない？",
    "えっと、三時半にミーティングがあるはずよ。",
    "別に心配してたわけじゃないんだからね。",
    "Steamのセールは明日の午前二時までだって。",
    "ゲームばっかりしてないで、ちゃんとご飯食べなさいよ。",
    "了解。じゃあ、十分後にもう一度声をかけるわ。",
    "メールが三件届いてる。急ぎのものはなさそう。",
    "ん、どうしたの？何か言いたいことでもあるの？",
    "エラーが出てるわね。ログを見てみる？",
]


def resolve(model: str) -> str:
    return ALIASES.get(model, model)


_KAKASI = None


def norm(text: str) -> str:
    """Compare readings, not spellings: 様/さま and 後で/あとで must not count as errors."""
    global _KAKASI
    if _KAKASI is None:
        import pykakasi
        _KAKASI = pykakasi.kakasi()
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(t["hira"] for t in _KAKASI.convert(text))
    return re.sub(r"[\s、。，．,.!?！？「」『』…ー〜~・]", "", text)


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def synth(client: httpx.Client, model: str, text: str, ref: str, interval: str, temperature=None):
    body = {"model": model, "input": text, "language": "japanese", "response_format": "wav",
            "ref_audio": base64.b64encode((ROOT / f"voices/natsume/ref_{ref}.wav").read_bytes()).decode(),
            "ref_text": (ROOT / f"voices/natsume/ref_{ref}.txt").read_text(encoding="utf-8")}
    if temperature is not None:
        body["temperature"] = temperature
    if interval != "ns":
        body["stream"] = True
        body["streaming_interval"] = float(interval)
    t0 = time.monotonic()
    t_first = None
    buf = bytearray()
    with client.stream("POST", "/v1/audio/speech", json=body) as r:
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code}: {r.read()[:300]!r}")
        for chunk in r.iter_bytes():
            buf += chunk
            if t_first is None and len(buf) > 44:
                t_first = time.monotonic()
    t_end = time.monotonic()
    sr = int.from_bytes(buf[24:28], "little")
    pcm = np.frombuffer(bytes(buf[44:]), dtype=np.int16).astype("float32") / 32768
    return pcm, sr, (t_first - t0) * 1000, (t_end - t0) * 1000


def transcribe(client: httpx.Client, path: Path) -> str:
    r = client.post("/v1/audio/transcriptions", data={"model": ASR_MODEL, "language": "japanese"},
                    files={"file": (path.name, path.read_bytes(), "audio/wav")})
    r.raise_for_status()
    return r.json()["text"]


class SpeakerScorer:
    def __init__(self, exclude: set[str], n_targets: int = 24):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.torch = torch
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
        # held-out real lines (story scenes only, 2-8 s, not used as references)
        pool = [p for p in sorted(NAT.glob("nat0*.ogg")) if p.stem not in exclude
                and 2.0 <= sf.info(p).duration <= 8.0]
        random.Random(0).shuffle(pool)
        embs = [self.embed(*sf.read(p, dtype="float32")) for p in pool[:n_targets]]
        self.centroid = self._unit(np.mean(embs, axis=0))
        # baseline: how similar real lines are to the centroid of other real lines
        self.real_baseline = statistics.mean(
            float(self._unit(self.embed(*sf.read(p, dtype="float32"))) @ self.centroid)
            for p in pool[n_targets:n_targets + 12])

    @staticmethod
    def _unit(v):
        return v / np.linalg.norm(v)

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if audio.ndim > 1:
            audio = audio.mean(1)
        if sr != 16000:
            idx = np.arange(0, len(audio), sr / 16000)
            audio = np.interp(idx, np.arange(len(audio)), audio).astype("float32")
        inputs = self.fe(audio, sampling_rate=16000, return_tensors="pt")
        with self.torch.no_grad():
            return self.model(**inputs).embeddings[0].numpy()

    def similarity(self, path: Path) -> float:
        return float(self._unit(self.embed(*sf.read(path, dtype="float32"))) @ self.centroid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", help="MODEL:REF[:INTERVAL], MODEL may be an alias")
    ap.add_argument("--name", default=time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--no-speaker", action="store_true")
    ap.add_argument("--rescore", action="store_true", help="only add speaker scores to an existing run")
    args = ap.parse_args()

    out = ROOT / "runs" / "tts_eval" / args.name
    (out / "wav").mkdir(parents=True, exist_ok=True)
    client = httpx.Client(base_url=OMLX, timeout=300)
    results = []
    if args.rescore:
        results = json.loads((out / "results.json").read_text())["results"]
        args.configs = []
        for res in results:
            for row in res.get("rows", []):
                if not row.get("asr"):  # rows from tts_direct.py have no transcript yet
                    row["asr"] = transcribe(client, Path(row["path"]))
                row["cer"] = round(cer(row["text"], row["asr"]), 3)
    for spec in args.configs:
        model_alias, ref, *rest = spec.split(":")
        interval = rest[0] if rest else "0.2"
        temp = float(rest[1]) if len(rest) > 1 else None
        model = resolve(model_alias)
        label = f"{model_alias}:{ref}:{interval}" + (f":T{temp}" if temp is not None else "")
        print(f"== {label}", flush=True)
        try:
            synth(client, model, "はい、わかったわ。", ref, interval, temp)  # load + warm
        except Exception as e:
            print(f"   FAILED: {e}")
            results.append({"config": label, "error": str(e)})
            continue
        rows = []
        for i, text in enumerate(SENTENCES):
            pcm, sr, first, total = synth(client, model, text, ref, interval, temp)
            path = out / "wav" / f"{label.replace(':', '_')}_{i:02d}.wav"
            sf.write(path, pcm, sr)
            hyp = transcribe(client, path)
            rows.append({"i": i, "text": text, "asr": hyp, "cer": round(cer(text, hyp), 3),
                         "first_ms": round(first), "total_ms": round(total),
                         "audio_s": round(len(pcm) / sr, 2), "path": str(path)})
            print(f"   {i:02d} first={first:5.0f}ms total={total:5.0f}ms cer={rows[-1]['cer']:.2f}  {hyp}", flush=True)
        results.append({"config": label, "model": model, "ref": ref, "interval": interval, "rows": rows})

    if not args.no_speaker:
        print("scoring speaker similarity ...", flush=True)
        scorer = SpeakerScorer(exclude={"nat013_034", "nat110_014", "nat013_007", "nat107_006", "nat013_086"})
        print(f"   real-vs-real baseline similarity: {scorer.real_baseline:.3f}")
        for res in results:
            for row in res.get("rows", []):
                row["spk"] = round(scorer.similarity(Path(row["path"])), 3)

    summary = []
    for res in results:
        if "rows" not in res:
            summary.append({"config": res["config"], "error": res["error"]})
            continue
        rows = res["rows"]
        audio = sum(r["audio_s"] for r in rows)
        summary.append({
            "config": res["config"],
            "first_p50": round(statistics.median(r["first_ms"] for r in rows)),
            "first_max": max(r["first_ms"] for r in rows),
            "rtf": round(sum(r["total_ms"] for r in rows) / 1000 / audio, 2),
            "cer_mean": round(statistics.mean(r["cer"] for r in rows), 3),
            "spk_mean": round(statistics.mean(r["spk"] for r in rows), 3) if "spk" in rows[0] else None,
        })
    (out / "results.json").write_text(json.dumps({"summary": summary, "results": results},
                                                 ensure_ascii=False, indent=1))
    print(f"\n{'config':28} {'first p50':>9} {'max':>6} {'RTF':>5} {'CER':>6} {'spk':>6}")
    for s in summary:
        if "error" in s:
            print(f"{s['config']:28} ERROR {s['error'][:60]}")
        else:
            print(f"{s['config']:28} {s['first_p50']:>9} {s['first_max']:>6} {s['rtf']:>5} "
                  f"{s['cer_mean']:>6} {s['spk_mean'] if s['spk_mean'] is not None else '-':>6}")
    print(f"\n{out}")


if __name__ == "__main__":
    main()
