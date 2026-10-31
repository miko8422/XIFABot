"""Synthesize the tts_eval sentence set in-process with a newer mlx-audio than oMLX bundles.

  uv run --isolated --no-project --python 3.12 --with mlx-audio --with httpx \
      scripts/tts_direct.py REPO REF --name NAME [--label LABEL]

Writes runs/tts_eval/NAME/results.json in tts_eval's format (without ASR/speaker scores);
then `uv run --group eval scripts/tts_eval.py x --name NAME --rescore` fills those in.
Uses streaming generation when the model supports it, so first_ms is time to first audio.
"""

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from tts_eval import SENTENCES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("ref")
    ap.add_argument("--name", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--interval", type=float, default=0.2)
    args = ap.parse_args()

    from mlx_audio.tts.utils import load_model
    from mlx_audio.utils import load_audio

    label = args.label or f"{args.repo.split('/')[-1]}:{args.ref}"
    out = ROOT / "runs" / "tts_eval" / args.name
    (out / "wav").mkdir(parents=True, exist_ok=True)

    if "fish" in args.repo.lower():
        # mlx-community's 8-bit Fish S2 Pro checkpoint quantized embeddings and fast_* layers too,
        # but mlx-audio 0.6.1's predicate skips them -> "parameters not in model". Match the checkpoint.
        from mlx_audio.tts.models.fish_qwen3_omni import fish_speech
        fish_speech.Model.model_quant_predicate = classmethod(
            lambda cls, path, module: hasattr(module, "to_quantized"))

    t0 = time.monotonic()
    model = load_model(args.repo)
    print(f"loaded {args.repo} in {time.monotonic() - t0:.1f}s", flush=True)
    params = inspect.signature(model.generate).parameters
    sr = model.sample_rate
    ref_path = ROOT / f"voices/natsume/ref_{args.ref}.wav"
    ref_text = (ROOT / f"voices/natsume/ref_{args.ref}.txt").read_text(encoding="utf-8")
    ref_audio = load_audio(str(ref_path), sample_rate=sr)

    kwargs = {"ref_audio": ref_audio}
    if "ref_text" in params:
        kwargs["ref_text"] = ref_text
    streaming = "stream" in params and "irodori" not in args.repo.lower()  # declared but NotImplemented
    if streaming:
        kwargs["stream"] = True
        if "streaming_interval" in params:
            kwargs["streaming_interval"] = args.interval
    if "lang_code" in params:
        kwargs["lang_code"] = "ja"
    print(f"sample_rate={sr} streaming={streaming} kwargs={sorted(k for k in kwargs if k != 'ref_audio')}")

    def synth(text):
        t0 = time.monotonic()
        t_first = None
        parts = []
        for res in model.generate(text=text, **kwargs):
            audio = np.array(res.audio, dtype=np.float32).flatten()
            if t_first is None and audio.size:
                t_first = time.monotonic()
            parts.append(audio)
        t_end = time.monotonic()
        return np.concatenate(parts) if parts else np.zeros(0, "float32"), (t_first - t0) * 1000, (t_end - t0) * 1000

    synth("はい、わかったわ。")  # warmup
    rows = []
    for i, text in enumerate(SENTENCES):
        pcm, first, total = synth(text)
        path = out / "wav" / f"{label.replace(':', '_').replace('/', '_')}_{i:02d}.wav"
        sf.write(path, pcm, sr)
        rows.append({"i": i, "text": text, "asr": None, "cer": None, "first_ms": round(first),
                     "total_ms": round(total), "audio_s": round(len(pcm) / sr, 2), "path": str(path)})
        print(f"   {i:02d} first={first:6.0f}ms total={total:6.0f}ms audio={len(pcm) / sr:.2f}s", flush=True)

    res_file = out / "results.json"
    existing = json.loads(res_file.read_text())["results"] if res_file.exists() else []
    existing = [r for r in existing if r["config"] != label]
    existing.append({"config": label, "model": args.repo, "ref": args.ref,
                     "interval": str(args.interval) if streaming else "ns", "rows": rows})
    res_file.write_text(json.dumps({"summary": [], "results": existing}, ensure_ascii=False, indent=1))
    print(res_file)


if __name__ == "__main__":
    main()
