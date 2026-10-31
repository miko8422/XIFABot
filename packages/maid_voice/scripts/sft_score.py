"""Score fine-tune eval wavs (from sft/eval_ckpt.py) per model and temperature.

CER: Qwen3-ASR round trip via oMLX (kana readings). spk: WavLM-SV cosine to real ナツメ lines.
No automatic glitch count: energy-dropout and high-frequency-spike detectors could not tell the
known "dropped mic" takes from clean ones (fricatives look the same), so bursts are judged by ear.

  uv run scripts/sft_score.py runs/sft/evalwav
"""

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import httpx
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from tts_eval import OMLX, SENTENCES, SpeakerScorer, cer, transcribe  # noqa: E402


def main() -> None:
    root = Path(sys.argv[1])
    texts = SENTENCES  # eval_ckpt.py renders the same list (sft_prep copies it into the data dir)
    client = httpx.Client(base_url=OMLX, timeout=300)
    spk = SpeakerScorer(exclude=set())
    rows = []
    for wav in sorted(root.glob("*/T*_s*_k*.wav")):
        m = re.match(r"T([\d.]+)_s(\d+)_k(\d+)", wav.stem)
        temp, i, k = float(m[1]), int(m[2]), int(m[3])
        hyp = transcribe(client, wav)
        rows.append({"model": wav.parent.name, "temp": temp, "s": i, "k": k, "text": texts[i], "asr": hyp,
                     "cer": round(cer(texts[i], hyp), 3), "spk": round(spk.similarity(wav), 3),
                     "dur": round(sf.info(wav).duration, 2)})
    (root / "scores.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["temp"])].append(r)
    print(f"real-line speaker baseline {spk.real_baseline:.3f}")
    print(f"{'model':9} {'T':>4} {'CER mean':>8} {'bad>0.1':>7} {'spk':>6} {'dur':>5}")
    for (model, temp), g in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):
        print(f"{model:9} {temp:4} {statistics.mean(r['cer'] for r in g):8.3f} {sum(r['cer'] > 0.1 for r in g):7d} "
              f"{statistics.mean(r['spk'] for r in g):6.3f} "
              f"{statistics.mean(r['dur'] for r in g):5.2f}")


if __name__ == "__main__":
    main()
