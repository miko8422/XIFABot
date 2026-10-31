"""Prove/disprove that streaming decode adds seam artifacts.

Same seed -> same codec tokens; decode once whole (stream=False) and once incrementally (stream=True).
Any waveform difference then comes from the streaming decoder alone.
Run with whichever mlx-audio you want to test (oMLX's bundled one or a newer PyPI release).
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
REPO = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
TEXT = "そういう生き方もあるのね。"
TOKEN = 1920

from mlx_audio.tts.utils import load_model  # noqa: E402

model = load_model(REPO)
ref = str(ROOT / "voices/natsume/ref_a.wav")
ref_text = (ROOT / "voices/natsume/ref_a.txt").read_text(encoding="utf-8")


def run(stream: bool) -> np.ndarray:
    mx.random.seed(1234)
    kw = dict(text=TEXT, ref_audio=ref, ref_text=ref_text, lang_code="japanese")
    if stream:
        kw.update(stream=True, streaming_interval=0.2)
    parts = [np.array(r.audio, dtype=np.float32).flatten() for r in model.generate(**kw)]
    return np.concatenate(parts)


whole = run(False)
streamed = run(True)
out = ROOT / "runs" / "artifact"
out.mkdir(parents=True, exist_ok=True)
sf.write(out / "proof_whole.wav", whole, 24000)
sf.write(out / "proof_streamed.wav", streamed, 24000)
n = min(len(whole), len(streamed))
print(f"lengths whole={len(whole)} streamed={len(streamed)}")
diff = np.abs(whole[:n] - streamed[:n])
print(f"max |diff| = {diff.max():.4f}, mean = {diff.mean():.5f}, signal rms = {np.sqrt(np.mean(whole ** 2)):.4f}")
# where is the difference concentrated? energy of diff vs phase within the 80 ms token period
phase = np.arange(n) % TOKEN
near = (phase < 96) | (phase > TOKEN - 96)  # within 4 ms of a token boundary (10% of the period)
print(f"share of diff energy within ±4ms of token boundaries: {(diff[near] ** 2).sum() / (diff ** 2).sum():.0%} "
      f"(uniform would be {near.mean():.0%})")
for k in range(0, n // TOKEN):
    seg = diff[k * TOKEN:(k + 1) * TOKEN]
    if seg.max() > 0.02:
        print(f"  token {k:3d} ({k * 80:5d} ms): max diff {seg.max():.3f} at +{seg.argmax() / 24:.1f} ms")
