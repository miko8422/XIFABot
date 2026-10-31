"""Isolate the Qwen3-TTS codec decoder: identical tokens decoded whole vs in streaming steps.

Encodes real recordings to codec tokens, then decodes them (a) in one pass and (b) with
decoder.streaming_step on chunks of N tokens, the way oMLX streams. Any difference = decoder seams.
Also reports the codec's own round-trip error, the floor that no TTS setting can beat.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model

sys.path.insert(0, str(Path(__file__).parent))
from build_reference import resample_half  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOKEN = 1920
REPO = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
CHUNK = int(sys.argv[1]) if len(sys.argv) > 1 else 2
FILES = sys.argv[2:] or [str(ROOT.parent / "nat" / f) for f in ("nat003_031.ogg", "nat002_057.ogg")]

model = load_model(REPO)
st = model.speech_tokenizer
dec = st.decoder
out = ROOT / "runs" / "artifact"
out.mkdir(parents=True, exist_ok=True)

for f in FILES:
    raw, sr = sf.read(f, dtype="float32")
    audio = mx.array(resample_half(raw) if sr == 48000 else raw)
    codes = st.encode(audio[None, None, :])  # [1, Q, T]
    mx.eval(codes)
    T = codes.shape[-1]
    whole = np.array(dec.chunked_decode(codes)).flatten()

    dec.reset_streaming_state()
    parts = []
    for i in range(0, T, CHUNK):
        parts.append(np.array(dec.streaming_step(codes[:, :, i:i + CHUNK])).flatten())
    streamed = np.concatenate(parts)

    n = min(len(whole), len(streamed))
    d = np.abs(whole[:n] - streamed[:n])
    phase = np.arange(n) % (TOKEN * CHUNK)
    near = (phase < 96) | (phase > TOKEN * CHUNK - 96)
    stem = Path(f).stem
    sf.write(out / f"{stem}_codec_whole.wav", whole, 24000)
    sf.write(out / f"{stem}_codec_stream{CHUNK}.wav", streamed, 24000)
    n = min(n, audio.shape[0]); a = np.array(audio)[:n]
    print(f"{stem}: tokens={T} len whole={len(whole)} streamed={len(streamed)}")
    print(f"   whole vs streamed: max|d|={d.max():.4f} rms(d)={np.sqrt(np.mean(d**2)):.5f} "
          f"signal rms={np.sqrt(np.mean(whole**2)):.4f}; diff energy near chunk seams "
          f"{(d[near]**2).sum() / max((d**2).sum(), 1e-12):.0%} (uniform {near.mean():.0%})")
    print(f"   codec round trip (original vs whole decode): rms err={np.sqrt(np.mean((a - whole[:n])**2)):.4f}")
