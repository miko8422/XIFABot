"""Locate waveform discontinuities and test whether they sit on the codec-token grid.

A discontinuity = a spike in the 2nd difference far above its local level (voiced regions only).
Qwen3-TTS-12Hz emits one codec token per 80 ms = 1920 samples at 24 kHz.
"""
import sys
import numpy as np
import soundfile as sf

TOKEN = 1920
def spikes(a, ratio=6.0):
    d2 = np.abs(np.diff(a, 2))
    win = 240
    local = np.convolve(d2, np.ones(win) / win, mode="same") + 1e-6
    env = np.convolve(np.abs(a), np.ones(480) / 480, mode="same")[1:-1]
    hits = np.flatnonzero((d2 / local > ratio) & (env > 0.02))
    keep = [h for i, h in enumerate(hits) if i == 0 or h - hits[i - 1] > 240]
    return np.array(keep) + 1

groups = {}
for f in sys.argv[1:]:
    a, sr = sf.read(f, dtype="float32")
    start = np.flatnonzero(np.abs(a) > 1e-4)[0]  # audio begins after leading digital zeros
    s = spikes(a)
    groups.setdefault(f.split("/")[-1].rsplit("_", 1)[0], []).append((s, a, start))

for g, takes in groups.items():
    n = sum(len(s) for s, _, _ in takes)
    dur = sum(len(a) for _, a, _ in takes) / 24000
    # phase relative to the token grid, measured from absolute sample 0 and from speech start
    ph0 = np.concatenate([s % TOKEN for s, _, _ in takes]) if n else np.array([])
    on_grid = np.mean((ph0 < 48) | (ph0 > TOKEN - 48)) if n else 0
    print(f"{g:12} spikes={n:3d} ({n / dur:.2f}/s)  within ±2ms of an 80ms token boundary: {on_grid:.0%}  (chance≈5%)")
