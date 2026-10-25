"""Count click-like discontinuities: sample jumps far above the local high-frequency level."""
import sys
import numpy as np
import soundfile as sf

def clicks(path, ratio=8.0, win=480):
    a, sr = sf.read(path, dtype="float32")
    d = np.abs(np.diff(a))
    # local median of |diff| over ~20 ms
    pad = np.pad(d, win // 2, mode="edge")
    local = np.array([np.median(pad[i:i + win]) for i in range(0, len(d), 48)])
    local = np.repeat(local, 48)[: len(d)] + 1e-4
    hits = np.flatnonzero((d / local > ratio) & (d > 0.05))
    # merge hits within 5 ms
    events = [h for i, h in enumerate(hits) if i == 0 or h - hits[i - 1] > sr // 200]
    return [round(h / sr, 3) for h in events], len(a) / sr

for p in sys.argv[1:]:
    ev, dur = clicks(p)
    print(f"{p.split('/')[-1]:45} dur={dur:.2f}s clicks={len(ev):2d} at {ev[:8]}")
