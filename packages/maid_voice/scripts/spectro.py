"""Grid of spectrograms (first SECONDS of each file) + per-file low-frequency / burst stats."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

SECONDS = float(sys.argv[1]); out = sys.argv[2]; files = sys.argv[3:]
cols = 6; rows = (len(files) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.4), squeeze=False)
for ax, f in zip(axes.flat, files):
    a, sr = sf.read(f, dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    a = a[: int(sr * SECONDS)]
    ax.specgram(a, NFFT=1024, Fs=sr, noverlap=768, cmap="magma", vmin=-120, vmax=-20)
    ax.set_ylim(0, 12000); ax.set_title(f.split("/")[-1][:28], fontsize=7); ax.tick_params(labelsize=6)
    # stats on 10 ms frames: share of energy below 100 Hz, and max frame-to-frame level jump
    hop = sr // 100; frames = a[: len(a) // hop * hop].reshape(-1, hop)
    spec = np.abs(np.fft.rfft(frames * np.hanning(hop), axis=1)) ** 2
    freqs = np.fft.rfftfreq(hop, 1 / sr)
    low = spec[:, freqs < 100].sum(1) / (spec.sum(1) + 1e-12)
    lvl = 10 * np.log10(spec.sum(1) + 1e-12)
    print(f"{f.split('/')[-1]:32} max_low<100Hz={low.max():.2f} at {low.argmax() * 10}ms  max_jump={np.diff(lvl).max():.1f}dB")
for ax in list(axes.flat)[len(files):]: ax.axis("off")
plt.tight_layout(); plt.savefig(out, dpi=90)
