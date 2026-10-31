"""Screen observer: watch the main display, find moments when the screen settles into something new.

Loop (every `interval_s` while あきら is active and the screen is unlocked):
  1. grab the main display, downscale; per-tile grayscale difference to the previous frame
  2. the screen counts as *settled* after `settle_frames` frames with no tile changing
  3. a settled frame is scored against the *reference* (the last frame that fired, i.e. what she saw):
       emb_dist  1 - cosine of SigLIP2 embeddings: did the meaning of the screen change
       tile_max  largest per-tile change: catches a small popup on an otherwise identical screen
       tile_frac share of tiles that changed
       app_switch front app differs from the reference
       app_z     emb_dist relative to this app's usual settled-to-settled change (games/video change a lot)
       seen_sim  similarity to frames fired recently (embedding, among those whose tiles also match):
                 switching back to something she saw is not new
  4. `would_fire` is an initial hand-set rule; the logger records every settled frame so the
     thresholds can be fitted to あきら's own yes/no labels (as MaiBot fits its reply gate).
Denylisted apps are never embedded, stored or scored. Nothing here talks to the LLM or speaks.
"""

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

GRID = (8, 6)  # tile columns x rows


@dataclass
class ObserverConfig:
    interval_s: float = 2.0
    settle_frames: int = 2           # consecutive unchanged frames before a state counts
    stable_tile_change: float = 0.015
    idle_pause_s: float = 300.0      # no input for this long: あきら is away, stop looking
    # initial fire rule (to be fitted from labels)
    fire_emb_dist: float = 0.12
    fire_tile_max: float = 0.20
    popup_tile_frac: float = 0.25    # localized change: few tiles but strong
    min_app_z: float = 1.0           # for apps with a known baseline, change must be unusual
    seen_sim: float = 0.97           # this similar to a recently fired frame: not new
    seen_window_s: float = 1800.0
    denylist: tuple[str, ...] = (
        "com.1password", "com.agilebits", "com.apple.keychainaccess", "com.apple.Passwords",
        "com.bitwarden", "com.lastpass",
    )


@dataclass
class Settled:
    """One settled screen state and its scores against what she last saw."""
    t: float
    app: str
    bundle_id: str
    emb_dist: float
    tile_max: float
    tile_frac: float
    app_switch: bool
    app_z: float | None
    seen_sim: float
    would_fire: bool
    reasons: list[str]
    image: Image.Image | None = field(default=None, repr=False)


def gray_tiles(image: Image.Image) -> np.ndarray:
    """Mean gray level per tile, from a small grayscale copy (cheap, no model)."""
    cols, rows = GRID
    g = np.asarray(image.convert("L").resize((cols * 16, rows * 16)), dtype=np.float32) / 255.0
    return g.reshape(rows, 16, cols, 16)


def tile_change(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-tile mean absolute difference, 0..1."""
    return np.abs(a - b).mean(axis=(1, 3))


class AppBaseline:
    """Running mean/std of settled-to-settled embedding distance per app (Welford)."""

    def __init__(self, min_n: int = 10):
        self.min_n = min_n
        self._s: dict[str, list[float]] = {}

    def z(self, app: str, value: float) -> float | None:
        n, mean, m2 = self._s.get(app, (0, 0.0, 0.0))
        if n < self.min_n:
            return None
        std = math.sqrt(m2 / (n - 1)) or 1e-6
        return (value - mean) / std

    def add(self, app: str, value: float) -> None:
        n, mean, m2 = self._s.get(app, (0, 0.0, 0.0))
        n += 1
        delta = value - mean
        mean += delta / n
        m2 += delta * (value - mean)
        self._s[app] = [n, mean, m2]


class Scorer:
    """Pure scoring state machine: feed frames, get Settled states. No capture, no I/O."""

    def __init__(self, cfg: ObserverConfig, embed):
        self.cfg = cfg
        self.embed = embed
        self.baseline = AppBaseline()
        self._prev_tiles: np.ndarray | None = None
        self._stable = 0
        self._settled_done = False          # already scored the current stable state
        self._last_settled_emb: dict[str, np.ndarray] = {}
        self._ref: tuple[np.ndarray, np.ndarray, str] | None = None  # emb, tiles, bundle_id
        self._seen: deque[tuple[float, np.ndarray, np.ndarray]] = deque()  # t, emb, tiles

    def feed(self, t: float, image: Image.Image, app: str, bundle_id: str) -> Settled | None:
        tiles = gray_tiles(image)
        if self._prev_tiles is not None and tile_change(tiles, self._prev_tiles).max() < self.cfg.stable_tile_change:
            self._stable += 1
        else:
            self._stable = 0
            self._settled_done = False
        self._prev_tiles = tiles
        if self._stable < self.cfg.settle_frames or self._settled_done:
            return None
        self._settled_done = True
        return self._score(t, image, tiles, app, bundle_id)

    def _score(self, t, image, tiles, app, bundle_id) -> Settled:
        cfg = self.cfg
        emb = self.embed(image)
        if self._ref is None:
            emb_dist, tmax, tfrac, switch = 1.0, 1.0, 1.0, True
        else:
            ref_emb, ref_tiles, ref_app = self._ref
            emb_dist = float(1.0 - emb @ ref_emb)
            change = tile_change(tiles, ref_tiles)
            tmax, tfrac = float(change.max()), float((change > 0.05).mean())
            switch = bundle_id != ref_app

        prev = self._last_settled_emb.get(bundle_id)
        step = float(1.0 - emb @ prev) if prev is not None else None
        z = self.baseline.z(bundle_id, step) if step is not None else None
        if step is not None:
            self.baseline.add(bundle_id, step)
        self._last_settled_emb[bundle_id] = emb

        while self._seen and t - self._seen[0][0] > cfg.seen_window_s:
            self._seen.popleft()
        # "seen" needs the meaning AND the layout to match: a popup barely moves the embedding
        seen = max((float(emb @ e) for _, e, tl in self._seen
                    if tile_change(tiles, tl).max() < cfg.fire_tile_max), default=0.0)

        reasons = []
        if switch:
            reasons.append("app_switch")
        if emb_dist >= cfg.fire_emb_dist:
            reasons.append("meaning_changed")
        if tmax >= cfg.fire_tile_max and tfrac <= cfg.popup_tile_frac:
            reasons.append("local_popup")
        fire = bool(reasons) and seen < cfg.seen_sim
        if fire and not switch and z is not None and z < cfg.min_app_z:
            fire = False  # big for most apps, but normal for this one
            reasons.append("usual_for_app")
        if seen >= cfg.seen_sim:
            reasons.append("seen_recently")
        if fire:
            self._ref = (emb, tiles, bundle_id)
            self._seen.append((t, emb, tiles))
        return Settled(t, app, bundle_id, round(emb_dist, 4), round(tmax, 4), round(tfrac, 3), switch,
                       None if z is None else round(z, 2), round(seen, 4), fire, reasons, image)


class ScreenObserver:
    """Capture loop around Scorer. `on_settled` receives every Settled state."""

    def __init__(self, cfg: ObserverConfig, on_settled, on_status=None):
        from . import platform_mac as plat
        from .embed import ImageEmbedder
        self.cfg = cfg
        self.plat = plat
        self.embedder = ImageEmbedder()
        self.scorer = Scorer(cfg, self.embedder.embed)
        self.on_settled = on_settled
        self.on_status = on_status or (lambda s: None)
        self.enabled = True
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def grab_main(self) -> Image.Image:
        import mss
        with mss.MSS() as s:
            shot = s.grab(s.monitors[1])  # monitors[1] = the main display (menu bar)
        return Image.frombytes("RGB", shot.size, shot.rgb)

    def run_forever(self) -> None:
        status = None
        while not self._stop.is_set():
            t0 = time.monotonic()
            fg = self.plat.foreground()
            if not self.enabled:
                state = "paused"
            elif self.plat.screen_locked():
                state = "locked"
            elif self.plat.idle_seconds() > self.cfg.idle_pause_s:
                state = "away"
            elif fg.bundle_id.startswith(self.cfg.denylist):
                state = "denylisted"
            else:
                image = self.grab_main()
                image.thumbnail((640, 640))
                # without the Screen Recording permission macOS returns a flat frame
                if np.asarray(image.convert("L"), dtype=np.float32).std() < 2.0:
                    state = "blank_capture"
                else:
                    state = "watching"
                    settled = self.scorer.feed(time.time(), image, fg.app, fg.bundle_id)
                    if settled:
                        self.on_settled(settled)
            if state != status:
                status = state
                self.on_status(state)
            self._stop.wait(max(0.0, self.cfg.interval_s - (time.monotonic() - t0)))
