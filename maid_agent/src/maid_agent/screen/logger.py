"""Calibration logger: record every settled screen state with its scores, plus a small thumbnail.

Nothing is sent anywhere. Files: runs/screen/<date>/settled.jsonl, thumbs/<time>.jpg (320 px),
status.jsonl (watching/away/locked/denylisted, and an hourly CPU heartbeat). Days older than
`keep_days` are deleted. The thumbnails are only for あきら to label "would I want her to see this?".
"""

import json
import resource
import shutil
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from .observer import ObserverConfig, ScreenObserver, Settled


class CalibrationLog:
    def __init__(self, root: Path, keep_days: int = 7):
        self.root = root
        self.keep_days = keep_days
        self._last_heartbeat = 0.0
        self._cpu0 = self._cpu()
        self._wall0 = time.monotonic()
        self.prune()

    @staticmethod
    def _cpu() -> float:
        r = resource.getrusage(resource.RUSAGE_SELF)
        return r.ru_utime + r.ru_stime

    def _day(self) -> Path:
        d = self.root / date.today().isoformat()
        (d / "thumbs").mkdir(parents=True, exist_ok=True)
        return d

    def prune(self) -> None:
        cutoff = date.today() - timedelta(days=self.keep_days)
        for d in self.root.glob("????-??-??"):
            try:
                if date.fromisoformat(d.name) < cutoff:
                    shutil.rmtree(d)
            except ValueError:
                continue

    def settled(self, s: Settled) -> None:
        day = self._day()
        stamp = datetime.fromtimestamp(s.t).strftime("%H%M%S")
        thumb = f"thumbs/{stamp}.jpg"
        if s.image is not None:
            img = s.image.copy()
            img.thumbnail((320, 320))
            img.save(day / thumb, quality=70)
        row = {k: v for k, v in s.__dict__.items() if k != "image"}
        row.update(time=datetime.fromtimestamp(s.t).isoformat(timespec="seconds"), thumb=thumb)
        with open(day / "settled.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._heartbeat()

    def status(self, state: str) -> None:
        self._write_status({"time": datetime.now().isoformat(timespec="seconds"), "state": state})
        self._heartbeat()

    def _heartbeat(self) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat < 3600:
            return
        cpu, wall = self._cpu() - self._cpu0, now - self._wall0
        self._write_status({"time": datetime.now().isoformat(timespec="seconds"), "state": "heartbeat",
                            "cpu_pct_avg": round(100 * cpu / max(wall, 1e-6), 2),
                            "rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)})
        self._last_heartbeat = now
        self.prune()

    def _write_status(self, row: dict) -> None:
        with open(self._day() / "status.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def run(root: Path, cfg: ObserverConfig | None = None) -> None:
    log = CalibrationLog(root)
    obs = ScreenObserver(cfg or ObserverConfig(), on_settled=log.settled, on_status=log.status)
    print(f"screen logger: main display every {obs.cfg.interval_s}s, embeddings on {obs.embedder.device}; "
          f"logging to {root}", flush=True)
    obs.run_forever()
