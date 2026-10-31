"""Screen reporter (ARCHITECTURE §5.6, P4): the observer and local vision as a harness reporter.

  observer thread   settled screen states, scored for change (observer.py); every one goes to the calibration log
  worker            the newest state that would fire -> wait until nobody is speaking -> local vision describes
                    it -> `screen.changed` event {app, activity, notable, worth_mentioning, sensitive}
  look(question)    on request from the reply call, also after waiting for quiet
  facts()           the latest observation for the state block

Vision waits while she or あきら speaks: vision and TTS share oMLX and the GPU, and overlapping made her voice
stutter (P2). A change that waited longer than `busy_wait_s` is stale and dropped. Reporters never speak or
decide; whether she says anything is the Maid's call (maid_agent/proactive.py, per-activity rules).
"""

from __future__ import annotations

import asyncio
import base64
import io
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from .logger import CalibrationLog
from .observer import ObserverConfig, ScreenObserver, Settled
from .vision import Observation, ScreenVision


def thumb_b64(image, size: int = 240) -> str:
    img = image.copy()
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class ScreenReporter:
    def __init__(self, publish: Callable[..., Any], root: Path, vision_url: str = "http://127.0.0.1:9800",
                 observer_cfg: ObserverConfig | None = None, busy: Callable[[], bool] = lambda: False,
                 on_ui: Callable[[dict], Any] = lambda e: None, busy_wait_s: float = 20.0) -> None:
        """publish(event_type, source, payload) = the inbox; busy() = someone is speaking; on_ui(event) = the web UI."""
        self.publish, self.root, self.busy, self.on_ui = publish, root, busy, on_ui
        self.observer_cfg = observer_cfg or ObserverConfig()
        self.vision = ScreenVision(vision_url)
        self.busy_wait_s = busy_wait_s
        self.observer: ScreenObserver | None = None
        self._latest: Settled | None = None
        self._wake = asyncio.Event()
        self._vision_lock = asyncio.Lock()  # one vision call at a time: a look and a change never overlap
        self.last_seen: tuple[float, str, Observation] | None = None

    # ---- lifecycle -------------------------------------------------------------------
    async def start(self) -> list[asyncio.Task]:
        self.loop = asyncio.get_running_loop()
        self.calib = CalibrationLog(self.root / "screen")
        self.observer = ScreenObserver(self.observer_cfg, on_settled=self._on_settled_thread,
                                       on_status=self._on_status_thread)
        threading.Thread(target=self.observer.run_forever, name="screen-observer", daemon=True).start()
        return [asyncio.create_task(self._worker())]

    @property
    def watching(self) -> bool:
        return bool(self.observer and self.observer.enabled)

    def set_watching(self, on: bool) -> None:
        if self.observer:
            self.observer.enabled = on

    # ---- observer thread -> event loop ---------------------------------------------------
    def _on_settled_thread(self, s: Settled) -> None:
        self.calib.settled(s)
        self.loop.call_soon_threadsafe(self._on_settled, s)

    def _on_status_thread(self, state: str) -> None:
        self.calib.status(state)
        self.loop.call_soon_threadsafe(lambda: self.on_ui({"type": "screen.status", "state": state}))

    def _on_settled(self, s: Settled) -> None:
        event = {"type": "screen.settled", "app": s.app, "would_fire": s.would_fire, "reasons": s.reasons,
                 "emb_dist": s.emb_dist, "tile_max": s.tile_max, "app_z": s.app_z, "seen_sim": s.seen_sim,
                 "time": datetime.fromtimestamp(s.t).strftime("%H:%M:%S")}
        if s.would_fire:
            event["thumb"] = thumb_b64(s.image)
            self._latest = s  # newest change wins; older unhandled ones are stale
            self._wake.set()
        self.on_ui(event)

    # ---- vision ---------------------------------------------------------------------------------
    async def wait_quiet(self, max_s: float) -> bool:
        deadline = time.monotonic() + max_s
        while self.busy():
            if time.monotonic() > deadline:
                return False
            await asyncio.sleep(0.1)
        return True

    async def _worker(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            s, self._latest = self._latest, None
            if s is None:
                continue
            if not await self.wait_quiet(self.busy_wait_s):
                self.publish("screen.skipped", "reporter.screen", {"app": s.app, "reason": "conversation busy; stale"})
                continue
            try:
                await self.describe(s)
            except Exception as e:  # noqa: BLE001 - vision trouble must never take the Maid down
                self.publish("screen.skipped", "reporter.screen", {"app": s.app, "reason": f"vision failed: {e}"})

    async def describe(self, s: Settled, manual: bool = False) -> Observation:
        async with self._vision_lock:
            self.on_ui({"type": "screen.stage", "stage": "looking"})
            image = await asyncio.to_thread(self.observer.grab_main)  # full resolution, the screen is settled
            obs = await self.vision.describe(image, s.app)
        self.last_seen = (time.time(), s.app, obs)
        self.publish("screen.changed", "reporter.screen", {
            "app": s.app, "activity": obs.activity, "notable": obs.notable, "worth_mentioning": obs.worth_mentioning,
            "sensitive": obs.sensitive, "vision_ms": round(obs.ms), "reasons": s.reasons, "manual": manual,
            "scores": {"emb_dist": s.emb_dist, "tile_max": s.tile_max, "app_z": s.app_z}})
        return obs

    async def look_now(self) -> None:
        """The UI's test button: treat the current screen as a fired change (no detector, no cooldown)."""
        image = await asyncio.to_thread(self.observer.grab_main)
        fg = self.observer.plat.foreground()
        await self.describe(Settled(time.time(), fg.app, fg.bundle_id, 0, 0, 0, False, None, 0, True, ["manual"], image),
                            manual=True)

    async def look(self, question: str = "") -> str:
        """Asked by the reply call: wait for her acknowledgement to finish, then look. English, for the LLM."""
        await self.wait_quiet(5.0)
        async with self._vision_lock:
            image = await asyncio.to_thread(self.observer.grab_main)
            app = self.observer.plat.foreground().app
            obs = await self.vision.describe(image, app, question)
        self.last_seen = (time.time(), app, obs)
        self.on_ui({"type": "screen.decision", "time": datetime.now().isoformat(timespec="seconds"), "app": app,
                    "manual": True, "action": "asked", "reason": question, "activity": obs.activity,
                    "notable": obs.notable, "vision_ms": round(obs.ms)})
        if obs.sensitive:
            return f"front app {app}; something private is on screen, so do not describe it"
        return f"front app {app}; {obs.activity}" + (f"; notable: {obs.notable}" if obs.notable else "") \
            + (f"; answer: {obs.answer}" if obs.answer else "")

    def facts(self, max_age_s: float = 600.0) -> list[str]:
        if self.last_seen is None or time.time() - self.last_seen[0] > max_age_s:
            return []
        t, app, obs = self.last_seen
        ago = int(time.time() - t)
        when = f"{ago // 60}分前" if ago >= 60 else f"{ago}秒前"
        what = "something private (not described)" if obs.sensitive else obs.activity
        return [f"画面（{when}に見た）: {app} / {what}"]
