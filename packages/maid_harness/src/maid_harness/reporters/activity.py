"""Desktop activity reporter (ARCHITECTURE §5.4, P4): what あきら is doing, as facts the Maid can use.

  source   ActivityWatch's local REST API when it runs (front app, window title, AFK), else the macOS
           adapter in reporters/screen/platform_mac.py (front app, idle time, screen lock)
  classify away (locked or idle) > call > game > working, from `activity_profiles` (config): app names,
           bundle ids or window-title words; any app can be listed as a game to simulate a session
  events   activity.changed {activity, app, previous, previous_s} once a new state has held for
           `debounce_s`, and user.back {away_s} when away ends. Facts only: the reporter never speaks or acts.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

ACTIVITIES = ("working", "gaming", "call", "away")


@dataclass
class ActivityProfiles:
    game: tuple[str, ...] = ("Steam", "com.valvesoftware", "Battle.net", "Epic Games", "Minecraft", "League of Legends",
                             "Genshin", "原神")
    call: tuple[str, ...] = ("zoom.us", "Microsoft Teams", "FaceTime", "Webex", "Google Meet", "Meet -", "Slack Huddle")
    afk_after_s: float = 180.0      # no keyboard/mouse input this long: away
    debounce_s: float = 5.0         # a new activity must hold this long before it counts (alt-tab is not a change)

    @classmethod
    def from_raw(cls, raw: dict) -> "ActivityProfiles":
        kw = {k: tuple(raw[k]) if k in ("game", "call") else raw[k] for k in cls.__dataclass_fields__ if k in raw}
        return cls(**kw)


@dataclass
class Sample:
    app: str
    bundle_id: str = ""
    title: str = ""
    idle_s: float = 0.0
    locked: bool = False


@dataclass
class ActivityState:
    activity: str
    app: str
    since: float
    title: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"activity": self.activity, "app": self.app, "since": self.since, "title": self.title,
                "source": self.source}


def classify(s: Sample, p: ActivityProfiles) -> str:
    if s.locked or s.idle_s >= p.afk_after_s:
        return "away"
    hay = f"{s.app}\n{s.bundle_id}\n{s.title}".casefold()
    if any(w.casefold() in hay for w in p.call):
        return "call"
    if any(w.casefold() in hay for w in p.game):
        return "gaming"
    return "working"


class NativeMacSource:
    name = "macos"

    def __init__(self) -> None:
        from .screen import platform_mac
        self.plat = platform_mac

    async def sample(self) -> Sample:
        fg = self.plat.foreground()
        return Sample(fg.app, fg.bundle_id, "", self.plat.idle_seconds(), self.plat.screen_locked())


class ActivityWatchSource:
    """aw-server on 127.0.0.1:5600: the newest window event and AFK state."""
    name = "activitywatch"

    def __init__(self, base_url: str = "http://127.0.0.1:5600") -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=2.0, trust_env=False)
        self._window = self._afk = None

    async def available(self) -> bool:
        try:
            buckets = (await self._client.get("/api/0/buckets/")).json()
        except (httpx.HTTPError, ValueError):
            return False
        self._window = next((b for b in buckets if b.startswith("aw-watcher-window")), None)
        self._afk = next((b for b in buckets if b.startswith("aw-watcher-afk")), None)
        return self._window is not None

    async def _last(self, bucket: str) -> dict:
        events = (await self._client.get(f"/api/0/buckets/{bucket}/events", params={"limit": 1})).json()
        return events[0] if events else {}

    async def sample(self) -> Sample:
        win = (await self._last(self._window)).get("data", {})
        afk = await self._last(self._afk) if self._afk else {}
        away = afk.get("data", {}).get("status") == "afk"
        return Sample(str(win.get("app", "")), "", str(win.get("title", "")), 1e9 if away else 0.0, False)


class ActivityReporter:
    def __init__(self, publish: Callable[..., Any], profiles: ActivityProfiles | None = None, source: str = "auto",
                 poll_s: float = 2.0, clock=time.time) -> None:
        """publish(event_type, source, payload) is the harness inbox's publish."""
        self.publish = publish
        self.profiles = profiles or ActivityProfiles()
        self.source_pref = source
        self.poll_s = poll_s
        self.clock = clock
        self.source = None
        self.state: ActivityState | None = None
        self._candidate: tuple[str, str, float] | None = None  # (activity, app, first seen)

    async def pick_source(self):
        if self.source_pref in ("auto", "activitywatch"):
            aw = ActivityWatchSource()
            if await aw.available():
                return aw
            if self.source_pref == "activitywatch":
                raise RuntimeError("ActivityWatch is not running on 127.0.0.1:5600")
        return NativeMacSource()

    async def run(self) -> None:
        self.source = self.source or await self.pick_source()
        while True:
            try:
                self.observe(await self.source.sample())
            except Exception:  # noqa: BLE001 - a reporter hiccup must never take the Maid down
                pass
            await asyncio.sleep(self.poll_s)

    def observe(self, s: Sample) -> ActivityState | None:
        """Feed one sample. Returns the new state when the activity changed."""
        now = self.clock()
        activity = classify(s, self.profiles)
        app = "" if activity == "away" else s.app
        if self.state is None:
            self.state = ActivityState(activity, app, now, s.title, getattr(self.source, "name", ""))
            return self.state
        if activity == self.state.activity:
            self._candidate = None
            if app and app != self.state.app:  # same activity, another app: a fact, not a change
                self.state.app, self.state.title = app, s.title
            return None
        if self._candidate is None or self._candidate[0] != activity:
            self._candidate = (activity, app, now)
        if now - self._candidate[2] < self.profiles.debounce_s and activity != "away":
            return None
        prev = self.state
        self.state = ActivityState(activity, app, self._candidate[2], s.title, prev.source)
        self._candidate = None
        held = self.state.since - prev.since
        self.publish("activity.changed", "reporter.activity",
                     {"activity": activity, "app": app, "previous": prev.activity, "previous_s": round(held)})
        if prev.activity == "away":
            self.publish("user.back", "reporter.activity", {"away_s": round(held), "activity": activity, "app": app})
        return self.state

    def facts(self) -> list[str]:
        if self.state is None:
            return []
        mins = int((self.clock() - self.state.since) // 60)
        label = {"working": "作業中", "gaming": "ゲーム中", "call": "通話中", "away": "席を外している"}[self.state.activity]
        where = f"（{self.state.app}）" if self.state.app else ""
        return [f"あきらの今: {label}{where}、{mins}分前から" if mins else f"あきらの今: {label}{where}、さっきから"]
