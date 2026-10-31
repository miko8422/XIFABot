"""macOS adapter: front app, idle time, lock state. A Windows adapter would expose the same three functions."""

from dataclasses import dataclass

import Quartz
from AppKit import NSWorkspace


@dataclass(frozen=True)
class Foreground:
    app: str
    bundle_id: str


def foreground() -> Foreground:
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None:
        return Foreground("", "")
    return Foreground(str(app.localizedName() or ""), str(app.bundleIdentifier() or ""))


def idle_seconds() -> float:
    """Seconds since the last keyboard/mouse input."""
    return float(Quartz.CGEventSourceSecondsSinceLastEventType(
        Quartz.kCGEventSourceStateCombinedSessionState, Quartz.kCGAnyInputEventType))


def screen_locked() -> bool:
    session = Quartz.CGSessionCopyCurrentDictionary() or {}
    return bool(session.get("CGSSessionScreenIsLocked", False))
