import numpy as np
from PIL import Image, ImageDraw

from maid_harness.reporters.screen.observer import ObserverConfig, Scorer


def screen(color, popup=False):
    img = Image.new("RGB", (640, 400), color)
    if popup:
        ImageDraw.Draw(img).rectangle((480, 300, 620, 380), fill=(255, 255, 255))
    return img


class FakeEmbed:
    """Embedding = the average colour, normalized: a different screen gets a different vector."""
    def __call__(self, image):
        v = np.asarray(image, dtype=np.float32).mean(axis=(0, 1)) + 1.0
        return v / np.linalg.norm(v)


def feed_until_settled(scorer, img, app="com.x", t0=0.0):
    out = None
    for i in range(4):
        out = scorer.feed(t0 + i, img, "X", app) or out
    return out


def test_first_state_fires_then_static_screen_never_fires_again():
    sc = Scorer(ObserverConfig(), FakeEmbed())
    first = feed_until_settled(sc, screen((20, 20, 20)))
    assert first and first.would_fire
    assert all(sc.feed(10 + i, screen((20, 20, 20)), "X", "com.x") is None for i in range(5))


def test_small_popup_fires_as_local_change():
    sc = Scorer(ObserverConfig(), FakeEmbed())
    feed_until_settled(sc, screen((20, 20, 20)))
    s = feed_until_settled(sc, screen((20, 20, 20), popup=True), t0=10)
    assert s and s.would_fire and "local_popup" in s.reasons and not s.app_switch


def test_returning_to_a_recently_seen_screen_does_not_fire():
    sc = Scorer(ObserverConfig(), FakeEmbed())
    feed_until_settled(sc, screen((20, 20, 20)), app="com.a")
    feed_until_settled(sc, screen((200, 30, 30)), app="com.b", t0=10)
    back = feed_until_settled(sc, screen((20, 20, 20)), app="com.a", t0=20)
    assert back and not back.would_fire and "seen_recently" in back.reasons
