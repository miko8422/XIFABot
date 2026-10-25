"""Maid Agent: persona, perception and (later) attention around the maid_voice runtime."""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / "static"


def main() -> None:
    ap = argparse.ArgumentParser(prog="maid-agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen-log", help="observe the main display and log settled states (no speaking)")
    s.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "screen")
    s.add_argument("--interval", type=float, default=2.0)
    v = sub.add_parser("serve", help="voice conversation + screen pipeline, with the agent web UI")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8766)
    v.add_argument("--cooldown", type=float, default=600.0, help="seconds between her unprompted remarks")
    args = ap.parse_args()
    if args.cmd == "screen-log":
        from .screen.logger import run
        from .screen.observer import ObserverConfig
        try:
            run(args.out, ObserverConfig(interval_s=args.interval))
        except KeyboardInterrupt:
            pass
    elif args.cmd == "serve":
        import asyncio

        from maid_voice.app import serve
        from maid_voice.config import load_config

        from .proactive import ProactiveConfig, ScreenProactive
        ext = ScreenProactive(PROJECT_ROOT / "runs", ProactiveConfig(cooldown_s=args.cooldown))
        try:
            asyncio.run(serve(load_config(), args.host, args.port, index=STATIC / "index.html",
                              run_name="agent", extension=ext))
        except KeyboardInterrupt:
            pass
