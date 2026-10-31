"""Maid Agent entry points: voice runtime, screen observation, and harness demo."""

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
    h = sub.add_parser("harness-mcp", help="run the v0.1 harness as an MCP stdio server")
    h.add_argument("--db", type=Path, default=Path("runs/harness/tasks.sqlite3"))
    d = sub.add_parser("demo", help="run a scripted Maid + harness v0.1 demo")
    l = sub.add_parser("latency", help="benchmark deterministic Maid request triage")
    l.add_argument("--samples", type=int, default=1000)
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
    elif args.cmd == "harness-mcp":
        import asyncio

        from .harness.core import Harness
        from .harness.mcp_server import run_stdio

        asyncio.run(run_stdio(Harness(args.db)))
    elif args.cmd == "demo":
        import asyncio

        from .harness.demo import run_demo

        asyncio.run(run_demo())
    elif args.cmd == "latency":
        import json

        from .harness.router import benchmark

        print(json.dumps(benchmark(args.samples), ensure_ascii=False))
