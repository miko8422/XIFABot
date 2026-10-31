"""Entry points.

  uv run maid-voice serve            local web UI, hold Space to talk  (http://127.0.0.1:8765)
  uv run maid-voice bench [--runs 3] replay fixture wavs through the full pipeline + real playback
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from aiohttp import WSMsgType, web

from .asr import OmlxASR
from .audio import Capture, Player
from .config import PROJECT_ROOT, Config, load_config
from .controller import TurnController
from .handsfree import HandsFree
from .llm import OpenAICompatLLM
from .metrics import MetricsSink, now_ns
from .tts import OmlxTTS
from .vad import VADConfig

STATIC = Path(__file__).parent / "static"
HEADLINE_KEYS = ["first_sound_ms", "endpoint_ms", "endpoint_to_sound_ms", "asr_ms", "llm_ttft_ms", "llm_first_clause_ms",
                 "tts_first_audio_ms", "playback_lag_ms", "stop_ms"]


def vad_config(cfg: Config) -> VADConfig:
    v = cfg.raw.get("vad", {})
    return VADConfig(**{k: v[k] for k in VADConfig.__dataclass_fields__ if k in v})


def load_persona(cfg: Config):
    """Role card + expression bank from [persona]; None keeps the built-in prompt."""
    p = cfg.raw.get("persona", {})
    if not p.get("enabled", False):
        return None
    from .persona import Persona  # imports torch, only when a persona is used
    expressions = p.get("expressions")
    return Persona(PROJECT_ROOT / p["card"], PROJECT_ROOT / expressions if expressions else None, p.get("top_k", 4))


def build(cfg: Config, run_name: str, emit=None):
    a, l, t = cfg["asr"], cfg["llm"], cfg["tts"]
    asr = OmlxASR(a["base_url"], a["model"], a.get("language", ""), a.get("prompt", ""))
    llm = OpenAICompatLLM(
        base_url=cfg.secret(l["base_url_env"]), api_key=cfg.secret(l["api_key_env"]),
        model=l["model"], max_tokens=l["max_tokens"], extra_body=l.get("extra_body", {}),
        price_in=l.get("price_in_per_mtok", 0.0), price_out=l.get("price_out_per_mtok", 0.0),
        persona=load_persona(cfg),
    )
    tts = OmlxTTS(t["base_url"], t["model"], t.get("streaming_interval", 0.2),
                  cfg["audio"]["playback_sample_rate"], t.get("temperature"))
    player = Player(cfg["audio"]["playback_sample_rate"], cfg["audio"]["output_device"])
    run_dir = PROJECT_ROOT / cfg["metrics"]["dir"] / f"{datetime.now():%Y%m%d-%H%M%S}-{run_name}"
    ctl = TurnController(asr, llm, tts, player, cfg.voice, MetricsSink(run_dir), emit,
                         history_turns=l.get("history_turns", 4), output_device=player.device_name)
    player.on_event = ctl.player_event
    return ctl, player, run_dir


async def warmup(ctl: TurnController) -> dict:
    out = {}
    for name, coro in [("asr", ctl.asr.warmup()), ("llm", ctl.llm.warmup()),
                       ("tts", ctl.tts.warmup(ctl.voice))]:
        t0 = time.monotonic()
        await coro
        out[name] = round((time.monotonic() - t0) * 1000)
    return out


def percentiles(values: list[float]) -> dict:
    v = sorted(x for x in values if x is not None)
    if not v:
        return {}
    pick = lambda p: v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]
    return {"n": len(v), "p50": round(statistics.median(v)), "p90": round(pick(90)),
            "p95": round(pick(95)), "min": round(v[0]), "max": round(v[-1])}


# ---------------------------------------------------------------- bench
async def feed_realtime(hf: HandsFree, pcm: np.ndarray, sr: int, tail_s: float) -> None:
    """Play a fixture into the endpointer at real-time pace, followed by low-level noise."""
    frame = sr // 50
    noise = (0.001 * np.random.default_rng(0).standard_normal(int(sr * tail_s))).astype("float32")
    lead = (0.001 * np.random.default_rng(1).standard_normal(int(sr * 0.4))).astype("float32")
    audio = np.concatenate([lead, pcm, noise])
    t0 = time.monotonic()
    for i in range(0, len(audio) - frame + 1, frame):
        target = t0 + (i + frame) / sr
        await asyncio.sleep(max(0.0, target - time.monotonic()))
        await hf.process(audio[i:i + frame], now_ns())


async def bench(cfg: Config, runs: int, fixtures: Path, gap_s: float, handsfree: bool,
                speculative: bool = True) -> None:
    ctl, player, run_dir = build(cfg, "bench-handsfree" if handsfree else "bench")
    hf = HandsFree(ctl, cfg["audio"]["capture_sample_rate"], vad_config(cfg),
                   speculative=speculative and cfg.raw.get("vad", {}).get("speculative", True)) if handsfree else None
    player.start()
    print(f"output device: {player.device_name}\nwarmup ms: {await warmup(ctl)}")
    if hf:
        print(f"hands-free: {hf.endpointer.cfg}  speculative={hf.speculative}")
    files = sorted(fixtures.glob("*.wav"))
    for run in range(runs):
        for f in files:
            pcm, sr = sf.read(f, dtype="float32")
            n_before = len(ctl.completed)
            if hf:
                await feed_realtime(hf, pcm, sr, tail_s=1.2)
                while len(ctl.completed) == n_before or (ctl._task and not ctl._task.done()):
                    await asyncio.sleep(0.02)
                rec = [r for r in ctl.completed if not str(r["error"]).endswith("speculation_dropped")][-1]
            else:
                # the fixture is "spoken" now: its last sample is the end of speech
                t_end = now_ns()
                await ctl.submit(pcm, sr, t_end, t_end, source=f"fixture:{f.name}")
                rec = ctl.completed[-1]
            print(f"[{run + 1}:{f.stem:>14}] first_sound={rec['first_sound_ms']}ms "
                  + (f"(endpoint {rec['endpoint_ms']} -> sound {rec['endpoint_to_sound_ms']}) " if hf else "")
                  + f"(asr {rec['asr_ms']} | llm ttft {rec['llm_ttft_ms']} / clause {rec['llm_first_clause_ms']}"
                  f" | tts {rec['tts_first_audio_ms']} | play {rec['playback_lag_ms']})  "
                  f"{rec['user_text']!r} -> {rec['assistant_text']!r}"
                  + (f"  ERROR {rec['error']}" if rec["error"] else ""))
            await asyncio.sleep(gap_s)
    dropped = sum(1 for r in ctl.completed if str(r["error"]).endswith("speculation_dropped"))
    if hf:
        print(f"\nspeculative generations dropped (user resumed): {dropped}")
    stats = {k: percentiles([r.get(k) for r in ctl.completed if not r["error"]]) for k in HEADLINE_KEYS}
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print("\nlatency ms (successful turns):")
    for k, s in stats.items():
        if s:
            print(f"  {k:<22} p50={s['p50']:>5}  p90={s['p90']:>5}  p95={s['p95']:>5}  "
                  f"min={s['min']:>5}  max={s['max']:>5}  n={s['n']}")
    print(f"\nrecords: {run_dir}")
    player.close()


# ---------------------------------------------------------------- serve
@dataclass
class ServeContext:
    """What an extension (e.g. maid_agent) gets to work with inside `serve`."""
    cfg: Config
    ctl: TurnController
    capture: Capture
    hf: HandsFree
    broadcast: Callable
    info: dict

    def user_speaking(self) -> bool:
        return self.capture.recording or (self.hf.enabled and self.hf.endpointer.in_speech)


async def serve(cfg: Config, host: str, port: int, index: Path = STATIC / "index.html",
                run_name: str = "live", extension=None) -> None:
    """The voice gateway. `extension` (optional) has `async start(ctx) -> list[Task]` and
    `async command(cmd) -> bool`; without one this is the plain voice test page."""
    clients: set[web.WebSocketResponse] = set()

    async def broadcast(event: dict) -> None:
        msg = json.dumps(event, ensure_ascii=False)
        for ws in list(clients):
            try:
                await ws.send_str(msg)
            except ConnectionError:
                clients.discard(ws)

    ctl, player, run_dir = build(cfg, run_name, broadcast)
    capture = Capture(cfg["audio"]["capture_sample_rate"], cfg["audio"]["input_device"])
    vad_raw = cfg.raw.get("vad", {})
    hf = HandsFree(ctl, capture.sample_rate, vad_config(cfg), broadcast,
                   speculative=vad_raw.get("speculative", True), barge_in=vad_raw.get("barge_in", False))
    hf.enabled = False  # start in push-to-talk; the UI switches modes
    capture.on_frame = hf.feed_threadsafe
    player.start()
    capture.start()
    info = {"type": "info", "asr": ctl.asr.model, "llm": ctl.llm.model, "tts": ctl.voice.model or ctl.tts.model,
            "voice": ctl.voice.name, "voices": list(cfg.voices), "input": capture.device_name,
            "output": player.device_name, "run_dir": str(run_dir), "mode": "ptt",
            "end_silence_ms": hf.endpointer.cfg.end_silence_ms, "barge_in": hf.barge_in}
    print(f"warming up models ...")
    info["warmup_ms"] = await warmup(ctl)
    print(f"warmup ms: {info['warmup_ms']}")
    ctx = ServeContext(cfg, ctl, capture, hf, broadcast, info)
    ext_tasks = await extension.start(ctx) if extension else []

    async def handle_ws(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clients.add(ws)
        await ws.send_str(json.dumps(info))
        await ws.send_str(json.dumps({"type": "state", "state": ctl.state}))
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            cmd = json.loads(msg.data)
            if cmd["cmd"] == "ptt_down" and not hf.enabled:
                # pressing talk while she speaks is an explicit barge-in
                await ctl.stop("ptt_barge_in")
                capture.begin()
                await broadcast({"type": "state", "state": "capturing"})
            elif cmd["cmd"] == "ptt_up" and not hf.enabled:
                t_release = now_ns()
                utt = capture.end()
                if utt.pcm.size < capture.sample_rate * 0.2 or utt.t_speech_end_ns is None:
                    await broadcast({"type": "state", "state": "listening"})
                    await broadcast({"type": "error", "message": "no speech detected"})
                    continue
                ctl.submit(utt.pcm, utt.sample_rate, utt.t_speech_end_ns, t_release,
                           t_speech_start_ns=utt.t_speech_start_ns)
            elif cmd["cmd"] == "stop":
                await ctl.stop("user_stop")
            elif cmd["cmd"] == "mode":
                hf.enabled = cmd.get("mode") == "handsfree"
                hf.endpointer.reset()
                info["mode"] = "handsfree" if hf.enabled else "ptt"
                await broadcast(info)
            elif cmd["cmd"] == "end_silence":
                ms = int(cmd.get("ms", 600))
                hf.endpointer.cfg.end_silence_ms = max(200, min(2000, ms))
                hf.endpointer.cfg.spec_silence_ms = min(hf.endpointer.cfg.spec_silence_ms,
                                                        hf.endpointer.cfg.end_silence_ms - 100)
                info["end_silence_ms"] = hf.endpointer.cfg.end_silence_ms
                await broadcast(info)
            elif cmd["cmd"] == "barge_in":
                hf.barge_in = bool(cmd.get("on"))
                info["barge_in"] = hf.barge_in
                await broadcast(info)
            elif cmd["cmd"] == "voice" and cmd.get("name") in cfg.voices:
                ctl.voice = cfg.voices[cmd["name"]]
                info["voice"] = ctl.voice.name
                info["tts"] = ctl.voice.model or ctl.tts.model
                await broadcast(info)
                await ctl.tts.warmup(ctl.voice)  # a voice with its own model may need loading
            elif extension:
                await extension.command(cmd)
        clients.discard(ws)
        return ws

    async def level_loop():
        while True:
            await asyncio.sleep(0.1)
            if clients:
                await broadcast({"type": "level", "rms": round(capture.level, 4)})

    app = web.Application()
    app.router.add_get("/", lambda r: web.FileResponse(index))
    app.router.add_get("/ws", handle_ws)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    print(f"open http://{host}:{port}  — hold Space to talk (or switch to hands-free), Esc to stop.  "
          f"records: {run_dir}", flush=True)
    level = asyncio.create_task(level_loop())
    hf_task = asyncio.create_task(hf.run())
    try:
        await asyncio.Event().wait()
    finally:
        for task in [level, hf_task, *ext_tasks]:
            task.cancel()
        capture.close()
        player.close()
        await runner.cleanup()


def main() -> None:
    ap = argparse.ArgumentParser(prog="maid-voice")
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "default.toml")
    ap.add_argument("--voice", help="voice profile, overrides [tts].voice_profile")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    b = sub.add_parser("bench")
    b.add_argument("--runs", type=int, default=3)
    b.add_argument("--fixtures", type=Path, default=PROJECT_ROOT / "runs" / "asr_fixtures")
    b.add_argument("--gap", type=float, default=0.5, help="seconds between turns")
    b.add_argument("--handsfree", action="store_true", help="stream fixtures through the VAD endpointer")
    b.add_argument("--no-speculative", action="store_true", help="hands-free without speculative start")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.voice:
        cfg.raw["tts"]["voice_profile"] = args.voice
    try:
        if args.cmd == "serve":
            asyncio.run(serve(cfg, args.host, args.port))
        else:
            asyncio.run(bench(cfg, args.runs, args.fixtures, args.gap, args.handsfree,
                              not args.no_speculative))
    except KeyboardInterrupt:
        pass
