"""P1 gate probe: does the reply call still answer as fast with the delegate tools attached, and does it route?

Usage: uv run packages/maid_agent/scripts/probe_delegate.py [--runs 3]
Same model, extra_body and role card as `maid-agent serve` (no expression bank, so no torch).
Each prompt runs with and without tools, interleaved, and prints TTFT, the spoken text and any tool calls.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx

from maid_harness import Capability, Harness

from maid_agent.tools import INSTRUCTIONS, capability_block, tools_for
from maid_voice.config import PROJECT_ROOT, load_config
from maid_voice.persona import RoleCard

STATE = """# 今の作業
- t-3f2a: Search the web for reviews of the Keychron Q1 Max keyboard / 実行中 / 依頼 2分前 / 40% reading reviews"""

# (prompt, expected): direct = no tool; delegate/cancel_task = that tool
PROMPTS = [
    ("今日ちょっと疲れたよ。", "direct"),
    ("How was your day?", "direct"),
    ("最新のPython 3.14の新機能を調べてくれる？", "delegate"),
    ("Can you look up how to set up ActivityWatch on macOS?", "delegate"),
    ("さっきのキーボードの件、どうなった？", "direct"),
    ("やっぱりキーボードの調べもの、やめて。", "cancel_task"),
    ("Steamを開いて。", "delegate"),
    ("このリポジトリのテストを全部走らせて、落ちたら直しておいて。", "delegate"),
    # 2026-10-31 live session: these went unrouted on the P0 build with card v1
    ("ああ一つの任務がある。今東京の天気を調べてくれ。", "delegate"),
    ("so I need you to check the weather on Tokyo today, it's a heavy task so launch a sub agent", "delegate"),
    ("今の僕のパソコンの画面、見えるかな。", "look_at_screen"),
    ("ナツメって何ができるの？", "direct"),
]


def capabilities():
    harness = Harness()
    harness.capabilities.register(Capability(
        "screen", "live", "can look at the user's main screen when asked (local vision, a few seconds), "
        "and notices big changes on its own", ("look_at_screen",), "the main display only"))
    return harness.capabilities.usable()


def run_once(client, model, extra, system, prompt, tools) -> dict:
    body = {"model": model, "stream": True, "max_tokens": 200, "stream_options": {"include_usage": True},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}], **extra}
    if tools:
        body["tools"] = tools_for(capabilities())
    t0 = time.monotonic()
    t_first = t_tool = None
    text, calls = "", {}
    with client.stream("POST", "/chat/completions", json=body) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.read()[:400]!r}")
        for line in r.iter_lines():
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            for ch in json.loads(line[5:]).get("choices", []):
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    t_first = t_first or time.monotonic()
                    text += delta["content"]
                for tc in delta.get("tool_calls") or []:
                    t_tool = t_tool or time.monotonic()
                    c = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    c["name"] += fn.get("name") or ""
                    c["arguments"] += fn.get("arguments") or ""
    ms = lambda t: None if t is None else round((t - t0) * 1000)
    return {"ttft_ms": ms(t_first), "tool_ms": ms(t_tool), "total_ms": ms(time.monotonic()), "text": text,
            "calls": [{"name": c["name"], "args": json.loads(c["arguments"] or "{}")} for c in calls.values()]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config()
    llm = cfg["llm"]
    card = RoleCard.load(PROJECT_ROOT / cfg["persona"]["card"])
    base = card.replyer_prompt()
    client = httpx.Client(base_url=cfg.secret(llm["base_url_env"]), timeout=30,
                          headers={"Authorization": f"Bearer {cfg.secret(llm['api_key_env'])}"})
    run_once(client, llm["model"], llm.get("extra_body", {}), base, "hi", False)  # warm the connection
    rows = []
    for run in range(args.runs):
        for prompt, expected in PROMPTS:
            for tools in (False, True):
                with_tools = "\n\n" + capability_block(capabilities()) + "\n\n" + INSTRUCTIONS + "\n\n" + STATE
                system = base + (with_tools if tools else "\n\n" + STATE)
                res = run_once(client, llm["model"], llm.get("extra_body", {}), system, prompt, tools)
                got = res["calls"][0]["name"] if res["calls"] else "direct"
                rows.append({"run": run, "prompt": prompt, "tools": tools, "expected": expected, "got": got, **res})
                print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    print()
    for tools in (False, True):
        sel = [r for r in rows if r["tools"] == tools and r["expected"] == "direct" and r["ttft_ms"]]
        t = [r["ttft_ms"] for r in sel]
        print(f"chitchat TTFT tools={tools!s:<5} p50={statistics.median(t):.0f} min={min(t)} max={max(t)} n={len(t)}")
    with_tools = [r for r in rows if r["tools"]]
    ok = sum(r["got"] == r["expected"] for r in with_tools)
    print(f"routing correct: {ok}/{len(with_tools)}")
    spoke_first = [r for r in with_tools if r["calls"] and r["ttft_ms"] and r["ttft_ms"] < (r["tool_ms"] or 1e9)]
    silent = [r for r in with_tools if r["calls"] and not r["text"].strip()]
    print(f"tool calls with an acknowledgement spoken first: {len(spoke_first)}, with no text at all: {len(silent)}")
    for r in with_tools:
        if r["got"] != r["expected"]:
            print(f"  MISS {r['prompt']!r}: expected {r['expected']}, got {r['got']}  text={r['text']!r}")
    out = Path(PROJECT_ROOT).parent / "maid_agent" / "runs" / f"probe_delegate-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"records: {out}")


if __name__ == "__main__":
    main()
