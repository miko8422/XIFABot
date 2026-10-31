# maid_agent

ナツメ's persona and perception, around the `maid_voice` runtime (used as a library).

```sh
uv run maid-agent serve        # voice + harness tasks + screen pipeline, web UI on http://127.0.0.1:8766
uv run maid-agent serve --no-screen --routing keyword   # without screen capture / with the keyword router
uv run python packages/maid_agent/scripts/probe_delegate.py   # delegate latency + routing probe (calls DeepSeek)
uv run maid-agent screen-log   # observer only: log settled screen states for calibration, never speaks
uv run pytest
```

- `persona/natsume.v2.toml`: role card in use (v1 approved by あきら 2026-10-26; v2 2026-10-31 only rewrites `boundaries` so her abilities come from the capability list). `natsume_expressions.jsonl/.npy`: 1079 "situation → how she says it → her real line" entries, mined by `../maid_voice/scripts/build_expressions.py`.
- `src/maid_agent/agent.py`: the `serve` extension. Harness + `turns.py` (routing through the `delegate` tool in the reply call, ARCHITECTURE §4.2) + `arbiter.py` (when task news is spoken, §4.3) + the screen pipeline. Tools and the T1 state block are in `tools.py`. Settings are in `[agent]` of `../maid_voice/config/default.toml`. Sub-agents are faked until P3.
- `src/maid_agent/router.py`, `runtime.py`, `demo.py`: the keyword router (now the fallback), request routing, the Maid-facing layer over the harness (replies, merged task notifications), and the scripted `maid-agent demo`.
- The screen observer now lives in `../maid_harness/src/maid_harness/reporters/screen/`.
- `src/maid_agent/proactive.py`: fired change → placeholder gate (cooldown, never while either of you is talking) → local vision (no tokens when nothing is notable) → one DeepSeek decision with the role card → `ctl.say()`. Decisions are logged to `runs/agent/<date>/decisions.jsonl`. The real "when to reply" design is still open.

Design: `../../docs/maid_agent_RESPONSE_DESIGN.md`.
