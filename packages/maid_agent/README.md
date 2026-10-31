# maid_agent

ナツメ's persona and perception, around the `maid_voice` runtime (used as a library).

```sh
uv run maid-agent serve        # voice + screen pipeline, web UI on http://127.0.0.1:8766
uv run maid-agent screen-log   # observer only: log settled screen states for calibration, never speaks
uv run pytest
```

- `persona/natsume.v1.toml`: role card (approved by あきら 2026-10-26). `natsume_expressions.jsonl/.npy`: 1079 "situation → how she says it → her real line" entries, mined by `../maid_voice/scripts/build_expressions.py`.
- `src/maid_agent/router.py`, `runtime.py`, `demo.py`: request routing, the Maid-facing layer over the harness (replies, merged task notifications), and the scripted `maid-agent demo`.
- The screen observer now lives in `../maid_harness/src/maid_harness/reporters/screen/`.
- `src/maid_agent/proactive.py`: fired change → placeholder gate (cooldown, never while either of you is talking) → local vision (no tokens when nothing is notable) → one DeepSeek decision with the role card → `ctl.say()`. Decisions are logged to `runs/agent/<date>/decisions.jsonl`. The real "when to reply" design is still open.

Design: `../../docs/maid_agent_RESPONSE_DESIGN.md`.
