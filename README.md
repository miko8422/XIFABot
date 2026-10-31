# XIFABot Demo

This is the clean project layout for the current XIFABot Maid Agent work. It keeps the working voice runtime and screen-aware agent as two small packages so they can later become parts of a larger Maid Agent system.

## What is here

- `packages/maid_voice`: microphone capture, Japanese ASR, DeepSeek chat, persona retrieval, Japanese TTS, interruption and playback.
- `packages/maid_agent`: the Maid Agent wrapper, screen observer, vision analysis, proactive attention gate, and the same voice runtime with screen events attached.
- `experiments/asr-ja`: the Japanese ASR comparison experiment and its recorded results.
- `docs/`: `ARCHITECTURE.md` (the current architecture and plan), plus the voice runtime design, persona/screen design notes, extracted feature map, and migration notes.

The current ASR setting is Japanese-only input (`language = "ja"`). The agent's output remains Japanese through the existing system prompt.

## Run locally

From this directory, create the new environment with `uv sync`. Then configure the local `.env` and the oMLX endpoints described in `docs/EXTRACTED_FEATURES.md`.

```bash
uv sync
cd packages/maid_voice
uv run maid-voice serve
```

To run the screen-aware Maid Agent instead:

```bash
cd packages/maid_agent
uv run maid-agent serve
```

## Harness v0.1 demo

The first Maid/harness slice is runnable without external model services. It keeps the Maid fast, sends heavy work to one supervised demo worker per task, exposes the harness through MCP stdio, and requires confirmation only for computer-use tasks.

```bash
uv run maid-agent demo
uv run maid-agent latency --samples 5000
uv run maid-agent harness-mcp --db runs/harness/tasks.sqlite3
```

The MCP server currently exposes `task_submit`, `task_get`, `task_list`, `task_cancel`, `task_confirm`, `event_publish`, and `event_list`. The demo worker is a placeholder for the real per-task sub-agent runner planned in `docs/ARCHITECTURE.md`; task and event records are already persisted in SQLite.

The runtime still expects local model services and private voice assets. Those are intentionally kept outside the clean source copy and are listed in `docs/MIGRATION.md`.
