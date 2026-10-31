# XIFABot Demo

The Maid Agent: a low-latency Japanese voice companion on top, and a harness env below it where sub-agents do heavy work and reporters watch the system. See `docs/ARCHITECTURE.md`.

## What is here

- `packages/maid_voice`: microphone capture, Japanese ASR, DeepSeek chat, persona retrieval, Japanese TTS, interruption and playback.
- `packages/maid_agent`: the Maid. Persona, request routing, task notifications, the prototype proactive screen remarks, and the CLI (`maid-agent`). Depends on `maid_voice` and `maid_harness`.
- `packages/maid_harness`: the harness env. Event inbox, task registry, supervisor and demo worker, MCP stdio server, and the screen reporter. Never imports `maid_agent`.
- `experiments/asr-ja`: the Japanese ASR comparison experiment and its recorded results.
- `docs/`: `ARCHITECTURE.md` (the current architecture and plan), plus the voice runtime design, persona/screen design notes, extracted feature map, and migration notes.

The current ASR setting is Japanese-only input (`language = "ja"`). The agent's output remains Japanese through the existing system prompt.

## Run locally

From this directory, create the environment with `uv sync --all-packages`. A plain `uv sync` installs only the empty root project and removes the packages. Then configure the local `.env` and the oMLX endpoints described in `docs/EXTRACTED_FEATURES.md`.

```bash
uv sync --all-packages
uv run pytest        # all three packages
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
