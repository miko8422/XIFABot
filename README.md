# XIFABot — the Maid Agent (v0.1 demo)

ナツメ, a Japanese voice companion that lives next to your Mac. She talks with low latency (about 1 s to first sound), knows what she is able to do, hands heavy work to sandboxed Codex sub-agents, asks you by voice when they need an answer or a permission, watches your screen and activity, and decides when to speak and when to stay quiet.

```text
        you (voice, web UI)
              │
  maid_agent    who she is: persona, routing through tools, speech arbiter, memory, web UI
  maid_voice    the voice loop: VAD, ASR, LLM, TTS, barge-in        (both used by maid_agent)
  maid_harness  where work and facts live: tasks, Codex sub-agents, reporters (activity, screen)
```

Design and plan: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Code map: [`docs/CODEBASE.md`](docs/CODEBASE.md).

## What she can do in this version

| | |
| --- | --- |
| Talk | Japanese, low latency, push-to-talk or hands-free, interruptible. Understands English words and sentences, always answers in Japanese |
| Know herself | Her 「できること」 and her tools are generated from what is actually connected, so she neither invents abilities nor denies real ones |
| Hand off work | Research with live web search, coding and file work, done by a Codex sub-agent in its own sandboxed folder. She acknowledges at once and reports the result at the next pause |
| Follow tasks | 「どうなった？」 from the real state; cancel, follow up, or change the conditions of a task by voice; read out full results; search older tasks |
| Ask before acting | Sub-agents read and write only their own folder. Anything else (another folder, network, full access) is asked by voice, yes or no, for that task only |
| Survive restarts | Sub-agents keep running if `serve` stops. She reports what finished while she was away, asks pending questions again, and remembers the conversation |
| Watch, gently | Knows if you are working, gaming, in a call or away. Holds task news during calls and while you are away, keeps it short while you game, says おかえり when you are back. Comments on the screen only when something is worth it, with a cooldown |

Not yet: memory of facts about you across days (P5), operating apps (P6, practice mode only), awareness of your own Codex/Claude sessions (planned).

## Requirements

- macOS on Apple Silicon, Python 3.12+, [uv](https://docs.astral.sh/uv/).
- [oMLX](https://github.com/jundot/omlx) on `127.0.0.1:9800` serving the local models in `packages/maid_voice/config/default.toml`:
  - ASR `mlx-community--Qwen3-ASR-0.6B-8bit`
  - TTS: Qwen3-TTS plus a voice. **Not included**: the fine-tuned ナツメ voice (`natsume-qwen3tts-sft-e1-8bit`) and the clone reference audio under `voices/` are private assets. Point `[tts]` and `[voices.*]` at your own model and voice.
  - Vision for the screen (optional): `mlx-community--Qwen3.6-35B-A3B-4bit`.
- An OpenAI-compatible chat API for her replies (DeepSeek by default): copy `.env.example` to `.env` and fill it in.
- For real sub-agents: the [Codex CLI](https://github.com/openai/codex), logged in. Sub-agents run in their own clean `CODEX_HOME` built from your login, with none of your skills, MCP servers or sessions. Without Codex, set `runner = "fake"` in `[agent]`.
- macOS permissions for your terminal: Microphone; Screen Recording (screen features). Optional: [ActivityWatch](https://activitywatch.net) for window titles; without it the macOS front app and idle time are used.

## Run

```bash
uv sync
cp .env.example .env            # then fill in BASE_URL and API_KEY
uv run pytest                   # 90 tests, no models or network needed
uv run maid-agent serve         # web UI on http://127.0.0.1:8766
uv run maid-agent serve --no-screen      # without screen capture
```

Without any model services you can still run `uv run maid-agent demo` (a scripted text demo of the harness) and the tests.

Settings are in `packages/maid_voice/config/default.toml`:
- `[agent]`: sub-agent model, budget and limits.
- `[agent.activity]`: which apps count as a game or a call. List any app as a game to simulate a play session.
- `[agent.speech]`: when she speaks, per activity.

## Data and privacy

Everything she records stays local under `packages/*/runs/` (git-ignored):
- turns and latencies
- the conversation log
- the task dataset with each sub-agent's full event stream
- screen calibration logs

Screen pixels go only to the local vision model. A sub-agent's environment never contains your `.env` keys. Text sent to the chat API: the conversation, task summaries and screen descriptions.
