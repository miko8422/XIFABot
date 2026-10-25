# XIFABot — ナツメ, a personal maid agent

A voice companion for あきら: she listens (English or Japanese), answers in Japanese in a fine-tuned ナツメ voice, and is starting to notice what happens on the screen.

| Path | What |
| --- | --- |
| `maid_agent_MASTER_GOAL.md` | Product vision and roadmap (v0.1 voice → v1.0 agent) |
| `maid_agent_RESPONSE_DESIGN.md` | Response system, role card and screen trigger design (MaiBot-inspired) |
| `maid_voice_cascade_GOAL.md`, `_ARCHITECTURE.md` | The v0.1 voice experiment spec |
| `maid_voice/` | Voice runtime: Qwen3-ASR → DeepSeek V4.1 Flash → fine-tuned Qwen3-TTS, streaming, with latency traces. Its web UI (port 8765) is the plain voice test gateway. `sft/` is the voice fine-tuning framework |
| `maid_agent/` | The agent: role card + expression bank (`persona/`), screen observer and the screen → speech pipeline. Web UI on port 8766 |

## Run (macOS, Apple Silicon)

Needs [uv](https://docs.astral.sh/uv/), oMLX on `127.0.0.1:9800` (ASR, TTS, Qwen3.6 for vision), and `.env` with the DeepSeek `BASE_URL`/`API_KEY`.

```sh
cd maid_voice && uv run maid-voice serve     # voice gateway, http://127.0.0.1:8765
cd maid_agent && uv run maid-agent serve     # agent + screen pipeline, http://127.0.0.1:8766
```

Run one of them at a time with hands-free mode; both use the same microphone and speakers.

## Not in this repository

- `.env` (secrets).
- The game voice data (`nat/`, system voice clips, `maid_voice/voices/`): © YUZUSOFT / JUNOS, personal use only.
- Model checkpoints, eval audio, screenshots and logs (`runs/`).
- The fine-tuned voice model itself: it lives in oMLX's model folder as `natsume-qwen3tts-sft-e1-8bit`.

`maid_agent/persona/` does contain short quotes of her lines (the expression bank and the role card examples). Keep this repository private.
