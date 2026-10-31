# Extracted feature map

## Working now

- Japanese microphone conversation: server-side PortAudio capture, adaptive VAD, whole-utterance Qwen ASR through oMLX, and turn cancellation.
- DeepSeek chat: OpenAI-compatible streaming client with the existing Japanese character prompt.
- Voice output: oMLX PCM TTS, clause segmentation, queued playback, interruption, generation IDs, and latency metrics.
- Persona behavior: Natsume role card plus embedding-based expression retrieval from the local expression bank.
- Screen observation: macOS frontmost-app metadata, MSS screenshots, settle and tile logic, embedding-based change checks, and a local vision model.
- Proactive speech prototype: cooldown, busy checks, a JSON decision prompt, and a `say` handoff into the voice controller.
- Runtime controls: watch, speak, cooldown, look-now, status, and screen log commands.
- ASR experiment: candidate Japanese model probes and recorded comparison results under `experiments/asr-ja`.

## Current model roles

- ASR: `mlx-community--Qwen3-ASR-0.6B-8bit` through oMLX, with the request language fixed to `ja`.
- Chat: `deepseek-flash` through the DeepSeek OpenAI-compatible endpoint.
- TTS: `mlx-community--Qwen3-TTS-12Hz-1.7B-Base-8bit`, using the local Natsume fine-tuned speaker when configured.
- Screen vision: `mlx-community--Qwen3.6-35B-A3B-4bit` through oMLX.

## Present limitations

- `output_language` exists in configuration but is not yet a runtime switch; Japanese output currently comes from the LLM system prompt.
- The ASR language is now set to `ja`, which guides decoding toward Japanese but does not remove the model's multilingual capability.
- Screen capture currently assumes `mss.monitors[1]`; on a multi-monitor Mac this can capture a different display from the frontmost app.
- The proactive planner is a prototype. It has no durable event inbox, memory store, or robust attention policy yet.
- Voice input is captured by the host service. There is no browser microphone path in this prototype.

## Package boundaries

`packages/maid_voice` owns the low-latency voice loop and provider adapters. `packages/maid_agent` adds the Maid identity, screen perception, and proactive decision layer while reusing that voice loop. This boundary is deliberately small so the packages can later be moved behind a larger event and memory service.
