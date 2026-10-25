# Maid Voice Cascade — GOAL.md

Status: implementation brief for v0.1 proof of concept
Date: 2026-10-24
Owner: Alan
Audience: Codex implementation agent
Related design: maid_voice_cascade_ARCHITECTURE.md

## 0. Mission

Build and test a **low-latency cascaded voice assistant** before building the larger Maid Agent. The experiment is intended to answer a single product question:

> Can a streaming speech-to-text → fast text LLM → cloned-voice streaming text-to-speech system feel nearly as responsive and interruptible as GPT Live, while giving us full control over the Japanese voice and a much lower operating cost?

Do not claim parity with end-to-end speech models in advance. Build an instrumented prototype, gather latency and conversational-quality evidence on Alan's hardware, and iterate.

## 1. Decisions already agreed

1. The Maid Agent will ultimately be a **separate application**, not a permanent Codex session. This prototype has no dependency on Manager Bot or Botmux.
2. User speech: **English and Japanese**, including mixed-language turns if practical. English-only is an acceptable temporary fallback while debugging.
3. Assistant speech: **Japanese by default**, even when the user speaks English. Keep the output language configurable.
4. Assistant voice: a **user-provided Japanese reference voice** using a zero-shot voice-cloning TTS model. Prioritize a recognizable, natural, consistent voice over English synthesis.
5. Interaction: cascaded STT / LLM / TTS with **concurrent capture, streaming, barge-in, and cancellation**. Do not require a continuous GPT Live session.
6. Model candidates for benchmarking: **GPT-6 Luna** and **DeepSeek V4.1 Flash**, in their lowest-latency supported modes. Treat access, model IDs, latency and prices as configuration to verify at build time.
7. This is a **voice-only milestone**: no Steam, Botmux, camera, screenshots, MCP orchestration, memory bank or proactive notification logic yet.

## 2. User experience: minimum demo

- Start the local app; select microphone, output device, LLM provider, and a reference-voice profile.
- First support push-to-talk for deterministic debugging; add hands-free VAD/turn detection before final acceptance.
- Speak: “Did my background task finish?” The app transcribes the speech, passes the text to the LLM with a Japanese-reply instruction, and speaks the answer in Japanese.
- While the assistant speaks, say “Wait, wait—another thing.” Playback should stop promptly, old audio should be discarded, and the new turn should be understood.
- Display a simple live trace: interim/final transcription, generated Japanese text, dialogue turns, interruption events, model selection, and per-stage latency.
- Have a repeatable 20–30-turn test script; allow recording and replay of timing metrics without needing to retain raw microphone audio.

## 3. Functional requirements

### R1 — Audio capture and endpointing

- Keep microphone capture independent of TTS playback.
- Use streaming voice activity detection (VAD), echo reduction where available, input buffering, and explicit turn states.
- Distinguish temporary pauses and backchannels from completed utterances; make endpointing thresholds adjustable.
- Handle device reconnect, silence, noise, and switching from English to Japanese.

### R2 — ASR

- Implement an adapter API supporting partial transcripts and final transcripts, with timestamps and detected language.
- First local candidate: Qwen3-ASR-0.6B; benchmark whether the **actual serving stack** delivers usable incremental hypotheses at low latency. Do not assume offline throughput implies low streaming latency.
- Keep a second ASR adapter available as a baseline if local streaming proves too slow or unstable.
- An interim ASR hypothesis must never be treated as an irreversible user command.

### R3 — LLM

- Provide interchangeable streaming LLM adapters. Benchmark GPT-6 Luna against DeepSeek V4.1 Flash using identical prompts and test turns.
- Use minimal conversational context, low reasoning effort/non-thinking mode where supported, a tight output-token budget, and a short Japanese first sentence.
- Stream tokens to a text segmenter; do not wait for the whole answer to finish before starting TTS.
- If generating speculatively from stable partial ASR, never speak or execute tools until the utterance is confirmed; invalidate speculative work on ASR revision.

### R4 — Japanese voice cloning and TTS

- First candidate: Qwen3-TTS-12Hz-0.6B-Base (the **Base** checkpoint, not CustomVoice), using reference audio and corresponding reference transcript.
- Support alternative TTS adapters and an ordinary Japanese fallback voice to isolate clone quality from pipeline delays.
- Measure *time to first playable audio*, not just average synthesis throughput. Verify that the selected wrapper really streams playable chunks instead of buffering a whole sentence.
- Cache reusable voice conditioning where supported. Segment by natural Japanese clauses while guarding prosody and audio joins.
- Use only samples that the user is entitled to use; keep the reference corpus and generated voice private, and do not present the synthetic voice as the actor's authentic speech.

### R5 — Interruptions and synchronization

- Keep listening while speaking. Detect deliberate barge-in; stop speaker output, cancel or invalidate pending LLM/TTS work, and process the new user turn.
- Prevent the assistant's own speaker audio from being transcribed as user speech, through headphones and/or acoustic echo cancellation.
- Tag every response with a turn ID / generation ID so late packets cannot play after interruption.
- Support explicit user stop, pause, mute, and shutdown.

### R6 — Observability and cost

- Instrument monotonic timestamps for mic speech start/end, VAD endpoint, ASR interim/final, LLM request/first token, TTS first audio packet, actual speaker playback, interruption detected, and playback halted.
- Write one structured record per turn to JSONL and a summary CSV; include model/provider IDs, token use, estimated API cost, device, and failure reason.
- Default to **not saving raw audio**. Add opt-in capture for debugging, with local deletion controls.
- Report p50/p90/p95 latency and qualitative ratings for responsiveness, voice similarity, Japanese naturalness, and interruptions.

## 4. Performance goals (targets, not guarantees)

- Main metric: **end of user speech → first audible Japanese response**.
- Stretch goal: median around **1.0 second** in a warm, stable session; reasonable initial prototype gate: median no more than **1.5 seconds**, p95 no more than **3 seconds** under defined test conditions.
- Interruption goal: measure **user barge-in onset → speaker output stopped**; target p90 no more than **350 ms** on the test device.
- No stale playback after a cancelled turn; no accidental tool execution from partial transcripts.
- Recover from temporary API/model errors without crashing the audio app.
- Track always-on idle resource use separately from inference use. Avoid competing with the RTX 5070 Ti when the desktop is being used for gaming; benchmark the M5 Pro Mac as an alternate host.

These targets are experimental. If they are not met, report where the time went and compare alternatives; do not fake or silently relax results.

## 5. Controlled comparison

Test the same fixed utterances under these conditions:

| Dimension | Initial variants |
| --- | --- |
| Input speech | English, Japanese, mixed/borrowed words, self-correction, long pauses |
| ASR | Local Qwen3-ASR-0.6B; one validated streaming baseline |
| LLM | GPT-6 Luna; DeepSeek V4.1 Flash |
| TTS | Qwen3-TTS 0.6B Base clone; Japanese non-cloned fallback |
| Endpointing | Conservative versus aggressive silence thresholds |
| Execution host | M5 Pro; optional RTX 5070 Ti for reference |
| Playback | Normal answer, interrupted answer, noisy room, headphones/speakers |

Use at least 20–30 scripted interactions and 5 deliberate interruptions. Distinguish cold-start from warm-session metrics; retain prompts and configurations for reproducibility. Track quality as well as latency, because an early but wrong reply is not a success.

## 6. Implementation phases

### Phase 0 — Preflight

- Inspect the target host and installed drivers. Ask for API keys, model paths, and voice reference samples only when needed; never hard-code secrets.
- Validate model access, exact model names, supported low-latency options, audio sample rates, Japanese clone quality and streaming behavior.
- Produce a reproducible configuration file with placeholders and a hardware/compatibility note.

### Phase 1 — Thin end-to-end happy path

- Implement microphone → ASR → streaming LLM → Japanese TTS → speaker with a short text/debug display.
- Start with push-to-talk and one LLM, one ASR, and one TTS. Demonstrate the first real conversation.

### Phase 2 — Streaming overlap

- Add interim ASR display, endpointing, token-to-clause TTS segmentation, audio chunk playback, prewarming and bounded queues.
- Verify overlap on the timeline; avoid delaying first playback until full model completion.

### Phase 3 — Duplex-like interaction control

- Add continuous capture during playback, VAD-assisted interruption, cancellation and echo control.
- Harden state transitions and late-packet rejection.

### Phase 4 — Benchmarks and provider swaps

- Add the second LLM and ASR baseline, compare TTS voices, run test scenarios and collect cost/latency/quality metrics.
- Record a recommendation for the v0.2 audio stack, with known blockers and tuning knobs.

## 7. Deliverables required from Codex

1. A runnable project with documented environment setup and one-command local start.
2. A modular codebase with audio, endpointing, ASR, LLM, TTS, turn controller, playback, configuration, and metrics separated behind interfaces.
3. A minimal local UI or diagnostics console for live transcription, model/voice selection, and timeline.
4. Unit/integration tests for cancellation, stale-packet rejection, and stage-failure recovery.
5. Benchmark fixtures, metrics export, example configuration, and a README with **measured** results.
6. An explicit list of unsupported features and engineering decisions, especially any model wrapper that does not truly stream.

## 8. Acceptance checklist

- [ ] A user can speak English and receive a spoken Japanese answer in the reference voice.
- [ ] A user can speak Japanese and receive a spoken Japanese answer.
- [ ] Streaming audio plays before the complete answer is generated, where supported.
- [ ] Barge-in stops old playback and creates a clean new turn.
- [ ] Late packets from cancelled turns never reach playback.
- [ ] The demo shows and exports an accurate per-stage latency timeline.
- [ ] Both LLM candidates can be benchmarked through the same harness.
- [ ] The app can run repeatedly without Botmux, Maid Agent runtime, Steam, MCP or cameras.
- [ ] No always-connected GPT Live session is required.
- [ ] Benchmarks and known limitations are documented honestly.

## 9. Constraints and deliberate non-goals

Do not construct a large general-purpose agent framework, fine-tune a model, add autonomous tool use, build a mobile app, integrate a camera, implement persistent relationship memory, or imitate GPT Live's internal audio model. Minimize engineering surface until the listening/speaking loop is demonstrably good.

Use mature components when they meet the latency target. A short LiveKit Agents spike is allowed for turn detection and interruptions, but the ASR/LLM/TTS and turn-controller interfaces must remain independent so we can replace bottlenecks. Building every audio primitive from scratch is not a requirement.

## 10. First instruction to Codex

Read this file and maid_voice_cascade_ARCHITECTURE.md. Implement Phase 0 and Phase 1 only, then show the runnable demo, exact dependencies, observed stage timings and a proposed Phase 2 plan. Do not silently implement the full Maid Agent. Ask only for missing credentials, voice references, and device-specific choices that block execution.

## 11. Reference implementations to verify

- [LiveKit Agents voice sessions and turn-taking](https://docs.livekit.io/agents/logic/sessions/)
- [LiveKit interruption tuning](https://docs.livekit.io/agents/logic/turns/tuning/)
- [Qwen3-ASR official repository](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-TTS official repository](https://github.com/QwenLM/Qwen3-TTS)
- [OpenAI API model documentation](https://developers.openai.com/api/docs/models)
- [DeepSeek API documentation](https://api-docs.deepseek.com/)

These are candidate implementations, not benchmark results. Reverify API compatibility and pricing at implementation time.
