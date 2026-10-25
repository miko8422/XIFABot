# Maid Voice Cascade — Streaming Architecture

Status: v0.1 engineering design and tuning guide
Date: 2026-10-24
Companion: maid_voice_cascade_GOAL.md

## 1. Design thesis

**Cascaded** means that speech recognition, language reasoning, and speech synthesis are separate stages. It does **not** require strictly sequential, half-duplex behavior. With independent microphone capture, overlapping streaming stages, a turn controller and cancellable playback, the user can interrupt and respond naturally without relying on a single end-to-end audio model.

What it cannot reproduce perfectly: subword acoustic cues, simultaneous listening-and-vocalizing learned inside an audio-native model, ultra-natural overlap, and fine-grained nonverbal behavior. Those are accepted limitations for v0.1. The evaluation is experiential and measured rather than a promise of GPT Live equivalence.

## 2. Proposed topology

~~~mermaid
flowchart TD
    MIC["Microphone / Capture"] --> FE["Audio Front End: echo control + VAD"]
    FE --> ASR["Streaming ASR Adapter"]
    ASR --> TC["Turn Controller"]
    TC --> LM["Streaming Text LLM"]
    LM --> SEG["Japanese Clause Segmenter"]
    SEG --> TTS["Voice-Clone TTS Adapter"]
    TTS --> PLAY["Cancellable Playback"]
    PLAY --> SPK["Output Device"]
    FE --> TC
    TC --> PLAY
    TC --> TTS
    TC --> LM
    TC --> OBS["Trace / Metrics"]
    ASR --> OBS
    LM --> OBS
    TTS --> OBS
    PLAY --> OBS
~~~

All model components are adapters rather than framework-specific classes. The turn controller has sole authority to commit utterances, cancel response generations and decide which audio chunks may play.

## 3. Processes and concurrency

Recommended v0.1 layout (one application, several asynchronous tasks):

| Task | Responsibilities | Must not block on |
| --- | --- | --- |
| Capture task | Audio frames, device status, echo-control metadata | LLM or TTS |
| Endpoint task | VAD transitions, pause/turn-end prediction, barge-in | LLM output |
| ASR task | Partial/final transcripts, language tags | TTS playback |
| Turn-controller task | State, active generation ID, cancellation, backpressure | Slow provider calls |
| LLM task | Stream Japanese response tokens, emit chunks | Complete reply |
| TTS task | Consume validated clauses, emit playable PCM chunks | Complete reply |
| Playback task | Buffered output, immediate cancel/flush | Further TTS chunks |
| Metrics task | Append structured events asynchronously | Audio real-time path |

Use bounded queues between each stage. Apply dropping/coalescing to low-value interim ASR hypotheses, never to committed audio or state transitions. Audio capture should not back up behind network inference.

## 4. Minimal canonical contracts

The names and fields below are illustrative and can be implemented as typed Python dataclasses / TypeScript interfaces.

### Inbound speech event

~~~json
{
  "event": "asr.partial",
  "session_id": "session-id",
  "utterance_id": "utterance-id",
  "revision": 5,
  "text": "Did my task",
  "language": "en",
  "stable_prefix_chars": 8,
  "is_final": false,
  "t_monotonic_ns": 0
}
~~~

Final ASR uses event = asr.final, is_final = true, and the authoritative final transcript. Stable prefixes are hints only; final transcripts govern committed dialog state.

### Generation request

~~~json
{
  "session_id": "session-id",
  "utterance_id": "utterance-id",
  "generation_id": 42,
  "committed_user_text": "Did my task finish?",
  "input_language": "en",
  "output_language": "ja",
  "voice_profile": "user-provided-profile",
  "stream": true,
  "cancel_token": "cancellation-handle"
}
~~~

All downstream LLM, TTS and playback packets carry generation_id. On cancellation, increment the active generation ID and ignore all packets from older IDs—even if upstream cancellation cannot halt a provider immediately.

### Common metrics record

~~~json
{
  "utterance_id": "utterance-id",
  "generation_id": 42,
  "t_speech_end_ns": 0,
  "t_vad_endpoint_ns": 0,
  "t_asr_final_ns": 0,
  "t_llm_first_token_ns": 0,
  "t_tts_first_audio_ns": 0,
  "t_playback_start_ns": 0,
  "llm_model": "selected-model",
  "asr_model": "selected-model",
  "tts_model": "selected-model",
  "first_sound_latency_ms": null,
  "cancelled": false,
  "error": null
}
~~~

Use a monotonic clock for differences; add UTC wall time separately for logs. Never compute latency from unrelated provider timestamps that may have clock skew.

## 5. Turn state machine

~~~mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Listening: microphone enabled
    Listening --> Capturing: speech starts
    Capturing --> Committing: turn end detected
    Committing --> Responding: final transcript accepted
    Responding --> Listening: playback completed
    Responding --> Interrupted: user barge-in
    Interrupted --> Capturing: cancel and start new utterance
    Capturing --> Listening: false start / silence
    Committing --> Listening: empty transcript
    Responding --> Listening: model failure handled
~~~

Key invariants:

1. Only one **committed** assistant response owns playback at a time.
2. Microphone capture remains active in Responding.
3. A valid barge-in cancels old playback before opening a new committed response.
4. An interim ASR edit can invalidate speculation; it cannot execute an irreversible action.
5. Repeated VAD events, duplicate endpoints, reconnects and late network responses must be idempotent.

## 6. End-of-turn detection and barge-in

- Begin with VAD + configurable silence duration. Tune on natural English pauses, Japanese phrase endings, corrections and deliberate hesitation.
- If available, use a turn-completion model as an optional second signal; do not equate a 200 ms silence gap with semantic completion in every situation.
- During TTS, start a potential-interruption candidate on near-end speech. Apply echo-cancellation, a minimum energy/duration rule, and optionally a small semantic detector to separate “mm-hm” from “stop, wait.”
- At confirmed barge-in: mute/flush playback **immediately**, cancel TTS and LLM, increment generation ID, and preserve user mic frames that arrived during cancellation.
- For the first demo, headphones are an acceptable engineering constraint. Benchmark speaker mode separately because the echo path changes the difficulty.
- Avoid a globally uninterruptible state except for explicit user-controlled alerts.

## 7. Latency and streaming strategy

The correct user-facing latency metric is:

**actual first played speaker sample − actual end of user's spoken turn**.

Break it down into:

| Stage | Measurement |
| --- | --- |
| Endpointer | speech end → committed turn end |
| ASR | speech end → usable final transcript |
| LLM | request dispatch → first useful Japanese token |
| Segmenter | first useful token → first speakable Japanese clause |
| TTS | first clause accepted → first playable PCM chunk |
| Playback | chunk ready → sample actually played |

These stages can overlap; simply adding their independently measured means will not equal the end-to-end latency.

Priority tuning order:

1. **Avoid cold starts:** prewarm selected STT/TTS model once, keep HTTP connections alive, cache reference-voice conditioning if supported.
2. **Tune endpointing:** overly conservative silence detection is often a larger delay than model inference; avoid clipping hesitant speech.
3. **Stream Japanese clauses:** flush at punctuation or short semantic boundaries, with a timeout/length escape hatch. Do not speak raw subword fragments.
4. **Overlap playback and generation:** play the first stable TTS chunk while LLM continues streaming later clauses.
5. **Reduce TTS startup:** batch versus streaming model wrappers must be A/B tested; a model labeled streaming may still have a buffering server.
6. **Minimize context work:** cache a fixed Japanese persona prompt and keep only the recent relevant dialogue in the voice benchmark.
7. **Speculative LLM prefill (later):** launch on stable ASR partials only after the baseline works; never publish speculative speech before final-turn confirmation.

Warm-start 1-second first sound is a stretch target. Record observed distributions before engineering around any assumed per-stage budget.

## 8. Suggested provider interfaces

### ASRAdapter

- open_stream(sample_rate, channels, language_hint)
- push_audio(pcm_frame, capture_timestamp)
- events(): AsyncIterator[PartialTranscript | FinalTranscript | ASRError]
- close_stream()

### LLMAdapter

- stream_response(turn_context, japanese_system_prompt, generation_id, cancellation)
- returns text deltas, usage and stop/error events
- exposes capability flags such as supports_reasoning_off and reports_usage

### TTSAdapter

- prepare_voice(ref_audio, ref_text, language = ja)
- stream_audio(text_clause, voice_profile, generation_id, cancellation)
- returns playable PCM frames with sample-rate/channel metadata
- exposes warmup, speaker-conditioning-cache and genuine-streaming capability flags

### PlaybackEngine

- enqueue(pcm_frame, generation_id)
- flush(generation_id)
- stop_now()
- report actual playback onset and stop acknowledgement

Adapters must not read provider secrets from source code or logs. The controller should work with simulated models for unit tests.

## 9. Default stack and alternatives

| Layer | Initial candidate | Important qualification |
| --- | --- | --- |
| Speech recognition | Qwen3-ASR-0.6B, local | Japanese/English support is documented; incremental serving performance must be tested on target hardware |
| Endpointing | Silero VAD or equivalent; optional LiveKit turn detector | Endpoint responsiveness and false cutoffs require tuning |
| Voice framework | Short LiveKit Agents baseline | Useful for session/interrupt handling; keep a path to a lean custom loop if measured overhead matters |
| Text LLM A | GPT-6 Luna | Confirm exact endpoint and lowest-latency supported reasoning controls |
| Text LLM B | DeepSeek V4.1 Flash | Confirm provider model ID, mode, region and current per-token rates |
| Voice synthesis | Qwen3-TTS-12Hz-0.6B-Base | Base supports reference voice cloning; verify Japanese pronunciation and streaming first-packet time |
| Local fallback | Simple Japanese TTS preset voice | Debug orchestration without clone latency |
| Logs | Structured JSONL + CSV summary | Keep audio/transcripts under explicit user controls |

Architecture must be provider-agnostic; model names and capabilities belong in configuration, not controller code.

## 10. Deployment options

**Option A — MacBook Pro M5 Pro, 64 GB:** candidate for always-on microphone capture, endpointing and local speech models while the user games on the Windows machine. Validate MLX/runtime compatibility, thermal load, memory pressure and round-trip audio device latency.

**Option B — Windows Ryzen 7700X / RTX 5070 Ti, 16 GB VRAM:** GPU acceleration candidate for local TTS/ASR, but must not degrade games. Consider local network transport overhead if microphone/speaker are on a different host.

**Option C — hybrid:** local capture + STT/TTS on whichever host wins the benchmark, text LLM via an API. Avoid repeatedly moving large PCM buffers over high-latency links; send framed audio with sequence numbers and bounded jitter buffers.

Benchmark latency and reliability on actual host configurations. Never assume theoretical model speed equals playback speed.

## 11. Diagnostics and test strategy

### Unit tests

- ASR interim revision does not cause duplicated user turns.
- Two end-of-turn events result in only one generation.
- Cancelled generation's late text/audio packets are discarded.
- Slow TTS queue cannot block mic capture.
- Barge-in is not triggered by the system's own output under headphone test conditions.
- Provider timeouts and restarts leave the controller in Listening.

### End-to-end scenarios

1. Simple English question, short Japanese answer.
2. Japanese question, Japanese answer.
3. Mid-sentence switch between English and Japanese.
4. Pause for one second, then self-correct without the assistant jumping in.
5. Interrupt a long Japanese answer with “Wait, wait.”
6. Speak a short backchannel while the maid is speaking; verify appropriate handling.
7. Wi-Fi delay, throttled LLM, unavailable TTS, cold-start ASR, input/output device change.
8. 30-minute continuous session: repeated turns, memory growth, reconnection and idle CPU use.

Keep model-quality assessment separate from system latency; label any errors as ASR, endpointing, Japanese LLM output, TTS/voice similarity, playback, or cancellation.

## 12. Compare a library-based versus custom turn runtime

Implement the smallest viable LiveKit Agents prototype (or another mature equivalent), then benchmark its timing/control behavior against the requirements. Adopt it if it meets the targets and permits external cloned-voice TTS, custom STT and Japanese LLM adapters. If it cannot provide the required streaming or interruption controls, implement only the missing coordinator/audio path—not a giant framework rewrite.

No architectural commitment should be justified by model marketing claims; use the latency trace and conversational test recordings.

## 13. Open decisions after first prototype

- Can the selected Qwen3-TTS Base serving stack yield high-quality **incremental** audio in Japanese with Alan's reference sample?
- Is local ASR on the M5 Pro responsive enough during gaming/work, or does a cloud/remote recognizer win overall?
- Are one-sentence chunks too robotic compared with clause chunks and overlap?
- Which provider has the best *first useful Japanese token* latency at acceptable quality?
- Does LiveKit materially add latency or improve endpoint accuracy enough to justify it?
- Would on-demand GPT Live fallback help rare high-overlap conversational situations after cost profiling?

## 14. Primary references (verify versions before coding)

- [LiveKit Agents voice session API](https://docs.livekit.io/reference/python/livekit/agents/voice/)
- [LiveKit turn-taking tuning](https://docs.livekit.io/agents/logic/turns/tuning/)
- [Qwen3-ASR code and supported languages](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-TTS cloning and streaming](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS technical paper](https://arxiv.org/abs/2601.15621)
- [OpenAI API pricing/models](https://developers.openai.com/api/docs/pricing)
- [DeepSeek official API documentation](https://api-docs.deepseek.com/quick_start/pricing/)
