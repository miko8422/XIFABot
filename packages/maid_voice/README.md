# Maid Voice Cascade — v0.1 (Phases 0–2)

Cascaded voice assistant, push-to-talk or hands-free: **Qwen3-ASR → DeepSeek V4.1 Flash → Qwen3-TTS (fine-tuned ナツメ voice)**, streaming at every stage, with a per-turn latency trace.
Goal and design: `../../docs/maid_voice_cascade_GOAL.md`, `../../docs/maid_voice_cascade_ARCHITECTURE.md`.

## Run

Requirements: macOS on Apple Silicon, [uv](https://docs.astral.sh/uv/), oMLX running on `127.0.0.1:9800`, and `../.env` containing `BASE_URL` and `API_KEY` (DeepSeek).

```sh
# one-time setup: models (oMLX picks them up from the HF cache) and the reference voice
hf download mlx-community/Qwen3-ASR-0.6B-8bit
hf download mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit
curl -X POST http://127.0.0.1:9800/admin/api/reload
uv run scripts/build_reference.py a 01 04 05 06 09     # -> voices/natsume/ref_a.{wav,txt}

# fine-tuned voice (default profile natsume_ft): see sft/README.md; the 8-bit MLX model lives in oMLX's
# model dir as natsume-qwen3tts-sft-e1-8bit, source checkpoint in runs/sft/ckpt/epoch-1
uv run maid-voice serve      # http://127.0.0.1:8765 — hold Space to talk, or switch to hands-free; Esc stops her
uv run maid-voice --voice natsume_a bench   # any command can pick a voice profile
uv run maid-voice bench      # replays runs/asr_fixtures/*.wav through the full pipeline + speakers
uv run maid-voice bench --handsfree [--no-speculative]   # same, but streamed through the VAD endpointer
uv run pytest                # controller / cancellation / stale-packet / failure-recovery tests
```

On first launch, macOS asks for microphone permission for the terminal app.
Every turn is appended to `runs/<timestamp>-<mode>/turns.jsonl` and `summary.csv`. Raw audio is never saved.

## Layout

| Module | Role |
| --- | --- |
| `config.py` | `config/default.toml` (settings) + `.env` (secrets) |
| `audio.py` | `Capture` (always-open mic, energy-based speech end), `PlaybackQueue` (generation-tagged, drops stale chunks), `Player` (reports estimated DAC time of first sample and halt) |
| `asr.py` / `llm.py` / `tts.py` | Adapter protocols + oMLX ASR, OpenAI-compatible streaming LLM, oMLX streaming clone TTS |
| `vad.py`, `handsfree.py` | Energy endpointer (adaptive noise floor) emitting start / speculative / resume / end; hands-free turn runtime with interim transcripts |
| `segmenter.py` | LLM deltas → Japanese clauses (early first clause, sentence ends after that) |
| `controller.py` | Turn state, generation id, cancellation, metrics. Works with fake adapters in tests |
| `metrics.py` | Monotonic per-turn trace → JSONL + CSV |
| `app.py`, `static/index.html` | `serve` (web UI) and `bench` commands |

## Measured results (M5 Pro, 64 GB, macOS 26.5.2, warm, 2026-10-25)

Headline metric: **end of user speech → first audible sample** (estimated time the first sample reaches the speaker DAC, from PortAudio).

### End to end (`maid-voice bench --runs 3`, 6 fixtures × 3, MacBook speakers)

| ms | p50 | p90 | p95 | min | max |
| --- | --- | --- | --- | --- | --- |
| **first sound** | **1249** | **1366** | **1467** | 876 | 1696 |
| ASR (whole utterance) | 167 | 209 | 211 | 133 | 237 |
| LLM first token | 692 | 805 | 986 | 433 | 1059 |
| LLM first clause | 862 | 988 | 1066 | 560 | 1218 |
| TTS first audio | 211 | 224 | 233 | 159 | 235 |
| queued → audible | 13 | 18 | 18 | 9 | 18 |

This meets the prototype gate (median ≤ 1.5 s, p95 ≤ 3 s). The ~1.0 s stretch goal is not met yet: **the LLM accounts for about 70 %** of the time.
The bench feeds pre-recorded speech, so the endpointing delay is zero. In live push-to-talk, the time between your last word and releasing the key comes on top; it is shown separately as `release_to_sound_ms`.

### Phase 0 component probes (`scripts/probe_*.py`)

| Component | Finding |
| --- | --- |
| DeepSeek `deepseek-flash` (= V4.1 Flash) | `thinking` **must be disabled**: in default mode 2 of 5 turns used the whole 120-token budget on reasoning and produced no speech. Non-thinking: TTFT p50 530 ms (short prompt, n=5) |
| Qwen3-TTS 1.7B Base 8-bit, oMLX `stream=true` | **Truly streams** (model-native, 28–64 chunks per sentence). First audio p50 206 ms vs 1.6–2.1 s without streaming. Generates at 0.6–0.7× real time |
| Qwen3-TTS 0.6B Base 8-bit | First audio p50 170 ms (−35 ms), 0.5× real time. **Not used by default**: 1.7B voice preferred |
| Qwen3-ASR 0.6B 8-bit, oMLX | 100–160 ms for a 2 s utterance, about 220–320 ms for 4 s. ja/en transcribed exactly; names can be wrong ("Kissa Stella" → "Kiss Estela") |
| Audio output (PortAudio, built-in speakers) | ~10–18 ms from queue to DAC |

## Phase 2: hands-free with a speculative start (2026-10-25)

When there has been `spec_silence_ms` (250) of silence, ASR, the LLM and TTS start on the utterance so far, but playback stays **held**.
When the silence reaches `end_silence_ms` (600), the turn is committed and the held audio plays. If you start speaking again first,
that generation is cancelled and none of it is ever heard (tested in `tests/test_phase2.py`). Interim transcripts come from
re-transcribing the growing buffer about every 400 ms, are shown only, and never trigger any action.

`maid-voice bench --handsfree --runs 2` (fixtures streamed in real time, 1.2 s trailing noise, MacBook speakers):

| ms, end of speech → first sound | p50 | p90 | min | max |
| --- | --- | --- | --- | --- |
| **speculative start** (default) | **1167** | **1404** | 871 | 1913 |
| no speculation | 1523 | 1603 | 1251 | 1656 |

Both include the 600 ms end-of-turn wait. With speculation, p50 work left after the endpoint is 659 ms (vs 922 ms without).
4 of 16 speculative starts were discarded because the speaker continued after a pause.

## Voice experiments (2026-10-25)

All runs use the 14 test sentences in `scripts/tts_eval.py` and are warm. CER compares kana readings from a Qwen3-ASR round trip.
Speaker similarity is WavLM-SV cosine to a centroid of 24 real ナツメ lines; the baseline of real line against real centroid is **0.901**,
and the same config varies by about ±0.02 between runs, so this metric cannot separate the candidates.

| config | first audio p50 ms | RTF | CER | spk | note |
| --- | --- | --- | --- | --- | --- |
| Qwen3-TTS 0.6B 8-bit | 158 | 0.51 | 0.019 | 0.93 | |
| **Qwen3-TTS 1.7B 8-bit** (default) | 168–216 | 0.64–0.72 | 0.007 | 0.89–0.94 | |
| Qwen3-TTS 1.7B bf16 | 377 | **2.75** | 0.007 | 0.92 | slower than real time → stutters live |
| 1.7B 8-bit, 0.5 s / 1.0 s chunks | 371 / 667 | 0.64 | 0.007 | 0.92 | no measurable quality change |
| 1.7B 8-bit, non-streaming | 2468 | 0.67 | 0.010 | 0.91 | |
| VoxCPM2 2B 8-bit (mlx-audio 0.6.1) | 4828 | **1.28** | 0.076 | 0.91 | no streaming, worse pronunciation |
| Irodori v4.1 Small (mlx-audio 0.6.1) | 2850 | 0.73 | 0.026 | 0.93 | streaming not implemented |
| Fish S2 Pro 8-bit, ~5B (mlx-audio 0.6.1) | 1047 | **4.07** | 0.022* | 0.94 | streams, but 2–8× slower than real time; *errors are spelling (スチーム/Steam) + interjections |

The reference choice (5 joined system lines vs single game lines from `../nat`) made no measurable difference to CER or speaker similarity.
oMLX's bundled mlx-audio 0.4.3 cannot load Irodori v4.1 or VoxCPM2, so `scripts/tts_direct.py` runs them with mlx-audio 0.6.1.
Fish needs a quantization-predicate override to load the mlx-community 8-bit checkpoint (in `tts_direct.py`).

### The "dropped mic" artifact (e.g. 「そうい」 in 「そういう生き方」)

Cause: **codec-token sampling**. Each Qwen3-TTS token is 80 ms of audio. At the default temperature (0.9), the model sometimes samples
a token that doesn't fit its neighbours in a smooth vowel glide, and that 80 ms decodes as a broadband burst. Evidence: 42–52 % of
discontinuities in default-setting takes start on an 80 ms token boundary (chance ≈ 5 %), and spectrogram stripes appear in about half
of the default takes vs 0 / 12 at T = 0.3 / 0.1. Ruled out: clipping (none), streaming decode (identical real-audio tokens decoded whole and
streamed give the same discontinuity counts, `scripts/decode_proof.py`), 8-bit quantization (bf16 also glitches), mispronunciation.
Fix: `[tts] temperature = 0.4` (latency, CER and speaker similarity unchanged). The automatic spike counter is too noisy to confirm this
across the whole 14-sentence set, so the A/B in `runs/artifact/ab/` is the real check.

A blind A/B pack is in `runs/listening/` (key in `KEY_dont_peek.txt`).

## Fine-tuned voice (2026-10-25)

The 1.7B Base model was fine-tuned on 3.94 h of her lines (3037 cleaned lines) into a CustomVoice model with speaker `natsume`
(`sft/`, one Vast.ai A100 40 GB, about 1.3 h and $1). The official Qwen script has alignment bugs, documented in `sft/README.md`.
Held-out loss: 2.888 (untouched) → 2.423 (epoch 0) → 2.429 (epoch 1), then overfitting (3.001 at epoch 5). By ear, **epoch 1** is
the best of epochs 0/1/3 and the clone, and it is now the default voice (`natsume_ft`). It speaks with the game lines' pacing (longer pauses at commas).

`maid-voice bench --runs 2`, run back to back on the same day, MacBook speakers:

| ms, p50 / p90 | first sound | TTS first audio | LLM first token |
| --- | --- | --- | --- |
| **fine-tuned (natsume_ft)** | **940 / 1097** | **95** / 101 | 584 / 663 |
| clone (natsume_a) | 956 / 1107 | 155 / 157 | 535 / 651 |

The TTS stage is 60 ms faster because no reference audio is encoded per request. Overall first sound is a tie here, because the LLM ran slower during the fine-tuned run.

## Known limitations / engineering decisions

- **ASR is not incremental.** oMLX transcription takes an uploaded file, and `stream=true` only streams the *decoding*. It is fast enough to re-transcribe a growing buffer for interim results (planned for Phase 2).
- **No echo cancellation yet.** PortAudio does not use Apple's voice-processing I/O. Use headphones. Push-to-talk avoids the problem for now.
- **Endpointing is energy-based** (adaptive noise floor), not a neural VAD or a semantic turn detector. Long thinking pauses longer than `end_silence_ms` will end the turn.
- **Hands-free needs headphones.** There is no echo cancellation, so while she is speaking new speech is ignored unless voice barge-in is switched on (headphones only).
- **Clause joins** use an 8 ms fade-in/fade-out per clause (adds about 8 ms of latency) so joins can't click.
- **Clone profiles re-send the reference voice with every TTS request** (oMLX has no speaker-conditioning cache). The fine-tuned profile does not.
- **Clauses are synthesized one after another, as independent TTS calls.** Prosody across clause boundaries is not tuned yet.
- **Cost is `null`.** DeepSeek prices are not configured (`price_in_per_mtok`/`price_out_per_mtok` in the `[llm]` section). Token usage is recorded.
- **LLM provider comparison** (same 5 prompts, warm, TTFT p50): DeepSeek V4.1 Flash (non-thinking) 530 ms; GPT-6 Luna (`reasoning_effort=none`) 1594 ms; local Qwen3.6-35B-A3B 4-bit on oMLX 568 ms (19 s cold load, and it shares the GPU with TTS). DeepSeek stays the default.
- Echo cancellation and robust voice barge-in (Phase 3) and the scripted 20–30-turn evaluation (Phase 4) are still to do.
- The reference clips are © YUZUSOFT / JUNOS, for personal use only. `voices/` is git-ignored, and the synthetic voice must not be distributed or presented as genuine.
