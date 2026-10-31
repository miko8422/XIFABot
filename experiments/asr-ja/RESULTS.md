# Japanese ASR experiment results

Completed 2026-10-27 on this machine: Apple M5 Pro, macOS 26.5.2.

**Recommendation: keep the current Qwen3-ASR-0.6B for now.** None of the tested replacements clearly improves recognition while maintaining the current speed. Qwen3-ASR-1.7B is the strongest accuracy candidate if another roughly 140 ms on short dialogue is acceptable. No application code, configuration or existing Python environment was changed, and no fine-tuning was performed.

## Main comparison

Accuracy uses the same 100 clean recordings: 60 Japanese FLEURS test clips and 40 existing character-dialogue clips. CER is character error rate; lower is better. It counts spelling differences as well as recognition mistakes. Kana CER reduces some kanji/kana spelling effects and is a secondary diagnostic.

Latency is warm, complete-utterance recognition, measured on 30 fixed clips repeated three times. It excludes model loading and audio-file reading. These are ASR times, not total voice-chat response times.

| Model / runtime | Clean CER | Kana CER | Median ASR | 95th percentile |
|---|---:|---:|---:|---:|
| **Current 0.6B, oMLX auto language** | **8.67%** | **4.34%** | **268 ms** | **354 ms** |
| Same endpoint, Japanese specified | 8.64% | 4.31% | 247 ms | 309 ms |
| Same 0.6B weights, direct MLX | 8.79% | 4.53% | 212 ms | 286 ms |
| **Qwen3-ASR-1.7B, 8-bit MLX** | **6.44%** | 3.17% | 420 ms | 538 ms |
| Kotoba-Whisper v2.0, MLX | 6.71% | **3.05%** | 414 ms | 469 ms |
| ReazonSpeech k2-v2, CPU int8-fp32 | 9.45% | 7.27% | **113 ms** | **144 ms** |

For the 15 shorter dialogue probes (2–6 seconds, median duration 3.87 seconds), the median ASR times were **231 ms current**, 209 ms with Japanese specified, 189 ms direct 0.6B, **371 ms Qwen 1.7B**, **398 ms Kotoba**, and **81 ms ReazonSpeech**. Each clip had three repetitions. The other 15 latency probes are longer FLEURS clips, 6–8 seconds.

## What the tradeoffs mean

- **Qwen 1.7B:** about 26% lower clean CER relative to the current model, but about 57% longer pooled median recognition time. It improved 38 clips and worsened 9; 53 were tied. Its short-dialogue median adds 140 ms. This is the first candidate to try if accuracy matters more than that delay.
- **Kotoba:** about 23% lower clean CER and the best kana score, but also slower. It improved 39 clips and worsened 18. It does not establish a clear advantage over Qwen 1.7B: their accuracy scores are close, and their relative speed depends on clip length.
- **ReazonSpeech:** substantially faster, but did not demonstrate an accuracy gain. Its observed clean CER and kana CER were worse. The primary CER difference is uncertain on this small sample, so this is not evidence that it is universally less accurate.
- **Japanese specified on the current endpoint:** almost identical recognition (97 of 100 clips tied by edit count), with modestly lower latency in this run. This is a small possible optimization, not an accuracy solution. It was used only on experiment requests.
- **Direct 0.6B runtime:** similar recognition with lower measured latency. That is a runtime comparison, not a better model. The live API measurement includes WAV encoding and localhost transport; direct MLX uses the experiment's library version and decoding settings. The difference cannot be attributed entirely to HTTP overhead.

Paired bootstrap intervals over the 100 clean clips support the accuracy improvements for Qwen 1.7B (CER change −2.24 percentage points, 95% interval −3.62 to −0.91) and Kotoba (−1.97 points, −3.43 to −0.57). ReazonSpeech's +0.78-point change has an interval of −1.71 to +3.58. These describe this selected corpus, not future microphone performance. Models ran sequentially in one session; small timing differences can reflect run order or machine load.

## Accuracy details and scope

| Model | FLEURS CER, 60 clips | Dialogue CER, 40 clips |
|---|---:|---:|
| Current 0.6B | 7.91% | 11.52% |
| Japanese specified | 7.87% | 11.52% |
| Direct 0.6B | 8.14% | 11.24% |
| Qwen 1.7B | 5.45% | 10.11% |
| Kotoba | 5.68% | 10.53% |
| ReazonSpeech | 8.85% | 11.66% |

The combined score is weighted by reference characters, so longer FLEURS sentences contribute more. The improvement on character dialogue is smaller than the combined result suggests. Manual transcript inspection found both real recognition mistakes and harmless spelling differences: for example, `色々` versus `いろいろ`, `良かった` versus `よかった`, and `２つ` versus `二つ`. Proper names such as `高嶺君` remain troublesome across models. The dialogue references are existing script labels; they are not newly verified verbatim microphone transcripts. Kana conversion also has ambiguous readings and is not phonetic ground truth.

Twenty paired clips with synthetic 15 dB white noise were scored separately. Qwen 1.7B had the lowest noisy FLEURS CER; Qwen 1.7B and Kotoba tied on noisy dialogue. This small synthetic check does not establish performance with real room noise. Two synthetic speech fixtures and two no-speech controls were also separate from the main score. On the silence/quiet-noise controls, current automatic-language 0.6B returned empty text; the other configurations emitted text or punctuation. These controls should not be treated as a no-speech error-rate estimate.

All six configurations completed **184 measured calls each, 1,104 total, with zero request/runtime failures**, plus three warmups each. There are 124 unique input clips including the separate diagnostic groups.

## Decision before fine-tuning

Keep 0.6B as the speed baseline. If an extra approximately 140 ms is acceptable, Qwen 1.7B is a reasonable accuracy trial. Otherwise, the next useful evidence is a small set of actual microphone recordings that the current system gets wrong, with corrected Japanese transcripts. Test those against this same harness before deciding on 0.6B fine-tuning; this experiment does not show whether the user's failures come from the model, speech cutoff or recording conditions.

## Evidence and reproduction

- [Setup, model configurations and primary sources](README.md)
- [Full aggregate scores, noise checks and controls](results/summary.json)
- [Paired clean-clip references, hypotheses and errors](runs/main/per_clip.csv)
- [Frozen local input manifest](data/manifest.jsonl), [selection](results/selection.json), and [model/data revisions](results/assets.json)
- Per-model `.meta.json` files in `results/` record package versions, configuration and script/manifest hashes. Raw per-call transcripts/timings are in `runs/main/`. Raw runs, audio and model downloads stay local and ignored.
