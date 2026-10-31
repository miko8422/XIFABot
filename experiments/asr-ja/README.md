# Japanese ASR comparison

An isolated experiment for improving XIFABot's Japanese recognition without sacrificing short-turn speed. The application, its configuration, its Python environments, and its oMLX model registration are unchanged.

**Completed comparison:** [results and recommendation](RESULTS.md). Aggregate scores and provenance are retained in `results/`; full local measurements are in `runs/main/`.

## Reproduce

From the repository root:

```sh
uv sync --project experiments/asr-ja --python 3.12
uv run --project experiments/asr-ja python experiments/asr-ja/prepare.py assets
uv run --project experiments/asr-ja python experiments/asr-ja/prepare.py data
```

Run one model at a time, to avoid GPU/CPU contention:

```sh
uv run --project experiments/asr-ja python experiments/asr-ja/bench.py omlx_auto
uv run --project experiments/asr-ja python experiments/asr-ja/bench.py omlx_ja
uv run --project experiments/asr-ja python experiments/asr-ja/bench.py qwen06
uv run --project experiments/asr-ja python experiments/asr-ja/bench.py qwen17
uv run --project experiments/asr-ja python experiments/asr-ja/bench.py reazon
uv run --project experiments/asr-ja python experiments/asr-ja/bench.py kotoba
uv run --project experiments/asr-ja python experiments/asr-ja/score.py
```

Use a new `--tag` for another run; the harness refuses to overwrite measurements. `--limit 3 --repeats 1 --tag smoke` is a quick integration check.

## Models

- `omlx_auto`: current production Qwen3-ASR-0.6B-8bit through the existing localhost API, unchanged automatic language detection.
- `omlx_ja`: same endpoint and weights, with `language=ja` on each experiment request. Does not modify server or app configuration.
- `qwen06`: same cached weights, direct MLX-Audio 0.6.1 with Japanese selected. Controls for differences between the production server and the experiment runtime.
- `qwen17`: Qwen3-ASR-1.7B-8bit, same direct MLX runtime and decoding settings as `qwen06`.
- `reazon`: Japanese ReazonSpeech k2-v2, official int8-fp32 layout (int8 encoder/joiner, fp32 decoder), Sherpa-ONNX CPU, four threads, greedy decoding, official 0.9-second padding on both ends.
- `kotoba`: Kotoba-Whisper v2.0 MLX, Japanese, temperature zero, independent utterances. No diarization or punctuation postprocessor.

Model revisions are frozen in `data/assets.json`; packages are pinned by `uv.lock`. The current 0.6B checkpoint is read from the existing Hugging Face cache. New model files and caches stay here.

## Evaluation

- 60 randomly selected Japanese FLEURS **test** recordings, balanced by male/female labels (30 each) and 6–12/12–20-second duration bins (30 each); unique sentence IDs. Within the shorter bin, 15 clips are 6–8 seconds. The test split has no recordings under 6 seconds.
- 40 randomly selected existing game dialogue recordings, balanced between 1–4 and 4–8 seconds. The existing text-cleaning filters are reused. These are a single character's performed speech, not the user's microphone recordings.
- 20 paired copies with synthetic 15 dB white noise, reported separately from clean recognition.
- Two existing synthetic Japanese fixtures and two no-speech controls, also separate.
- Selection seed: 20261027. No candidate transcript is used to select the clips. Each prepared WAV has a SHA-256 in the manifest.

Primary accuracy is character error rate after Unicode NFKC normalization and removing punctuation, symbols, separators and control characters. Lowercase Latin letters are normalized; Japanese spelling variants and numeral variants are not silently rewritten. Kana-reading CER is a secondary diagnostic using pykakasi, not a replacement for the primary score. Failed requests count as empty transcripts and remain in accuracy results. No-speech hallucinations are reported independently.

Timing excludes model loading and WAV reading, includes feature extraction and complete decoding, and synchronizes MLX before/after each call. Production API times also include WAV encoding and localhost transport. After three warmups, 30 fixed clips receive three timed repetitions each (15 dialogue clips of 2–6 seconds and 15 FLEURS clips of 6–8 seconds). Report p50 and p95. These are full-utterance ASR timings, not first-token latency and not end-to-end voice-response latency. The runtime control distinguishes model gains from server/library effects.

Raw transcripts, metadata, paired per-clip errors, aggregate scores and measurements are in `runs/<tag>/`. Raw runs, downloaded audio/models and generated data are git-ignored; the completed comparison's aggregate scores and provenance are saved in `results/`. This is a screening experiment, not evidence about the user's accent, room noise or endpointing failures. No ASR fine-tuning is performed.

## Sources

- [Qwen3-ASR models](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Qwen3-ASR-1.7B MLX conversion](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit)
- [ReazonSpeech k2-v2](https://huggingface.co/reazon-research/reazonspeech-k2-v2) and [official loading/transcription implementation](https://github.com/reazon-research/ReazonSpeech/tree/master/pkg/k2-asr/src)
- [Kotoba-Whisper v2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0) and [MLX conversion](https://huggingface.co/kaiinui/kotoba-whisper-v2.0-mlx)
- [Google FLEURS](https://huggingface.co/datasets/google/fleurs), Japanese test split, CC BY 4.0
