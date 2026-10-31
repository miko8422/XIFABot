# ナツメ voice fine-tune (Qwen3-TTS 12Hz 1.7B Base → CustomVoice)

Goal: remove the random "dropped mic" bursts by training the model on her voice instead of cloning it from a 7 s reference.

## Pipeline

| step | where | command |
| --- | --- | --- |
| data | Mac | `uv run scripts/sft_prep.py` → `runs/sft/data/` (3037 lines, 3.94 h, 620 MB; rejects in `dropped.tsv`) |
| upload | Mac → GPU | `rsync -a sft/ runs/sft/data <host>:work/` (data lands in `work/data`) |
| setup, codec tokens, train, eval | GPU | `bash run.sh all` (resumable; stages: `setup`, `codes`, `train`, `eval`) |
| pick a checkpoint | Mac | download `evalwav.tgz` only (≈25 MB), score CER + listen, then fetch one `ckpt/epoch-N` (≈4.5 GB) |
| deploy | Mac | `mlx_audio.convert` to 8-bit, load in oMLX, `voice="natsume"`, language japanese |

Default run: full fine-tune, fp32 master weights + bf16 autocast, AdamW lr 1e-5, warmup 30, cosine, effective batch 32 (16 × 2), 6 epochs ≈ 555 steps, checkpoint + held-out loss every epoch. Needs ≈ 40 GB VRAM (A100/H100 80 GB; on a 48 GB card use `--batch-size 8 --grad-accum 4`; codec dir is symlinked into each checkpoint (use `rsync -L` when downloading)).

## Why not the upstream `sft_12hz.py` (commit 022e286)

Measured on the untouched 0.6B Base model (held-out lines; lower = the model already understands the layout):

| issue in upstream | upstream loss | fixed |
| --- | --- | --- |
| codebook-0 labels shifted twice (script + HF loss) | 8.82 (chance is 8.0) | 1.86 |
| codebooks 1–15 predicted from the frame's own position, plus a second shift | 10.12 (chance 7.6) | 5.26 |
| text embeddings skip `text_projection` (crashes on 0.6B, silently wrong on 1.7B) | — | applied |
| trains the all-text-first layout; oMLX speaks with streaming text | 3.05 (other layout) | 3.13 streaming + `japanese` token |
| pure bf16 weights: most AdamW updates at lr ≤ 2e-5 round away | — | fp32 master weights |

Smoke-tested end to end on the Mac (CPU, 0.6B, 12 lines): codec extraction, training, checkpoint save, qwen-tts inference, and loading + streaming the checkpoint with oMLX's mlx-audio 0.4.3 as a custom voice.
