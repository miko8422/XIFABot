"""Single-speaker SFT for Qwen3-TTS 12Hz Base -> a CustomVoice checkpoint.

Adapted from QwenLM/Qwen3-TTS finetuning/sft_12hz.py (Apache-2.0, commit 022e286). Changes:
- fp32 master weights with bf16 autocast (upstream trains pure bf16, where most AdamW updates
  at lr 1e-5 are smaller than bf16 resolution and get rounded away); --bf16-weights restores upstream
- loss alignment fixed and text_projection applied (see compute_loss)
- streaming-text sequence layout, the one oMLX uses to speak, with the language token (see StreamingTextDataset)
- sdpa attention (no flash-attn build needed), frozen speaker encoder
- warmup + cosine LR, held-out validation loss every epoch, metrics.jsonl
- checkpoints saved in bf16 under a fixed speaker name

  python train.py --init models/base --data data --out ckpt [--epochs 6 --lr 1e-5]
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, get_cosine_schedule_with_warmup

sys.path.insert(0, os.environ.get("QWEN_FT_DIR", "Qwen3-TTS/finetuning"))
from dataset import TTSDataset  # noqa: E402
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel  # noqa: E402

SPK_ID = 3000  # codec_embedding row that upstream uses for the new speaker


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:  # paths in the jsonl are relative to the data dir
        for k in ("audio", "ref_audio"):
            r[k] = str(path.parent / r[k])
    return rows


class StreamingTextDataset(TTSDataset):
    """Training sequences in the layout mlx-audio/oMLX (and qwen-tts non_streaming_mode=False) use
    at inference: role, codec prefix, first text token on the codec-BOS step, then one trailing
    text token added to each audio frame. Upstream trains the all-text-first layout instead."""

    def __init__(self, rows, processor, config, language: str):
        super().__init__(rows, processor, config)
        self.cfg, tc = config, config.talker_config
        self.prefill = ([tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id] if language == "auto" else
                        [tc.codec_think_id, tc.codec_think_bos_id, tc.codec_language_id[language], tc.codec_think_eos_id])
        self.spk_pos = 3 + len(self.prefill)
        self.bos_pos = self.spk_pos + 2  # speaker slot, codec pad, then codec BOS + first text token

    def collate_fn(self, batch):
        c, tc = self.cfg, self.cfg.talker_config
        s = self.bos_pos
        B, T = len(batch), max(s + 2 + b["audio_codes"].shape[0] for b in batch)
        text = torch.full((B, T), c.tts_pad_token_id, dtype=torch.long)
        codec = torch.full((B, T), tc.codec_pad_id, dtype=torch.long)
        codes = torch.zeros((B, T, 16), dtype=torch.long)
        frame = torch.zeros((B, T), dtype=torch.bool)
        target = torch.full((B, T), -100, dtype=torch.long)
        attn = torch.zeros((B, T), dtype=torch.long)
        for i, b in enumerate(batch):
            ids, frames = b["text_ids"][0], b["audio_codes"]  # ids = role(3) + text tokens
            m = frames.shape[0]
            text[i, :3] = ids[:3]
            text[i, s - 1] = c.tts_bos_token_id
            text[i, s] = ids[3]
            trailing = torch.cat([ids[4:], torch.tensor([c.tts_eos_token_id])])[:m]
            text[i, s + 1 : s + 1 + len(trailing)] = trailing
            codec[i, 3 : self.spk_pos] = torch.tensor(self.prefill)
            codec[i, s] = tc.codec_bos_id
            codec[i, s + 1 : s + 1 + m] = frames[:, 0]
            codes[i, s + 1 : s + 1 + m] = frames
            frame[i, s + 1 : s + 1 + m] = True
            target[i, s : s + m] = frames[:, 0]  # position p predicts the frame at p+1
            target[i, s + m] = tc.codec_eos_token_id
            attn[i, : s + 1 + m] = 1
        return {"text_ids": text, "codec_channel": codec, "codec_ids": codes, "frame_mask": frame,
                "target": target, "attention_mask": attn, "ref_mels": torch.cat([b["ref_mel"] for b in batch])}

    def text_overflow(self) -> int:
        """Lines whose text has more tokens than the audio has frames (trailing text would be cut)."""
        return sum(len(self._tokenize_texts(self._build_assistant_text(r["text"]))[0]) - 8 > len(r["audio_codes"])
                   for r in self.data_list)


def compute_loss(model, batch, spk_pos: int):
    """Codebook-0 LM loss + 0.3 x sub-talker (codebooks 1-15) loss, aligned like inference.

    Upstream shifts the codebook-0 labels twice and predicts codebooks 1-15 from the frame's own
    position (which already saw them) with another shift; on the untouched Base model that scores
    8.82 / 10.12, worse than chance, vs 1.86 / 5.26 aligned. Text embeddings go through
    text_projection as at inference (upstream skips it; shapes only happen to match on 1.7B).
    """
    talker = model.talker
    speaker_embedding = model.speaker_encoder(batch["ref_mels"].to(model.device).to(model.dtype)).detach()
    attn, frame, codes = batch["attention_mask"], batch["frame_mask"], batch["codec_ids"]

    emb = talker.text_projection(talker.model.text_embedding(batch["text_ids"]))
    codec_emb = talker.model.codec_embedding(batch["codec_channel"])
    codec_emb[:, :3] = 0  # the role tokens carry no codec input
    codec_emb[:, spk_pos] = speaker_embedding
    emb = emb + codec_emb
    for i in range(1, 16):
        emb = emb + talker.code_predictor.get_input_embeddings()[i - 1](codes[:, :, i]) * frame.unsqueeze(-1)
    emb = emb * attn.unsqueeze(-1)

    out = talker(inputs_embeds=emb, attention_mask=attn, output_hidden_states=True)
    sel = batch["target"] != -100
    loss0 = F.cross_entropy(out.logits[sel].float(), batch["target"][sel])

    before_frame = torch.zeros_like(frame)  # hidden state that predicted each frame's codebook 0
    before_frame[:, :-1] = frame[:, 1:]
    targets = codes[frame]
    sub_logits, _ = talker.forward_sub_talker_finetune(targets, out.hidden_states[0][-1][before_frame])
    sub_loss = F.cross_entropy(sub_logits.float().reshape(-1, sub_logits.shape[-1]), targets[:, 1:].reshape(-1))
    return loss0 + 0.3 * sub_loss, loss0, sub_loss, speaker_embedding


def save_checkpoint(accelerator, model, init: Path, out_dir: Path, speaker: str, spk_emb: torch.Tensor):
    def skip(d, names):  # old talker weights, and the unchanged codec (linked below, saves 0.7 GB per checkpoint)
        return [n for n in names if Path(d) == init and (n == "speech_tokenizer" or
                                                         (n.startswith("model") and ".safetensors" in n))]

    shutil.copytree(init, out_dir, dirs_exist_ok=True, ignore=skip)
    if not (out_dir / "speech_tokenizer").exists():
        (out_dir / "speech_tokenizer").symlink_to((init / "speech_tokenizer").resolve(), target_is_directory=True)
    cfg = json.loads((init / "config.json").read_text(encoding="utf-8"))
    cfg["tts_model_type"] = "custom_voice"
    cfg.setdefault("talker_config", {})
    cfg["talker_config"]["spk_id"] = {speaker: SPK_ID}
    cfg["talker_config"]["spk_is_dialect"] = {speaker: False}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    state = {k: v.detach().to("cpu", torch.bfloat16) for k, v in accelerator.unwrap_model(model).state_dict().items()
             if not k.startswith("speaker_encoder")}
    state["talker.model.codec_embedding.weight"][SPK_ID] = spk_emb.to("cpu", torch.bfloat16)
    save_file(state, str(out_dir / "model.safetensors"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=Path, required=True, help="local Base model dir")
    ap.add_argument("--data", type=Path, required=True, help="dir with train/val_with_codes.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--speaker", default="natsume")
    ap.add_argument("--language", default="japanese", help="codec language token, or auto")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=30, help="optimizer steps")
    ap.add_argument("--train-text-embedding", action="store_true")
    ap.add_argument("--bf16-weights", action="store_true", help="upstream behaviour: pure bf16 weights")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum, cpu=args.cpu,
                              mixed_precision="no" if args.cpu else "bf16")
    tts = Qwen3TTSModel.from_pretrained(str(args.init), attn_implementation="sdpa",
                                        torch_dtype=torch.bfloat16 if args.bf16_weights else torch.float32)
    config = AutoConfig.from_pretrained(str(args.init))
    model = tts.model
    model.speaker_encoder.requires_grad_(False)
    if not args.train_text_embedding:  # 311M-row multilingual table; 4 h of one speaker shouldn't move it (saves ~3.7 GB)
        model.talker.model.text_embedding.requires_grad_(False)

    train_ds = StreamingTextDataset(load_jsonl(args.data / "train_with_codes.jsonl"), tts.processor, config, args.language)
    val_ds = StreamingTextDataset(load_jsonl(args.data / "val_with_codes.jsonl"), tts.processor, config, args.language)
    spk_pos = train_ds.spk_pos
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=train_ds.collate_fn, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=val_ds.collate_fn, num_workers=2)

    params = [p for p in model.parameters() if p.requires_grad]
    accelerator.print(f"trainable {sum(p.numel() for p in params) / 1e9:.2f}B of {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")
    optimizer = AdamW(params, lr=args.lr, weight_decay=0.01)
    total = math.ceil(len(train_dl) / args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, total)
    model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(model, optimizer, train_dl, val_dl, scheduler)

    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "metrics.jsonl", "a", encoding="utf-8")

    def emit(**row):
        row["t"] = round(time.time(), 1)
        log.write(json.dumps(row) + "\n")
        log.flush()
        accelerator.print(json.dumps(row))

    accelerator.print(f"train {len(train_ds)} val {len(val_ds)} ({train_ds.text_overflow()} lines with text cut) | "
                      f"{total} optimizer steps | language {args.language} | "
                      f"weights {'bf16' if args.bf16_weights else 'fp32'}")

    @torch.no_grad()
    def validate(epoch):
        model.eval()
        losses = []
        for batch in val_dl:
            with accelerator.autocast():
                loss, *_ = compute_loss(model, batch, spk_pos)
            losses.append(accelerator.gather_for_metrics(loss.detach().float().reshape(1)))
        model.train()
        emit(epoch=epoch, val_loss=round(torch.cat(losses).mean().item(), 4))

    validate(-1)  # loss of the untouched Base model, the reference point
    spk_emb = None
    step = 0
    model.train()
    for epoch in range(args.epochs):
        for i, batch in enumerate(train_dl):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss, l0, lsub, emb = compute_loss(model, batch, spk_pos)
                if spk_emb is None:  # every line shares ref.wav, so one embedding is the speaker
                    spk_emb = emb[0].detach().float()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    gnorm = accelerator.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                step += 1
                if step % 5 == 0:
                    emit(epoch=epoch, step=step, loss=round(loss.item(), 4), loss0=round(l0.item(), 4),
                         loss_sub=round(lsub.item(), 4), gnorm=round(float(gnorm), 3), lr=scheduler.get_last_lr()[0])
                if args.max_steps and step >= args.max_steps:
                    break
        validate(epoch)
        if accelerator.is_main_process:
            save_checkpoint(accelerator, model, args.init, args.out / f"epoch-{epoch}", args.speaker, spk_emb)
            emit(epoch=epoch, saved=str(args.out / f"epoch-{epoch}"))
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
