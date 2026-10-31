"""Render the eval sentences with each checkpoint (and the Base model as a voice-clone baseline).

Same seeds for every model, so takes are comparable. Writes OUT/<model>/T<temp>_s<NN>_k<take>.wav.
Only these small wavs are downloaded for scoring; checkpoints stay on the server until one is chosen.

  python eval_ckpt.py --data data --out evalwav --base models/base ckpt/epoch-*
"""

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="*", type=Path)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", type=Path, help="Base model dir: also render the current voice-clone setup")
    ap.add_argument("--speaker", default="natsume")
    ap.add_argument("--temps", type=float, nargs="+", default=[0.9, 0.4])
    ap.add_argument("--takes", type=int, default=3)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    texts = json.loads((args.data / "eval_sentences.json").read_text(encoding="utf-8"))
    ref_wav, ref_text = str(args.data / "ref.wav"), (args.data / "ref.txt").read_text(encoding="utf-8").strip()
    jobs = ([("base", args.base)] if args.base else []) + [(c.name, c) for c in args.ckpts]
    device = "cpu" if args.cpu else "cuda:0"

    for name, path in jobs:
        tts = Qwen3TTSModel.from_pretrained(str(path), device_map=device, attn_implementation="sdpa",
                                            torch_dtype=torch.float32 if args.cpu else torch.bfloat16)
        (args.out / name).mkdir(parents=True, exist_ok=True)
        for temp in args.temps:
            sampling = dict(temperature=temp, subtalker_temperature=temp)
            for k in range(args.takes):
                torch.manual_seed(1000 + k)
                if name == "base":  # what the app does today: ICL clone from ref_a
                    wavs, sr = tts.generate_voice_clone(text=texts, language=["Japanese"] * len(texts),
                                                        ref_audio=ref_wav, ref_text=ref_text, **sampling)
                else:  # streaming-text layout + language token: what training used and oMLX does
                    wavs, sr = tts.generate_custom_voice(text=texts, speaker=[args.speaker] * len(texts),
                                                         language=["Japanese"] * len(texts),
                                                         non_streaming_mode=False, **sampling)
                for i, w in enumerate(wavs):
                    sf.write(args.out / name / f"T{temp}_s{i:02d}_k{k}.wav", w, sr)
            print(f"{name}: T={temp} {args.takes} takes x {len(texts)} sentences", flush=True)
        del tts
        if not args.cpu:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
