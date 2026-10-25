#!/usr/bin/env bash
# Remote pipeline, run from the work dir on the GPU server (data/ uploaded next to this script).
#   bash run.sh setup|codes|train|eval|all      extra args after the stage go to train.py
# Each stage writes a .done marker, so a rerun of "all" resumes where it stopped.
set -euo pipefail
cd "$(dirname "$0")"
STAGE=${1:-all}; shift || true
QWEN_COMMIT=022e286b98fbec7e1e916cb940cdf532cd9f488e
export QWEN_FT_DIR=$PWD/Qwen3-TTS/finetuning HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
PY=$PWD/.venv/bin/python

done_or() { [ -f "logs/$1.done" ] && { echo "[$1] already done"; return 0; }; return 1; }

setup() {
  done_or setup && return
  nvidia-smi --query-gpu=name,memory.total --format=csv
  command -v sox >/dev/null || { (sudo -n true 2>/dev/null && sudo apt-get -qq update && sudo apt-get -qq install -y sox) ||
    (apt-get -qq update && apt-get -qq install -y sox) || echo "sox not installed (only a warning)"; }
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=$HOME/.local/bin:$PATH
  uv venv -q --python 3.11 .venv
  uv pip install -q --python $PY torch torchaudio --index-url https://download.pytorch.org/whl/cu128
  uv pip install -q --python $PY qwen-tts==0.1.1 tensorboard hf_transfer "huggingface_hub[cli]"
  [ -d Qwen3-TTS ] || git clone -q https://github.com/QwenLM/Qwen3-TTS.git
  git -C Qwen3-TTS checkout -q $QWEN_COMMIT
  $PY -m huggingface_hub.commands.huggingface_cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir models/base >/dev/null
  $PY -m huggingface_hub.commands.huggingface_cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir models/tokenizer >/dev/null
  $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  touch logs/setup.done
}

codes() {
  done_or codes && return
  for split in train val; do
    (cd data && $PY $QWEN_FT_DIR/prepare_data.py --device cuda:0 --tokenizer_model_path ../models/tokenizer \
      --input_jsonl ${split}_raw.jsonl --output_jsonl ${split}_with_codes.jsonl)
  done
  wc -l data/*_with_codes.jsonl
  touch logs/codes.done
}

train() {
  done_or train && return
  $PY train.py --init models/base --data data --out ckpt "$@"
  touch logs/train.done
}

evaluate() {
  done_or eval && return
  $PY eval_ckpt.py --data data --out evalwav --base models/base ckpt/epoch-*
  tar czf evalwav.tgz evalwav ckpt/metrics.jsonl
  touch logs/eval.done
}

case $STAGE in
  setup) setup ;; codes) codes ;; train) train "$@" ;; eval) evaluate ;;
  all) setup; codes; train "$@"; evaluate; echo ALL DONE ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
