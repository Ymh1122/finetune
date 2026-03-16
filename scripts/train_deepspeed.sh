#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_deepspeed.sh \
#     /ABS/PATH/TO/qwen3-30b-a3b-instruct-2507 \
#     data

MODEL_PATH=${1:?model path required}
DATASET_DIR=${2:?dataset dir required}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export TOKENIZERS_PARALLELISM=false

# LLaMA-Factory entrypoints vary by install; common ones:
#   llamafactory-cli train ...
# or
#   python -m llamafactory.cli train ...

llamafactory-cli train \
  configs/qlora.yaml \
  --model_name_or_path "$MODEL_PATH" \
  --dataset_dir "$DATASET_DIR" \
  --deepspeed configs/ds_zero2.json
