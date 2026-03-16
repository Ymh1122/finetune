#!/usr/bin/env bash
set -euo pipefail

# Example:
#   bash scripts/serve_vllm_lora.sh \
#     /ABS/PATH/TO/qwen3-30b-a3b-instruct-2507 \
#     /ABS/PATH/TO/outputs/qwen3-30b-a3b-lora-material \
#     8000

MODEL_PATH=${1:?model path required}
LORA_PATH=${2:?lora adapter path required}
PORT=${3:-8000}

# Optional env overrides
TP_SIZE=${TP_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
VLLM_EXECUTOR_BACKEND=${VLLM_EXECUTOR_BACKEND:-}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}

# vLLM supports LoRA via --enable-lora and --lora-modules.
# Depending on vLLM version, the flag name may differ slightly.

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --port "$PORT" \
  --trust-remote-code \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable-lora \
  --lora-modules material="$LORA_PATH" \
  --max-model-len "$MAX_MODEL_LEN" \
  ${VLLM_EXECUTOR_BACKEND:+--distributed-executor-backend "$VLLM_EXECUTOR_BACKEND"} \
  ${VLLM_EXTRA_ARGS}
