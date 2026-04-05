#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LTX23_PROMPT="${LTX23_PROMPT:-A beautiful sunset over the ocean}"
export LTX23_IMAGE_PATH="${LTX23_IMAGE_PATH:-/tmp/ltx23_i2v_input_sunset.png}"
export LTX23_OUTPUT_PATH="${LTX23_OUTPUT_PATH:-/tmp/ltx23_official_i2v_sunset.mp4}"
export LTX23_STREAMING_PREFETCH_COUNT="${LTX23_STREAMING_PREFETCH_COUNT:-1}"
export LTX23_OFFICIAL_VENV="${LTX23_OFFICIAL_VENV:-/tmp/ltx23_official_venv}"

if [ -d /tmp/LTX-2 ]; then
  LTX_REPO_ROOT=/tmp/LTX-2
elif [ -d /tmp/LTX-2-official ]; then
  LTX_REPO_ROOT=/tmp/LTX-2-official
else
  echo "LTX repo not found under /tmp" >&2
  exit 1
fi

CHECKPOINT="$(find /root/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots \( -path '*/ltx-2.3-20b-dev.safetensors' -o -path '*/ltx-2.3-22b-dev.safetensors' \) | head -n 1)"
GEMMA_ROOT="$(find /root/.cache/huggingface/hub/models--Lightricks--LTX-2/snapshots -path '*/tokenizer/tokenizer.model' | head -n 1 | xargs dirname | xargs dirname)"

if [ ! -f "$CHECKPOINT" ]; then
  echo "LTX-2.3 checkpoint not found in HF cache" >&2
  exit 1
fi

if [ ! -d "$GEMMA_ROOT" ]; then
  echo "Gemma root not found in HF cache" >&2
  exit 1
fi

. "$LTX23_OFFICIAL_VENV/bin/activate"
export PYTHONPATH="$LTX_REPO_ROOT/packages/ltx-core/src:$LTX_REPO_ROOT/packages/ltx-pipelines/src"

python -m ltx_pipelines.ti2vid_one_stage \
  --checkpoint-path "$CHECKPOINT" \
  --gemma-root "$GEMMA_ROOT" \
  --prompt "$LTX23_PROMPT" \
  --output-path "$LTX23_OUTPUT_PATH" \
  --image "$LTX23_IMAGE_PATH" 0 1.0 \
  --streaming-prefetch-count "$LTX23_STREAMING_PREFETCH_COUNT"
