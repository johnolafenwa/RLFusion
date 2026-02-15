#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B-Base}"
MAX_LEN="${MAX_LEN:-8192}"
OUT_DIR="${OUT_DIR:-./outputs/reasoning_full_8k_20260212}"
USE_VLLM="${USE_VLLM:-0}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"

mkdir -p "$OUT_DIR/logs"

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Starting full reasoning pipeline"
echo "model=$MODEL_ID"
echo "max_len=$MAX_LEN"
echo "use_vllm=$USE_VLLM"
echo "out_dir=$OUT_DIR"

VLLM_FLAGS=()
if [[ "$USE_VLLM" == "1" ]]; then
  VLLM_FLAGS+=(--use-vllm --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION")
fi

"$UV_BIN" run python examples/reasoning/aime2025_evaluate.py \
  --model "$MODEL_ID" \
  --output-dir "$OUT_DIR/aime_eval_base" \
  --max-new-tokens "$MAX_LEN" \
  --show-progress \
  2>&1 | tee "$OUT_DIR/logs/01_aime_eval_base.log"

"$UV_BIN" run python examples/reasoning/reasoning_sft_train.py \
  --model "$MODEL_ID" \
  --output-dir "$OUT_DIR/reasoning_sft_full" \
  --num-epochs 1 \
  --batch-size 1 \
  --saving-steps 100 \
  --logging-steps 10 \
  --max-seq-len "$MAX_LEN" \
  --lr 1e-5 \
  --seed 42 \
  2>&1 | tee "$OUT_DIR/logs/02_sft_train_full.log"

"$UV_BIN" run python examples/reasoning/aime2025_evaluate.py \
  --model "$OUT_DIR/reasoning_sft_full/final" \
  --output-dir "$OUT_DIR/aime_eval_sft" \
  --max-new-tokens "$MAX_LEN" \
  --show-progress \
  2>&1 | tee "$OUT_DIR/logs/03_aime_eval_sft.log"

"$UV_BIN" run python examples/reasoning/reasoning_grpo_train.py \
  --sft-checkpoint "$OUT_DIR/reasoning_sft_full/final" \
  --output-dir "$OUT_DIR/reasoning_grpo_full" \
  --num-epochs 1 \
  --batch-size 1 \
  --group-size 4 \
  --ppo-steps 2 \
  --max-new-tokens "$MAX_LEN" \
  --sampling-temperature 0.7 \
  --top-p 0.9 \
  --learning-rate 1e-5 \
  --saving-steps 50 \
  --logging-steps 5 \
  --seed 42 \
  "${VLLM_FLAGS[@]}" \
  2>&1 | tee "$OUT_DIR/logs/04_grpo_train_full.log"

"$UV_BIN" run python examples/reasoning/aime2025_evaluate.py \
  --model "$OUT_DIR/reasoning_grpo_full/final" \
  --output-dir "$OUT_DIR/aime_eval_grpo" \
  --max-new-tokens "$MAX_LEN" \
  --show-progress \
  2>&1 | tee "$OUT_DIR/logs/05_aime_eval_grpo.log"

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Pipeline completed"
