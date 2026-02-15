#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B-Base}"
BASE_EVAL_MAX_NEW_TOKENS="${BASE_EVAL_MAX_NEW_TOKENS:-256}"
MAX_LEN="${MAX_LEN:-4096}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}"
SFT_MAX_SEQ_LEN="${SFT_MAX_SEQ_LEN:-$MAX_LEN}"
SFT_GRADIENT_CHECKPOINTING="${SFT_GRADIENT_CHECKPOINTING:-0}"
SFT_TRAIN_MAX_SAMPLES="${SFT_TRAIN_MAX_SAMPLES:-10000}"
OUT_DIR="${OUT_DIR:-./outputs/reasoning_full_4k_20260212}"
SKIP_BASE_EVAL="${SKIP_BASE_EVAL:-0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
USE_VLLM="${USE_VLLM:-0}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"

mkdir -p "$OUT_DIR/logs"

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Starting full reasoning pipeline"
echo "model=$MODEL_ID"
echo "base_eval_max_new_tokens=$BASE_EVAL_MAX_NEW_TOKENS"
echo "sft_grpo_len=$MAX_LEN"
echo "sft_batch_size=$SFT_BATCH_SIZE"
echo "sft_max_seq_len=$SFT_MAX_SEQ_LEN"
echo "sft_gradient_checkpointing=$SFT_GRADIENT_CHECKPOINTING"
echo "sft_train_max_samples=$SFT_TRAIN_MAX_SAMPLES"
echo "skip_base_eval=$SKIP_BASE_EVAL"
echo "attn_implementation=${ATTN_IMPLEMENTATION:-<auto>}"
echo "use_vllm=$USE_VLLM"
echo "out_dir=$OUT_DIR"

if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
  export RLFUSION_ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION"
fi

if [[ "$SFT_GRADIENT_CHECKPOINTING" == "1" ]]; then
  SFT_GRADIENT_CHECKPOINTING_FLAG="--gradient-checkpointing"
else
  SFT_GRADIENT_CHECKPOINTING_FLAG="--no-gradient-checkpointing"
fi

VLLM_FLAGS=()
if [[ "$USE_VLLM" == "1" ]]; then
  VLLM_FLAGS+=(--use-vllm --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION")
fi

if [[ "$SKIP_BASE_EVAL" != "1" ]]; then
  "$UV_BIN" run python examples/reasoning/aime2025_evaluate.py \
    --model "$MODEL_ID" \
    --output-dir "$OUT_DIR/aime_eval_base_256" \
    --max-new-tokens "$BASE_EVAL_MAX_NEW_TOKENS" \
    --show-progress \
    2>&1 | tee "$OUT_DIR/logs/01_aime_eval_base_256.log"
fi

"$UV_BIN" run python examples/reasoning/reasoning_sft_train.py \
  --model "$MODEL_ID" \
  --output-dir "$OUT_DIR/reasoning_sft_full_4k" \
  --num-epochs 1 \
  --batch-size "$SFT_BATCH_SIZE" \
  --train-max-samples "$SFT_TRAIN_MAX_SAMPLES" \
  --saving-steps 100 \
  --logging-steps 10 \
  --max-seq-len "$SFT_MAX_SEQ_LEN" \
  --lr 1e-5 \
  --seed 42 \
  "$SFT_GRADIENT_CHECKPOINTING_FLAG" \
  2>&1 | tee "$OUT_DIR/logs/02_sft_train_full_4k.log"

"$UV_BIN" run python examples/reasoning/aime2025_evaluate.py \
  --model "$OUT_DIR/reasoning_sft_full_4k/final" \
  --output-dir "$OUT_DIR/aime_eval_sft_4k" \
  --max-new-tokens "$MAX_LEN" \
  --show-progress \
  2>&1 | tee "$OUT_DIR/logs/03_aime_eval_sft_4k.log"

"$UV_BIN" run python examples/reasoning/reasoning_grpo_train.py \
  --sft-checkpoint "$OUT_DIR/reasoning_sft_full_4k/final" \
  --output-dir "$OUT_DIR/reasoning_grpo_full_4k" \
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
  2>&1 | tee "$OUT_DIR/logs/04_grpo_train_full_4k.log"

"$UV_BIN" run python examples/reasoning/aime2025_evaluate.py \
  --model "$OUT_DIR/reasoning_grpo_full_4k/final" \
  --output-dir "$OUT_DIR/aime_eval_grpo_4k" \
  --max-new-tokens "$MAX_LEN" \
  --show-progress \
  2>&1 | tee "$OUT_DIR/logs/05_aime_eval_grpo_4k.log"

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Pipeline completed"
