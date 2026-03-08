# Reasoning Pipeline: SFT → GRPO → AIME2025 Evaluation

This folder provides an end-to-end flow:

1. SFT on `johnolafenwa/highschool-math-reasoning` (`train` / `test` splits)
2. GRPO on `johnolafenwa/highschool-math-reasoning-rl` using the SFT checkpoint
3. AIME2025 evaluation on base/SFT/GRPO checkpoints

Scripts:

- `reasoning_sft_train.py`
- `reasoning_grpo_train.py`
- `aime2025_evaluate.py`

## 1) Install and run SFT

From the repo root:

```bash
uv pip install -e .
```

Optional accelerators:

```bash
# HF Liger + bitsandbytes
uv sync --extra gpu-train --extra dev --extra test

# Optional HF FlashAttention backend
uv sync --extra gpu-train --extra flash --extra dev --extra test

# vLLM support for RL generation
uv sync --extra vllm

# Full validated CUDA stack without FlashAttention
uv sync --extra gpu-train --extra vllm --extra dev --extra test

# Full validated CUDA stack with FlashAttention added explicitly
uv sync --extra gpu-train --extra flash --extra vllm --extra dev --extra test
```

`reasoning_sft_train.py` uses native splits from `johnolafenwa/highschool-math-reasoning`:

- `train` for training
- `test` for evaluation

Data mapping is strict:

- `messages` → full chat transcript used directly for SFT
- `answer` → retained in the dataset but not used by the SFT trainer because the target assistant turn is already embedded in `messages`

```bash
python /Users/johnolafenwa/source/rlfusion/RLFusion/examples/reasoning/reasoning_sft_train.py \
  --model Qwen/Qwen3-8B-Base \
  --output-dir ./outputs/reasoning/reasoning_sft \
  --num-epochs 3 \
  --batch-size 1 \
  --train-max-samples 20000 \
  --test-max-samples 2000 \
  --eval-steps 100 \
  --saving-steps 100 \
  --logging-steps 10 \
  --max-seq-len 4096 \
  --lr 1e-5 \
  --seed 42
```

Tips:

- Use `--num-epochs N` to guarantee full passes over the split before repeating (`--num-steps` is still supported for legacy behavior).
- `save_final_only` defaults to `True`, so the run saves only the `final` checkpoint.
- Pass `--no-save-final-only` if you need intermediate `step_<N>` checkpoints.

## 2) GRPO on highschool-math-reasoning-rl

`reasoning_grpo_train.py` starts from SFT checkpoint and applies reward that requires **all**:

1. exactly one top-level `<think>...</think>` block
2. non-empty content inside that think block
3. exactly one terminal `\boxed{...}` answer after `</think>`
4. zero reward if the format contract is violated anywhere
5. boxed answer correctness checked with `math-verify` instead of raw string equality

Dataset uses native splits from `johnolafenwa/highschool-math-reasoning-rl`:

- `train` for GRPO updates
- `test` for periodic evaluation

Data mapping is strict:

- `messages[:-1]` → prompt turns used for rollout, with the format instruction appended to the final user turn
- `messages[-1]` → held-out assistant example, not shown to the policy during GRPO
- `answer` → target answer used by `math-verify` reward matching

**Forced HF generation:**

```bash
python examples/reasoning/reasoning_grpo_train.py \
  --sft-checkpoint ./outputs/reasoning/reasoning_sft/final \
  --output-dir ./outputs/reasoning/reasoning_grpo \
  --no-use-vllm \
  --num-epochs 1 \
  --batch-size 1 \
  --group-size 4 \
  --ppo-steps 2 \
  --max-new-tokens 1024 \
  --sampling-temperature 0.7 \
  --top-p 0.9 \
  --learning-rate 1e-5 \
  --train-max-samples 5000 \
  --test-max-samples 500 \
  --eval-steps 100 \
  --saving-steps 50 \
  --logging-steps 5 \
  --seed 42
```

**Default CUDA / vLLM path (3-5x faster generation):**

```bash
python examples/reasoning/reasoning_grpo_train.py \
  --sft-checkpoint ./outputs/reasoning/reasoning_sft/final \
  --output-dir ./outputs/reasoning/reasoning_grpo \
  --vllm-gpu-memory-utilization 0.5 \
  --num-epochs 1 \
  --batch-size 1 \
  --group-size 4 \
  --ppo-steps 2 \
  --max-new-tokens 4096 \
  --sampling-temperature 0.7 \
  --top-p 0.9 \
  --learning-rate 1e-5 \
  --train-max-samples 5000 \
  --test-max-samples 500 \
  --eval-steps 100 \
  --saving-steps 50 \
  --logging-steps 5 \
  --seed 42
```

On CUDA, the script defaults to colocated vLLM generation. Add `--no-use-vllm` to force the HF
generation path instead.

KL baseline is off by default. Enable with:

```bash
python .../reasoning_grpo_train.py --use-base-kl --kl-penalty 0.02
```

## 3) Evaluate checkpoints on AIME2025

Evaluate base/SFT/GRPO checkpoints separately with the evaluator:

```bash
python /Users/johnolafenwa/source/rlfusion/RLFusion/examples/reasoning/aime2025_evaluate.py \
  --model Qwen/Qwen3-8B-Base \
  --output-dir ./outputs/reasoning/aime_eval_base

python /Users/johnolafenwa/source/rlfusion/RLFusion/examples/reasoning/aime2025_evaluate.py \
  --model /Users/.../outputs/reasoning/reasoning_sft/final \
  --output-dir ./outputs/reasoning/aime_eval_sft

python /Users/johnolafenwa/source/rlfusion/RLFusion/examples/reasoning/aime2025_evaluate.py \
  --model /Users/.../outputs/reasoning/reasoning_grpo/final \
  --output-dir ./outputs/reasoning/aime_eval_grpo
```

Outputs are written to:

- `results.jsonl`
- `metrics.json`

## 4) Full pipeline scripts

The root directory has shell scripts that run the full SFT → GRPO → evaluation pipeline:

```bash
# 4K context pipeline
bash run_reasoning_full_4k.sh

# 8K context pipeline
bash run_reasoning_full_8k.sh

# Default auto-vLLM pipeline
bash run_reasoning_full_4k.sh
VLLM_GPU_MEMORY_UTILIZATION=0.6 bash run_reasoning_full_8k.sh

# Force HF generation instead of vLLM
USE_VLLM=0 bash run_reasoning_full_4k.sh
USE_VLLM=0 bash run_reasoning_full_8k.sh
```

## vLLM installation

vLLM is an optional dependency used to accelerate generation during GRPO training. Generation is the dominant bottleneck in RL training (80-90% of wall-clock time), and vLLM provides 3-5x speedup via PagedAttention, continuous batching, and KV-cache optimizations.

**Requirements:**

- Linux with CUDA (vLLM does not support macOS)
- CUDA 12.1+
- Python 3.12+ for this repo

**Install:**

```bash
uv pip install -e ".[vllm]"
```

This repo's `vllm` extra standardizes on `vllm 0.17.x` plus the matching Linux Torch stack
(`torch 2.10.x`, `torchaudio 2.10.x`, `torchvision 0.25.x`).

**vLLM flags for GRPO:**

| Flag | Default | Description |
|------|---------|-------------|
| `--use-vllm` / `--no-use-vllm` | auto (`on` for CUDA, `off` otherwise) | Override the default rollout backend |
| `--vllm-gpu-memory-utilization` | 0.5 | Fraction of GPU memory for vLLM KV cache (0-1). Lower values leave more memory for training. |
| `--vllm-tensor-parallel-size` | 1 | Number of GPUs for tensor-parallel vLLM inference |
| `--vllm-enable-sleep` | off | Put vLLM to sleep between generations to free GPU memory for training |

**Tips:**

- Start with `--vllm-gpu-memory-utilization 0.5`. If you get OOM during training, lower it. If generation is slow, raise it.
- `--vllm-enable-sleep` adds some latency per step (wake/sleep overhead) but allows higher memory utilization since vLLM releases GPU memory during the training phase.
- `--vllm-enable-sleep` automatically enables the underlying vLLM sleep mode.
- vLLM weights are automatically synced from the training model after each optimizer step.
- By default, CUDA runs use vLLM and non-GPU runs fall back to HF generation.
- With `--use-accelerate`, keep `--vllm-tensor-parallel-size 1` so each trainer process owns one local vLLM engine.
- Use `--vllm-tensor-parallel-size > 1` only when vLLM is managing the full multi-GPU inference group by itself.
- vLLM now uses its own default attention-backend auto-selection unless you explicitly override `VLLM_ATTENTION_BACKEND`.
- To request FlashAttention 4 on supported Blackwell GPUs, add
  `attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 4}` to the
  `vllm_args` dict in [reasoning_grpo_train.py](/home/jovyan/work/RLFusion/examples/reasoning/reasoning_grpo_train.py).
- Hopper generally uses FA3; Ampere and Ada remain on FA2.

## Notes

- SFT and GRPO now both use dataset-native `train` / `test` splits with no custom split logic.
- GRPO and SFT scripts support both `--num-steps` (legacy) and `--num-epochs` (preferred for full data coverage).
- Both scripts support periodic evaluation via `--eval-steps`.
