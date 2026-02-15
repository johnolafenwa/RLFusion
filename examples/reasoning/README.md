# Reasoning Pipeline: SFT → GRPO → AIME2025 Evaluation

This folder provides an end-to-end flow:

1. SFT on `johnolafenwa/reasoning-sft` (`train` / `test` splits)
2. GRPO on `johnolafenwa/reasoning-rl` using the SFT checkpoint
3. AIME2025 evaluation on base/SFT/GRPO checkpoints

Scripts:

- `reasoning_sft_train.py`
- `reasoning_grpo_train.py`
- `aime2025_evaluate.py`

## 1) Install and run SFT

`reasoning_sft_train.py` uses native splits from `johnolafenwa/reasoning-sft`:

- `train` for training
- `test` for evaluation

Data mapping is strict:

- `input` → user prompt message
- `output` → assistant target message

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

## 2) GRPO on reasoning-rl

`reasoning_grpo_train.py` starts from SFT checkpoint and applies reward that requires **all**:

1. response starts with `<think>` (case-sensitive)
2. includes `</think>`
3. non-empty `<think>...</think>` content
4. `\boxed{...}` exists in text after `</think>`
5. boxed answer exactly equals dataset answer (`get_boxed_answer(... )` match, case-sensitive)

Dataset uses native splits from `johnolafenwa/reasoning-rl`:

- `train` for GRPO updates
- `test` for periodic evaluation

Data mapping is strict:

- `problem` → user prompt message
- `answer` → target answer used by reward matching

```bash
python /Users/johnolafenwa/source/rlfusion/RLFusion/examples/reasoning/reasoning_grpo_train.py \
  --sft-checkpoint /Users/johnolafenwa/source/rlfusion/RLFusion/outputs/reasoning/reasoning_sft/final \
  --output-dir ./outputs/reasoning/reasoning_grpo \
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

## Notes

- SFT and GRPO now both use dataset-native `train` / `test` splits with no custom split logic.
- GRPO and SFT scripts support both `--num-steps` (legacy) and `--num-epochs` (preferred for full data coverage).
- Both scripts support periodic evaluation via `--eval-steps`.
