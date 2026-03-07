# Experiment Notes - 2026-02-11

## Scope

Ran end-to-end experiments on `Qwen/Qwen3-4B-Base` with:

1. UltraChat SFT
2. MMLU comparison (base vs SFT)
3. GRPO on RiddleBench from SFT checkpoint (`ref KL disabled`, `kl_penalty=0.0`)
4. RiddleSense / ARC-Challenge comparison (SFT vs RL)
5. RiddleBench held-out split analysis and reward/prompt fixes

All dates/times below refer to `2026-02-11` (UTC logs in `runs/logs`).

## Environment / Execution

- Repo: `johnolafenwa/RLFusion`
- Hardware: 2x NVIDIA H200
- Launch mode for training: `accelerate --num_processes 2 --mixed_precision bf16`
- Package manager/runtime: `uv`

## Major Runs and Outputs

### 1) SFT (4B, accelerate)

- Output: `outputs/ultrachat_sft_qwen3_4b_v4_accel/final`
- Config highlights:
  - `num_steps=1200`
  - `batch_size=1`
  - `max_seq_len=4096`
  - `lr=1e-5`

### 2) MMLU (base vs SFT)

- Base: `outputs/evals/mmlu_base_qwen3_4b/metrics.json`
  - `reward_mean=0.5819725` (`n=1531`)
- SFT: `outputs/evals/mmlu_sft_qwen3_4b_v4/metrics.json`
  - `reward_mean=0.6087524` (`n=1531`)
- Delta (SFT - Base): `+0.0267799`

### 3) GRPO v2 (pre reward/prompt fix)

- Output: `outputs/grpo_riddlebench_qwen3_4b_from_sft_v2_noeval/final`
- Config highlights:
  - `num_steps=400`, `group_size=4`, `ppo_steps=2`
  - `max_new_tokens=64`
  - `kl_penalty=0.0`
  - in-train eval disabled (`eval_steps=1000000`) to avoid distributed eval deadlock
- Reward trace (`runs/logs/grpo_riddlebench_qwen3_4b_from_sft_v2_noeval.log`):
  - 40 logged points
  - avg `reward_mean=0.09375`
  - non-zero at 4 points: steps `30, 60, 100, 130`

### 4) SFT vs RL(v2) on RiddleSense / ARC

- SFT RiddleSense: `0.6715000` (`outputs/evals/riddlesense_sft_qwen3_4b_v4/metrics.json`)
- RL(v2) RiddleSense: `0.6850000` (`outputs/evals/riddlesense_rl_qwen3_4b_v2/metrics.json`)
- Delta: `+0.0135`

- SFT ARC-Challenge: `0.8260869` (`outputs/evals/arc_challenge_sft_qwen3_4b_v4/metrics.json`)
- RL(v2) ARC-Challenge: `0.8227425` (`outputs/evals/arc_challenge_rl_qwen3_4b_v2/metrics.json`)
- Delta: `-0.0033445`

## RiddleBench Test-Split Findings

Dataset has only `train` split on HF; "test" used here is local held-out partition:

- `RiddleBenchDataset(train=False, train_split_ratio=0.9)` (`n=174`)

### Before reward/prompt fixes

- SFT: `0.0` (`outputs/evals/riddlebench_test_sft_qwen3_4b_v2/metrics.json`)
- RL(v2): `0.0` (`outputs/evals/riddlebench_test_rl_qwen3_4b_v2/metrics.json`)

### After reward parser + stricter short-answer instruction (no retrain)

- SFT: `0.1839080` (`outputs/evals/riddlebench_test_sft_qwen3_4b_after_rewardfix/metrics.json`)
- RL(v2): `0.1091954` (`outputs/evals/riddlebench_test_rl_qwen3_4b_after_rewardfix/metrics.json`)

## Fixes Applied During Session

### Trainer/accelerate robustness

- Per-rank `device_map` under accelerate in:
  - `src/rlfusion/trainers/sft_trainer.py`
  - `src/rlfusion/trainers/grpo_trainer.py`
- DDP generate compatibility fix:
  - `src/rlfusion/inference/hf_utils.py` (unwrap `.module` when needed)
- Attention impl/device-map helper updates:
  - `src/rlfusion/trainers/utils.py`

### Tokenizer compatibility for local checkpoints

- Added compatibility kwargs for `extra_special_tokens` list-vs-dict mismatch:
  - `src/rlfusion/trainers/utils.py`
  - wired into:
    - `src/rlfusion/evaluation/evaluator.py`
    - `src/rlfusion/trainers/sft_trainer.py`
    - `src/rlfusion/trainers/grpo_trainer.py`

### New eval script

- Added `examples/riddlesense_arc_eval.py` for evaluator-based RiddleSense/ARC runs.

### RiddleBench reward/prompt changes

- File: `src/rlfusion/datasets/riddlebench.py`
- Added robust candidate extraction and normalization for:
  - boxed answers
  - MCQ letter forms (`A`, `A.`, `Option (A)`, etc.)
  - numeric/text value matching normalization
- Prompting updated to boxed-output policy:
  - system prompt requests `\boxed{...}`
  - user instruction requests exactly one boxed final answer
- Tests extended in `tests/test_riddlebench_dataset.py` (all passing).

## GRPO v3 (boxed prompt + reward fix)

- Output: `outputs/grpo_riddlebench_qwen3_4b_from_sft_v3_boxed_rewardfix/final`
- Config highlights:
  - `num_steps=400`
  - `kl_penalty=0.0`
  - `max_new_tokens=16` (reduced)
  - `eval_steps=1000000` (no in-train eval)
- Reward trace (`runs/logs/grpo_riddlebench_qwen3_4b_from_sft_v3_boxed_rewardfix.log`):
  - 40 logged points
  - avg `reward_mean=0.24375`
  - non-zero at 11 points:
    - `30, 60, 100, 130, 140, 250, 290, 300, 330, 360, 390`

## Final Comparison After GRPO v3

### RiddleBench held-out (boxed protocol)

- SFT: `0.0` (`outputs/evals/riddlebench_test_sft_qwen3_4b_after_grpo_v3/metrics.json`)
- RL(v3): `0.2701149` (`outputs/evals/riddlebench_test_rl_v3_qwen3_4b_after_grpo_v3/metrics.json`)
- Delta: `+0.2701149`

### RiddleSense

- SFT: `0.6715000`
- RL(v3): `0.6815000`
- Delta: `+0.0100`

### ARC-Challenge

- SFT: `0.8260869`
- RL(v3): `0.8193980`
- Delta: `-0.0066889`

## Error Analysis Summary (from `results.jsonl`)

### RiddleBench SFT under boxed protocol

- Predominant failure mode: blank/whitespace output (format/output control issue).

### RiddleBench RL(v3)

- Improvement over SFT is substantial, but remaining errors are mostly wrong content:
  - wrong choice/value predictions
  - frequent chat leakage (`...\\nuser\\n...`) in completions

### RiddleSense / ARC

- Dominant errors are wrong-choice reasoning, not parser formatting misses.
- ARC has a small label-schema quirk in dataset rows with numeric answer keys (`1-4`).

## Operational Notes

- In-train GRPO eval with distributed setup deadlocked when only rank 0 evaluated and rank 1 waited at barrier; workaround used: disable in-train eval.
- Local push to GitHub failed in this environment due missing credentials; changes were committed locally and bundled:
  - commit: `9b01e8a`
  - bundle: `/home/jovyan/work/RLFusion-main-9b01e8a.bundle`
