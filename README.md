# RLFusion
Minimalist post-training utilities for LLMs with a focus on clarity and ease of learning.

## Features
- SFT with role-aware masking (mask non-assistant tokens, train assistant tokens)
- RLVR with GRPO
- On-policy distillation (reverse KL to a teacher on student samples)
- Simple `EnvBase` for prompts + answers

## Install
From the repo root:

```bash
uv pip install -e .
```

If you don't have `uv` installed:
```bash
python -m pip install uv
```

Optional extras:

```bash
# FlashAttention support (Linux + CUDA)
uv pip install -e ".[flash]" --no-build-isolation

# vLLM support (Linux + CUDA)
uv pip install -e ".[vllm]"

# Both
uv pip install -e ".[flash,vllm]" --no-build-isolation
```

## Flash Attention (Optional)
If you have a CUDA-enabled GPU and want FlashAttention acceleration:

```bash
uv pip install -e ".[flash]" --no-build-isolation
```

RLFusion selects the fastest supported HF attention backend by default:
- FlashAttention-3 on Hopper GPUs when a compatible FA3 backend is available
- FlashAttention-2 on Ampere/Ada/Hopper GPUs when a compatible FA2 backend is available
- PyTorch SDPA otherwise

FlashAttention-3 is Hopper-only and requires a compatible FA3 installation. If no compatible
FlashAttention backend is available, the trainers automatically fall back to PyTorch SDPA.

You can override the automatic choice with:

```bash
export RLFUSION_ATTN_IMPLEMENTATION=sdpa
```

## vLLM (Optional)
To use vLLM:

```bash
uv pip install -e ".[vllm]"
```

Set `engine="vllm"` in the `Evaluator`. GRPO and on-policy distillation default to colocated
vLLM rollouts on CUDA and fall back to HF generation on non-GPU devices; pass `use_vllm=False`
to force HF generation, or `use_vllm=True` to require vLLM explicitly. Optional `vllm_args` are
forwarded to `vllm.LLM`.

Notes:
- The repo standardizes on `vllm 0.17.x` on Linux, with a matching `torch 2.10.x` /
  `torchaudio 2.10.x` / `torchvision 0.25.x` stack from the `vllm` extra.
- `vllm_enable_sleep=True` automatically enables the underlying vLLM sleep mode.
- When combining `use_accelerate=True` with `use_vllm=True`, keep
  `tensor_parallel_size=1` and `pipeline_parallel_size=1` so each trainer process owns exactly one
  local vLLM engine. Use multi-GPU vLLM only when not launching the trainer with Accelerate.

vLLM now uses its own default attention-backend auto-selection unless you override it yourself.
If you need to force a backend, use:

```bash
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

## Dev Setup
```bash
uv pip install -e ".[dev,test]"
```

## Quickstart
Minimal, inline examples for each trainer are included below.

## Distributed Training (Accelerate)
All trainers support multi-GPU distributed training via Hugging Face Accelerate.

1) Set `use_accelerate=True` on `SFTTrainer`, `GRPOTrainer`, or `OnPolicyDistillationTrainer`.
2) Launch your script with Accelerate:

```bash
accelerate launch --num_processes 2 your_script.py
```

Notes:
- `batch_size` is per-process; effective batch size is `batch_size * num_processes`.
- Logging, evaluation, and checkpoint saving are handled on the main process (rank 0).
- Reward-based evaluation (`eval_steps` + `eval_dataset`) runs on the main process.

## Core Concepts
### Environment
`EnvBase` represents a single sample with a chat-style prompt and an optional answer.

```python
from rlfusion.envs import EnvBase

env = EnvBase(
    prompt=[
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is 2 + 2?"},
    ],
    answer="4",
)
```

## Trainers
### SFT
`SFTTrainer` consumes `(prompt, answer)` and masks non-assistant tokens while training assistant tokens.
Eval supports two modes:
- Default (`eval_sample_completions=False`): logs token-level `ce_loss` and `perplexity` from labeled assistant tokens.
- Reward sampling (`eval_sample_completions=True`): generates completions and logs reward metrics. This mode requires `eval_dataset` environments with `get_reward`.

```python
from torch.utils.data import Dataset

from rlfusion.envs import EnvBase
from rlfusion.trainers.sft_trainer import SFTTrainer


class ToySFTDataset(Dataset):
    def __init__(self) -> None:
        self.samples = [
            EnvBase(
                prompt=[
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "What is 2 + 2?"},
                ],
                answer="4",
            )
        ]

    def __getitem__(self, index: int) -> EnvBase:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=ToySFTDataset(),
    num_steps=2,
    batch_size=1,
    saving_steps=2,
    logging_steps=1,
)
trainer.train()
```

For a real SFT corpus, you can use UltraChat:

```python
from rlfusion.datasets import UltraChatSFTDataset

train_dataset = UltraChatSFTDataset(train=True, max_samples=10_000, seed=42)
eval_dataset = UltraChatSFTDataset(train=False, max_samples=1_000, seed=42)
```

### RLVR (GRPO)
`GRPOTrainer` samples completions from the model, computes rewards via the environment, and optimizes a GRPO objective. To evaluate during training, set `eval_steps` and pass an `eval_dataset` of environments with `get_reward`.

```python
from rlfusion.datasets import MathDataset
from rlfusion.trainers.grpo_trainer import GRPOTrainer

train_dataset = MathDataset(num_samples=200, min_val=0, max_val=50, operand="add")
eval_dataset = MathDataset(num_samples=50, min_val=0, max_val=50, operand="add")

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=train_dataset,
    num_steps=2,
    saving_steps=2,
    logging_steps=1,
    eval_steps=1,
    eval_dataset=eval_dataset,
    group_size=2,
    ppo_steps=1,
    max_new_tokens=64,
    vllm_args={"gpu_memory_utilization": 0.5},
)
trainer.train()
```

### On-policy Distillation
`OnPolicyDistillationTrainer` follows the on-policy distillation recipe:
- sample trajectories from the student policy
- score sampled completion tokens under student and teacher
- optimize a PPO-style objective with token-level advantage `logp_teacher - logp_student_old`

This is distillation (no external RL reward in the loss). Reward logging during train/eval is optional and only used for monitoring.

```python
from rlfusion.datasets import MathDataset
from rlfusion.trainers.onpolicy_distillation_trainer import OnPolicyDistillationTrainer

train_dataset = MathDataset(num_samples=200, min_val=0, max_val=50, operand="add")
eval_dataset = MathDataset(num_samples=50, min_val=0, max_val=50, operand="add")

trainer = OnPolicyDistillationTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    teacher_model="Qwen/Qwen2.5-1.5B-Instruct",
    train_dataset=train_dataset,
    num_steps=2,
    saving_steps=2,
    logging_steps=1,
    eval_steps=1,
    eval_dataset=eval_dataset,
    sampling_temperature=0.7,
    generation_args={"top_p": 0.9},
    ppo_steps=1,
    clip_eps=0.2,
    max_new_tokens=64,
    max_grad_norm=1.0,
    vllm_args={"gpu_memory_utilization": 0.5},
)
trainer.train()
```

Notes:
- For reward metrics, `eval_dataset` environments should implement `get_reward`, and samples should have `answer` populated.
- In `trainer.test(...)`, `num_batches` must be `>= 1` when set.
- In `GRPOTrainer.test(...)` and `OnPolicyDistillationTrainer.test(...)`, `eval_temperature` must be `> 0` when set.
- Do not pass `temperature` via `generation_args`; use `sampling_temperature` during training/evaluation, and `eval_temperature` for RL trainer eval overrides.

## Training Guides
- SFT: `training_guide/sft.md`
- GRPO: `training_guide/grpo.md`
- On-policy distillation: `training_guide/onpolicy_distillation.md`
- Reasoning pipeline (dataset-specific): `examples/reasoning/README.md`

## Testing
```bash
uv run pytest
```

## Linting
```bash
uv run ruff check .
```

## Type Checking
```bash
uv run ty check src tests
```

If you prefer not to install it, you can run:
```bash
uvx ty check src tests
```

## Weights & Biases Logging
To enable W&B logging, pass `enable_wandb=True` to a trainer and optionally set:
- `wandb_project` (default: `sft`, `grpo`, or `onpolicy_distill`)
- `wandb_run_name`

Login with `uv`:
```bash
uv run wandb login
```

Or non-interactive:
```bash
uv run wandb login $WANDB_API_KEY
```

Example:
```python
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=dataset,
    enable_wandb=True,
    wandb_project="rlfusion",
    wandb_run_name="sft-demo",
)
```

## Build a Wheel
```bash
uv pip install --upgrade build
uv run python -m build --wheel
```
Artifacts land in `./dist`.

## Citation
If you use RLFusion in your work, please cite:

```bibtex
@software{rlfusion,
  title = {RLFusion},
  author = {Olafenwa, John},
  year = {2025},
  url = {https://github.com/johnolafenwa/rlfusion}
}
```
