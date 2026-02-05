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

## Flash Attention (Optional)
If you have a CUDA-enabled GPU and want FlashAttention2:

```bash
uv pip install flash-attn --no-build-isolation
```

If FlashAttention is not available, the trainers automatically fall back to PyTorch SDPA.

## vLLM (Optional)
To use vLLM in the `Evaluator`:

```bash
uv pip install vllm
```

Set `engine="vllm"` and optionally pass `vllm_args` (forwarded to `vllm.LLM`). RLFusion also sets
`VLLM_ATTENTION_BACKEND=FLASH_ATTN` when the variable is unset.

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
)
trainer.train()
```

### On-policy Distillation
`OnPolicyDistillationTrainer` samples from the student and minimizes reverse KL to a fixed teacher distribution over completion tokens. To evaluate during training, set `eval_steps` and pass an `eval_dataset` of environments with `get_reward`.

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
    max_new_tokens=64,
)
trainer.train()
```

## Training Guides
- SFT: `training_guide/sft.md`
- GRPO: `training_guide/grpo.md`
- On-policy distillation: `training_guide/onpolicy_distillation.md`

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
