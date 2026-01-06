# RLFusion
Minimalist post-training utilities for LLMs with a focus on clarity and speed.

## Features
- SFT with role-aware masking (mask user tokens, train everything else)
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

## Dev Setup
```bash
uv pip install -e ".[dev,test]"
```

## Quickstart
Minimal, inline examples for each trainer are included below.

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
`SFTTrainer` consumes `(prompt, answer)` and masks any `user`-role tokens while training all other tokens. It accepts `train_dataset` and optional `eval_dataset`, and exposes `test()` for evaluation.

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
    eval_dataset=ToySFTDataset(),
    num_steps=2,
    batch_size=1,
    saving_steps=2,
    logging_steps=1,
)
trainer.train()
```

### RLVR (GRPO)
`GRPOTrainer` samples completions from the model, computes rewards via the environment, and optimizes a GRPO objective. It accepts `train_dataset` and optional `eval_dataset`, and exposes `test()` for evaluation.

```python
from rlfusion.envs import EnvBase
from rlfusion.trainers.grpo_trainer import GRPOTrainer
from rlfusion.utils import get_boxed_answer


class SimpleMathEnv(EnvBase):
    def get_reward(self, prediction: str) -> float:
        if self.answer is None:
            return 0.0
        boxed = get_boxed_answer(prediction)
        return 1.0 if boxed == str(self.answer) else 0.0


dataset = [
    SimpleMathEnv(
        prompt=[{"role": "user", "content": "What is 2 + 2?"}],
        answer="4",
    ),
    SimpleMathEnv(
        prompt=[{"role": "user", "content": "What is 3 + 5?"}],
        answer="8",
    ),
]

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=dataset,
    eval_dataset=dataset,
    num_steps=2,
    saving_steps=2,
    logging_steps=1,
    group_size=2,
    ppo_steps=1,
    max_new_tokens=64,
)
trainer.train()
```

### On-policy Distillation
`OnPolicyDistillationTrainer` samples from the student and minimizes reverse KL to a fixed teacher distribution over completion tokens. It accepts `train_dataset` and optional `eval_dataset`, and exposes `test()` for evaluation.

```python
from rlfusion.envs import EnvBase
from rlfusion.trainers.onpolicy_distillation_trainer import OnPolicyDistillationTrainer
from rlfusion.utils import get_boxed_answer


class SimpleMathEnv(EnvBase):
    def get_reward(self, prediction: str) -> float:
        if self.answer is None:
            return 0.0
        boxed = get_boxed_answer(prediction)
        return 1.0 if boxed == str(self.answer) else 0.0


dataset = [
    SimpleMathEnv(
        prompt=[{"role": "user", "content": "What is 2 + 2?"}],
        answer="4",
    ),
    SimpleMathEnv(
        prompt=[{"role": "user", "content": "What is 3 + 5?"}],
        answer="8",
    ),
]

trainer = OnPolicyDistillationTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    teacher_model="Qwen/Qwen2.5-1.5B-Instruct",
    train_dataset=dataset,
    eval_dataset=dataset,
    num_steps=2,
    saving_steps=2,
    logging_steps=1,
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
