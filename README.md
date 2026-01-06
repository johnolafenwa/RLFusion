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

## Dev Setup
```bash
uv pip install -e ".[dev,test]"
```

## Quickstart
Use the provided examples:

```bash
python examples/sft_trainer_example.py
python examples/grpo_trainer_math.py
python examples/onpolicy_distillation_example.py
```

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

### RLVR (GRPO)
`GRPOTrainer` samples completions from the model, computes rewards via the environment, and optimizes a GRPO objective. It accepts `train_dataset` and optional `eval_dataset`, and exposes `test()` for evaluation.

### On-policy Distillation
`OnPolicyDistillationTrainer` samples from the student and minimizes reverse KL to a fixed teacher distribution over completion tokens. It accepts `train_dataset` and optional `eval_dataset`, and exposes `test()` for evaluation.

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

## Build a Wheel
```bash
uv pip install --upgrade build
uv run python -m build --wheel
```
Artifacts land in `./dist`.
