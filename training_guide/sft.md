# SFT Training Guide

Recommended starting point:
- model: `Qwen/Qwen2.5-0.5B-Instruct` (or similar size)
- optimizer: AdamW
- learning rate: 1e-5 to 2e-5
- batch size: 1 to 8 (per device)
- num steps: 1k to 5k for small demos
- max sequence length: 1024 or 2048
- mask_prompt: true (mask non-assistant tokens)
- max grad norm: 1.0

Notes:
- If loss diverges, lower the learning rate or reduce max sequence length.
- If you want to train on prompts too, set `mask_prompt=False`.
- For eval during training, `eval_dataset` items must implement `get_reward`.
- Eval metrics are reward-based even though training optimizes token-level loss.

## Distributed Training (Accelerate)
To run SFT on multiple GPUs:

1) Pass `use_accelerate=True` to `SFTTrainer`.
2) Launch with Accelerate:

```bash
accelerate launch --num_processes 2 examples/sft_trainer_example.py
```

Notes:
- `batch_size` is per-process; effective batch size is `batch_size * num_processes`.
- Checkpoints and evaluation (when `eval_steps` + `eval_dataset` are set) run on the main process.
