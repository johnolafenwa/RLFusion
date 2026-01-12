# On-Policy Distillation Training Guide

Recommended starting point:
- student model: `Qwen/Qwen2.5-0.5B-Instruct`
- teacher model: `Qwen/Qwen2.5-1.5B-Instruct` (or larger)
- optimizer: AdamW
- learning rate: 1e-5
- sampling temperature: 0.6 to 0.8
- ppo steps: 1 to 2
- clip eps: 0.2
- max new tokens: 64 to 256
- batch size: 1 to 4 (per device)
- max grad norm: 1.0

Notes:
- Use a stronger teacher for better distillation signal.
- If reverse KL is noisy, lower temperature or reduce max new tokens.
- Set `eval_steps` and pass an `Evaluator` to run evaluation during training.
- For multi-GPU, run with `accelerate launch` and set `use_accelerate=True`.
- vLLM is supported in the `Evaluator` via `engine="vllm"` and optional `vllm_args`.

## Distributed Training (Accelerate)
To run on-policy distillation on multiple GPUs:

1) Pass `use_accelerate=True` to `OnPolicyDistillationTrainer`.
2) Launch with Accelerate:

```bash
accelerate launch --num_processes 2 examples/onpolicy_distillation_example.py
```

Notes:
- `batch_size` is per-process; effective batch size is `batch_size * num_processes`.
- Student and teacher models are loaded in each process; plan GPU memory accordingly.
- Checkpoints and evaluation (when `eval_steps` + `evaluator` are set) run on the main process.
