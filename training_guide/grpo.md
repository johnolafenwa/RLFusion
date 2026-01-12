# GRPO Training Guide

Recommended starting point:
- model: `Qwen/Qwen2.5-0.5B-Instruct`
- optimizer: AdamW
- learning rate: 1e-5
- sampling temperature: 0.6 to 0.8
- group size: 2 to 4
- ppo steps: 1 to 2
- clip eps: 0.2
- kl penalty: 0.0 to 0.1
- max new tokens: 64 to 256
- batch size: 1 to 4 (per device)
- max grad norm: 1.0

Notes:
- Increase `group_size` for more stable advantage estimates.
- If outputs collapse, increase `kl_penalty` slightly.
- Set `eval_steps` and pass an `Evaluator` to run evaluation during training.
- vLLM is supported in the `Evaluator` via `engine="vllm"` and optional `vllm_args`.

## Distributed Training (Accelerate)
To run GRPO on multiple GPUs:

1) Pass `use_accelerate=True` to `GRPOTrainer`.
2) Launch with Accelerate:

```bash
accelerate launch --num_processes 2 examples/grpo_trainer_math.py
```

Notes:
- `batch_size` and `group_size` are per-process; effective sample throughput scales with `num_processes`.
- Checkpoints and evaluation (when `eval_steps` + `evaluator` are set) run on the main process.
