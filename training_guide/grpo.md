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
- Set `eval_steps` and pass an `eval_dataset` to run reward evaluation during training.
- `eval_dataset` items must implement `get_reward`.

## Distributed Training (Accelerate)
To run GRPO on multiple GPUs:

1) Pass `use_accelerate=True` to `GRPOTrainer`.
2) Launch with Accelerate:

```bash
accelerate launch --num_processes 2 examples/grpo_trainer_math.py
```

Notes:
- `batch_size` and `group_size` are per-process; effective sample throughput scales with `num_processes`.
- Checkpoints and evaluation (when `eval_steps` + `eval_dataset` are set) run on the main process.

## vLLM-Accelerated Generation

Generation is the dominant bottleneck in GRPO training (80-90% of wall-clock time). Enable colocated vLLM generation for 3-5x speedup:

```python
trainer = GRPOTrainer(
    model="your-model",
    train_dataset=dataset,
    use_vllm=True,
    vllm_args={
        "gpu_memory_utilization": 0.5,
        "tensor_parallel_size": 1,
    },
    # ... other args
)
```

Install vLLM first (Linux + CUDA only):
```bash
pip install vllm
```

Parameters:
- `use_vllm=True` — enable colocated vLLM engine for generation.
- `vllm_args` — dict passed to `vllm.LLM()`. Key settings:
  - `gpu_memory_utilization` (default 0.5) — fraction of GPU memory for KV cache. Lower = more memory for training.
  - `tensor_parallel_size` — number of GPUs for vLLM inference.
- `vllm_enable_sleep=True` — put vLLM to sleep between generations to free GPU memory (requires vLLM >= 0.7).

Notes:
- Weights are automatically synced from the training model to vLLM after each optimizer step.
- Start with `gpu_memory_utilization=0.5` and adjust based on OOM behavior.
- `vllm_enable_sleep` adds per-step overhead but allows higher memory utilization.
