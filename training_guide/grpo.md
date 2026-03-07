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
- On compatible CUDA GPUs, RLFusion now defaults to FlashAttention automatically. Hopper prefers FA3; Ampere/Ada/Hopper otherwise prefer FA2.

## vLLM-Accelerated Generation

Generation is the dominant bottleneck in GRPO training (80-90% of wall-clock time). On CUDA,
`GRPOTrainer` now defaults to colocated vLLM generation for 3-5x speedup and falls back to HF
generation on non-GPU devices:

```python
trainer = GRPOTrainer(
    model="your-model",
    train_dataset=dataset,
    vllm_args={
        "gpu_memory_utilization": 0.5,
        "tensor_parallel_size": 1,
    },
    # ... other args
)
```

Install vLLM first (Linux + CUDA only):
```bash
uv pip install -e ".[vllm]"
```

The repo-standard vLLM stack is `vllm 0.17.x` with matching Linux builds of
`torch 2.10.x`, `torchaudio 2.10.x`, and `torchvision 0.25.x`.

For HF-side FlashAttention, Liger, and `AdamW8bit`, install the repo-standard CUDA training extra:

```bash
uv sync --extra gpu-train --extra vllm --extra dev --extra test
```

Notes:
- On CUDA, HF trainers automatically use the fastest available HF attention backend.
- If `liger-kernel` is installed, the trainable HF model path automatically uses the Liger wrappers.
- `flash-attn` may compile from source on the current `torch 2.10.x` stack because upstream does
  not ship a matching wheel for every Linux platform.

Parameters:
- `use_vllm` — leave unset for the default behavior (`True` on CUDA, `False` otherwise), set
  `False` to force HF generation, or set `True` to require vLLM explicitly.
- `vllm_args` — dict passed to `vllm.LLM()`. Key settings:
  - `gpu_memory_utilization` (default 0.5) — fraction of GPU memory for KV cache. Lower = more memory for training.
  - `tensor_parallel_size` — number of GPUs for vLLM inference.
- `vllm_enable_sleep=True` — put vLLM to sleep between generations to free GPU memory.

Notes:
- Weights are automatically synced from the training model to vLLM after each optimizer step.
- Start with `gpu_memory_utilization=0.5` and adjust based on OOM behavior.
- `vllm_enable_sleep` adds per-step overhead but allows higher memory utilization.
- `vllm_enable_sleep` automatically enables the underlying vLLM sleep mode.
- If `use_accelerate=True`, keep `tensor_parallel_size=1` and `pipeline_parallel_size=1` so each process owns one local vLLM engine.
- vLLM now uses its own default attention-backend auto-selection unless you explicitly override `VLLM_ATTENTION_BACKEND`.

FlashAttention 4:

```python
trainer = GRPOTrainer(
    ...,
    vllm_args={
        "gpu_memory_utilization": 0.5,
        "attention_config": {
            "backend": "FLASH_ATTN",
            "flash_attn_version": 4,
        },
    },
)
```

Notes:
- This requests FA4 through vLLM's `FLASH_ATTN` backend.
- FA4 is relevant on Blackwell-class GPUs; Hopper generally uses FA3, while Ampere/Ada remain on FA2.
