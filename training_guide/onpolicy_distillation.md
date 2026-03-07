# On-Policy Distillation Training Guide

Reference method: [Thinking Machines on-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)

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

Objective (aligned with the reference method):
1. Sample completions from the student.
2. Compute student and teacher token log-probs on those sampled completions.
3. Build completion-only masks (prompt tokens excluded).
4. Use token-level advantage `logp_teacher - logp_student_old`.
5. Apply PPO-style clipping with `clip_eps` for stable updates across `ppo_steps`.

Notes:
- Use a stronger teacher for better distillation signal.
- If reverse KL is noisy, lower temperature or reduce max new tokens.
- Set `eval_steps` and pass an `eval_dataset` to run reward evaluation during training.
- `eval_dataset` items must implement `get_reward`.
- Reward metrics are only computed when sample `answer` is populated; distillation loss itself does not require reward.
- If generation stops immediately, the trainer still keeps a one-token completion mask so the sample contributes distillation signal.
- For multi-GPU, run with `accelerate launch` and set `use_accelerate=True`.

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
- Checkpoints and evaluation (when `eval_steps` + `eval_dataset` are set) run on the main process.
- On compatible CUDA GPUs, RLFusion now defaults to FlashAttention automatically. Hopper prefers FA3; Ampere/Ada/Hopper otherwise prefer FA2.

## vLLM-Accelerated Generation

On CUDA, `OnPolicyDistillationTrainer` now defaults to colocated vLLM generation and falls back
to HF generation on non-GPU devices.

```python
trainer = OnPolicyDistillationTrainer(
    model="your-student-model",
    teacher_model="your-teacher-model",
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

Notes:
- Weights are automatically synced from the training model to vLLM after each optimizer step.
- Set `use_vllm=False` to force HF generation, or `use_vllm=True` to require vLLM explicitly.
- vLLM now uses its own default attention-backend auto-selection unless you explicitly override `VLLM_ATTENTION_BACKEND`.
- `vllm_enable_sleep=True` can reduce peak memory pressure at the cost of per-step wake/sleep overhead.
- `vllm_enable_sleep=True` automatically enables the underlying vLLM sleep mode.
- If `use_accelerate=True`, keep `tensor_parallel_size=1` and `pipeline_parallel_size=1` so each process owns one local vLLM engine.

FlashAttention 4:

```python
trainer = OnPolicyDistillationTrainer(
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
