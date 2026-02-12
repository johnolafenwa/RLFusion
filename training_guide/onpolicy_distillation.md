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
