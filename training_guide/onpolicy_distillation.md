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
- Set `eval_steps` and pass `eval_dataset` to run evaluation during training.
- vLLM is supported via `engine="vllm"` and optional `vllm_args`.
- When using vLLM, weights are refreshed each training step by saving to
  `output_dir/vllm_latest` and re-initializing the engine.
- RLFusion sets `VLLM_ATTENTION_BACKEND=FLASH_ATTN` when the variable is unset.
