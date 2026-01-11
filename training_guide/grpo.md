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
- Set `eval_steps` and pass `eval_dataset` to run evaluation during training.
- vLLM is supported via `engine="vllm"` and optional `vllm_args`.
- When using vLLM, weights are refreshed each training step by saving to
  `output_dir/vllm_latest` and re-initializing the engine.
- RLFusion sets `VLLM_ATTENTION_BACKEND=FLASH_ATTN` when the variable is unset.
