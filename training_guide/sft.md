# SFT Training Guide

Recommended starting point:
- model: `Qwen/Qwen2.5-0.5B-Instruct` (or similar size)
- optimizer: AdamW
- learning rate: 1e-5 to 2e-5
- batch size: 1 to 8 (per device)
- num steps: 1k to 5k for small demos
- max sequence length: 1024 or 2048
- mask_prompt: true (mask user-role tokens)
- max grad norm: 1.0

Notes:
- If loss diverges, lower the learning rate or reduce max sequence length.
- If you want to train on prompts too, set `mask_prompt=False`.
