# On-Policy Distillation Training Guide

Recommended starting point:
- student model: `Qwen/Qwen2.5-0.5B-Instruct`
- teacher model: `Qwen/Qwen2.5-1.5B-Instruct` (or larger)
- optimizer: AdamW
- learning rate: 1e-5
- sampling temperature: 0.6 to 0.8
- max new tokens: 64 to 256
- batch size: 1 to 4 (per device)
- max grad norm: 1.0

Notes:
- Use a stronger teacher for better distillation signal.
- If reverse KL is noisy, lower temperature or reduce max new tokens.
