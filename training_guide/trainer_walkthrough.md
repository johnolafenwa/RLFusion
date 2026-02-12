## Trainer Walkthrough

This walkthrough explains how each trainer moves data through the system and where the key algorithm steps live in code. It mirrors the flow in `src/rlfusion/trainers/` and points to the exact methods to read.

### SFT (Supervised Fine-Tuning)

Goal: take prompt/answer pairs and train the model to predict assistant tokens only.

Key files and methods:
- `src/rlfusion/trainers/sft_trainer.py` -> `SFTTrainer.train`
- `src/rlfusion/trainers/sft_trainer.py` -> `SFTTrainer._build_batch`
- `src/rlfusion/trainers/sft_trainer.py` -> `SFTTrainer._chat_template_message_spans`

Flow:
1. `SFTTrainer.train` samples a random batch of examples and extracts prompts/answers.
2. `_build_batch` turns each prompt into chat-template token ids and builds labels.
   - It uses `_chat_template_message_spans` to find token spans per role.
   - Only assistant spans are labeled; prompts are masked to avoid learning the prompt format.
3. The model runs a standard forward pass with `labels`, and backprop updates weights.
4. Logging and checkpointing happen on the configured intervals.

### GRPO (Group Reinforcement Policy Optimization)

Goal: generate multiple completions per prompt, score them with a reward, and apply PPO-style updates with an optional KL penalty to a reference model.

Key files and methods:
- `src/rlfusion/trainers/grpo_trainer.py` -> `GRPOTrainer.train`
- `src/rlfusion/trainers/grpo_trainer.py` -> `GRPOTrainer.compute_advantage`
- `src/rlfusion/trainers/grpo_trainer.py` -> `GRPOTrainer.get_log_probs`
- `src/rlfusion/trainers/grpo_trainer.py` -> `GRPOTrainer.grpo_loss_batch`
- `src/rlfusion/trainers/grpo_trainer.py` -> `GRPOTrainer.generate_mask`
- `src/rlfusion/inference/hf_utils.py` -> `sample_completions_batch_hf`
- `src/rlfusion/trainers/utils.py` -> `build_full_attention_mask`

Flow:
1. `GRPOTrainer.train` expands each prompt into a group (`group_size`) and samples completions.
2. `sample_completions_batch_hf` returns token ids, decoded completions, true prompt lengths, and completion lengths.
3. Rewards are computed per completion and normalized in `compute_advantage` for stable updates.
4. A full attention mask is constructed via `build_full_attention_mask` so padding tokens do not affect log-probs.
5. `get_log_probs` computes token-level log-probs for the current policy (and reference policy if enabled).
6. `generate_mask` builds a mask over completion tokens only (prompt tokens are excluded).
7. `grpo_loss_batch` applies a PPO-style clipped objective plus the optional KL penalty.
8. Logging reports reward statistics, PPO ratios, and mask/completion token counts.

### On-Policy Distillation

Goal: train a student to match a teacher’s distribution on its own rollouts using a reverse-KL signal.

Key files and methods:
- `src/rlfusion/trainers/onpolicy_distillation_trainer.py` -> `OnPolicyDistillationTrainer.train`
- `src/rlfusion/trainers/onpolicy_distillation_trainer.py` -> `OnPolicyDistillationTrainer._build_masks`
- `src/rlfusion/trainers/onpolicy_distillation_trainer.py` -> `OnPolicyDistillationTrainer.get_log_probs`
- `src/rlfusion/inference/hf_utils.py` -> `sample_completions_batch_hf`
- `src/rlfusion/trainers/utils.py` -> `build_full_attention_mask`

Flow:
1. `OnPolicyDistillationTrainer.train` samples prompts and generates student completions.
2. Student and teacher log-probs are computed with `get_log_probs`, using a full attention mask
   from `build_full_attention_mask` to ignore prompt padding.
3. `_build_masks` creates a completion-only mask so the KL focuses on generated tokens.
   - If generation stops immediately (completion length 0), the trainer still keeps the first generated token in-mask so the sample contributes a distillation signal.
4. Reverse KL is computed per token; its negative acts as the advantage signal.
5. A PPO-style clipped update step applies the distillation signal while keeping training stable.

### Naming Conventions (Prompt vs Input vs Completion)

- `prompt_len` / `prompt_lens`: true prompt token counts (no padding).
- `input_length`: padded prompt length, which is the boundary where generation begins.
- `completion_len` / `completion_lens`: number of generated tokens before EOS or pad.

When reviewing mask logic, make sure you’re using:
- `input_length` for “where generation starts,” and
- `prompt_len` for “how long the real prompt is.”
