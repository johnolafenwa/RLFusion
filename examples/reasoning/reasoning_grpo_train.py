"""GRPO on johnolafenwa/highschool-math-reasoning-rl with strict reasoning-format reward.

Expected answer format:
  <think>...</think>
  \boxed{answer}

Reward is hard-gated on a single top-level think block and a single terminal
boxed answer. Correctness is then verified with Math-Verify instead of raw
string matching.
"""

from __future__ import annotations

import argparse
import logging
import math

from rlfusion.datasets import HighSchoolMathReasoningRLDataset
from rlfusion.trainers.grpo_trainer import GRPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GRPO training on johnolafenwa/highschool-math-reasoning-rl."
    )
    parser.add_argument(
        "--sft-checkpoint",
        type=str,
        default="./outputs/reasoning/reasoning_sft/final",
        help="Path to the SFT checkpoint to use as the GRPO base.",
    )
    parser.add_argument("--output-dir", type=str, default="./outputs/reasoning/reasoning_grpo")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--saving-steps", type=int, default=50)
    parser.add_argument(
        "--save-final-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save only the final model checkpoint.",
    )
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--ppo-steps", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--sampling-temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-max-samples", type=int, default=None)
    parser.add_argument("--test-max-samples", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument("--use-accelerate", action="store_true", help="Enable Accelerate multi-process training")
    parser.add_argument("--max-error", type=float, default=100.0)
    parser.add_argument("--invalid-penalty", type=float, default=1.0)
    parser.add_argument("--log-completions", action="store_true")
    parser.add_argument("--max-log-chars", type=int, default=320)
    parser.add_argument("--use-base-kl", action="store_true", help="Enable KL penalty with a reference model.")
    parser.add_argument("--kl-penalty", type=float, default=0.0)
    parser.add_argument(
        "--use-vllm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use colocated vLLM generation. Defaults to enabled on CUDA and disabled otherwise.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.5,
        help="vLLM GPU memory utilization (0-1).",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallel size. Keep this at 1 when using --use-accelerate.",
    )
    parser.add_argument(
        "--vllm-enable-sleep",
        action="store_true",
        help="Enable vLLM sleep mode between generations to free GPU memory during training.",
    )
    parser.add_argument(
        "--vllm-attention-backend",
        type=str,
        default=None,
        choices=("FLASH_ATTN", "TRITON_ATTN", "FLEX_ATTENTION", "FLASHINFER"),
        help="Explicit vLLM attention backend override.",
    )
    parser.add_argument(
        "--vllm-flash-attn-version",
        type=int,
        default=None,
        choices=(2, 3, 4),
        help="Optional FlashAttention version when using --vllm-attention-backend FLASH_ATTN.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_dataset = HighSchoolMathReasoningRLDataset(
        train=True,
        max_samples=args.train_max_samples,
        seed=args.seed,
    )
    eval_dataset = HighSchoolMathReasoningRLDataset(
        train=False,
        max_samples=args.test_max_samples,
        seed=args.seed,
    )

    if args.num_epochs is not None:
        steps_for_checkpoint_interval = math.ceil(len(train_dataset) / args.batch_size) * args.num_epochs
    else:
        steps_for_checkpoint_interval = args.num_steps
    saving_steps = args.saving_steps
    if args.save_final_only:
        saving_steps = steps_for_checkpoint_interval + 1

    vllm_args = {
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "tensor_parallel_size": args.vllm_tensor_parallel_size,
    }
    if args.vllm_attention_backend is not None:
        attention_config: dict[str, object] = {"backend": args.vllm_attention_backend}
        if args.vllm_flash_attn_version is not None:
            attention_config["flash_attn_version"] = args.vllm_flash_attn_version
        vllm_args["attention_config"] = attention_config

    trainer = GRPOTrainer(
        model=args.sft_checkpoint,
        train_dataset=train_dataset,
        num_steps=args.num_steps,
        num_epochs=args.num_epochs,
        saving_steps=saving_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_dataset=eval_dataset,
        sampling_temperature=args.sampling_temperature,
        kl_penalty=args.kl_penalty if args.use_base_kl else 0.0,
        output_dir=args.output_dir,
        generation_args={"top_p": args.top_p},
        optimizer_args={"lr": args.learning_rate},
        batch_size=args.batch_size,
        group_size=args.group_size,
        ppo_steps=args.ppo_steps,
        clip_eps=args.clip_eps,
        max_new_tokens=args.max_new_tokens,
        max_grad_norm=args.max_grad_norm,
        log_completions=args.log_completions,
        max_log_chars=args.max_log_chars,
        max_error=args.max_error,
        invalid_penalty=args.invalid_penalty,
        enable_wandb=False,
        seed=args.seed,
        use_accelerate=args.use_accelerate,
        use_vllm=args.use_vllm,
        vllm_args=vllm_args,
        vllm_enable_sleep=args.vllm_enable_sleep,
        log_level=args.log_level,
    )

    trainer.train()


if __name__ == "__main__":
    main()
