"""SFT on johnolafenwa/highschool-math-reasoning using native train/test splits."""

from __future__ import annotations

import argparse
import logging
import math

from torch.optim import AdamW

from rlfusion.datasets import HighSchoolMathReasoningSFTDataset
from rlfusion.trainers import SFTTrainer


logger = logging.getLogger(__name__)


def _unwrap_model(model: object) -> object:
    return model.module if hasattr(model, "module") else model


def _get_adamw8bit() -> object:
    try:
        from bitsandbytes.optim import AdamW8bit
    except ImportError as exc:
        raise ImportError(
            "bitsandbytes is required for AdamW8bit. Install with: uv sync --extra gpu-train --extra vllm --extra dev --extra test"
        ) from exc
    return AdamW8bit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SFT on johnolafenwa/highschool-math-reasoning."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--output-dir", type=str, default="./outputs/reasoning/reasoning_sft")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-max-samples", type=int, default=None)
    parser.add_argument("--test-max-samples", type=int, default=None)
    parser.add_argument("--saving-steps", type=int, default=100)
    parser.add_argument(
        "--save-final-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save only the final model checkpoint.",
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-8bit-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bitsandbytes AdamW8bit to reduce optimizer-state memory.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable model gradient checkpointing to reduce activation memory.",
    )
    parser.add_argument("--use-accelerate", action="store_true", help="Enable Accelerate multi-process training")
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=args.log_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    logger.setLevel(args.log_level)

    train_dataset = HighSchoolMathReasoningSFTDataset(
        split="train",
        max_samples=args.train_max_samples,
        seed=args.seed,
    )
    eval_dataset = HighSchoolMathReasoningSFTDataset(
        split="test",
        max_samples=args.test_max_samples,
        seed=args.seed,
    )

    if args.num_epochs is not None:
        steps_for_checkpoint_interval = math.ceil(len(train_dataset) / args.batch_size) * args.num_epochs
    else:
        steps_for_checkpoint_interval = args.num_steps
    saving_steps = args.num_steps + 1 if args.save_final_only else args.saving_steps
    if args.save_final_only and args.num_epochs is not None:
        saving_steps = steps_for_checkpoint_interval + 1
    optimizer_cls = _get_adamw8bit() if args.use_8bit_optimizer else AdamW
    optimizer_args = {"lr": args.lr} if args.use_8bit_optimizer else {"lr": args.lr, "foreach": False}

    trainer = SFTTrainer(
        model=args.model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_steps=args.num_steps,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        saving_steps=saving_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        max_seq_len=args.max_seq_len,
        optimizer=optimizer_cls,
        optimizer_args=optimizer_args,
        output_dir=args.output_dir,
        seed=args.seed,
        use_accelerate=args.use_accelerate,
        log_level=args.log_level,
    )
    if args.gradient_checkpointing:
        checkpointing_enable = getattr(_unwrap_model(trainer.model), "gradient_checkpointing_enable", None)
        if callable(checkpointing_enable):
            checkpointing_enable()
        else:
            logging.getLogger(__name__).warning(
                "gradient checkpointing requested, but model does not expose gradient_checkpointing_enable()."
            )
    trainer.train()


if __name__ == "__main__":
    main()
