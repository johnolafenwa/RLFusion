"""HF inference probe for Qwen/Qwen3-4B-Instruct-2507 on reasoning-rl.

The model card recommends prompting math tasks with:
  "Please reason step by step, and put your final answer within \\boxed{}."

This model is non-thinking only, so the expected behavior is a direct answer
without `<think></think>` blocks, ideally ending in a boxed final answer.
"""

from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlfusion.utils import get_boxed_answer

BOXED_INSTRUCTION = "Please reason step by step in at most three short steps, and put your final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an inference probe for Qwen3-4B-Instruct-2507.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional custom prompt. If set, the dataset is not loaded.",
    )
    parser.add_argument(
        "--expected-answer",
        type=str,
        default=None,
        help="Optional expected answer for a custom prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling and use greedy decoding.")
    return parser.parse_args()


def build_prompt(problem: str) -> str:
    return f"{problem}\n\n{BOXED_INSTRUCTION}"


def main() -> None:
    args = parse_args()

    if args.prompt is None:
        row = load_dataset("johnolafenwa/reasoning-rl", split=args.split)[args.sample_index]
        problem = str(row["problem"])
        expected_answer = str(row["answer"])
    else:
        problem = args.prompt
        expected_answer = args.expected_answer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        dtype="auto",
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    messages = [{"role": "user", "content": build_prompt(problem)}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            do_sample=not args.greedy,
            temperature=None if args.greedy else args.temperature,
            top_p=None if args.greedy else args.top_p,
            top_k=None if args.greedy else args.top_k,
            min_p=None if args.greedy else args.min_p,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    completion = tokenizer.decode(output_ids, skip_special_tokens=True)
    boxed_answer = get_boxed_answer(completion)
    contains_think_tags = "<think>" in completion or "</think>" in completion

    print("MODEL", args.model)
    if args.prompt is None:
        print("SPLIT", args.split)
        print("SAMPLE_INDEX", args.sample_index)
    else:
        print("SPLIT", "custom")
        print("SAMPLE_INDEX", -1)
    print("EXPECTED_ANSWER", expected_answer)
    print("PROMPT")
    print(build_prompt(problem))
    print("COMPLETION")
    print(completion)
    print("BOXED_ANSWER", boxed_answer)
    print("CONTAINS_THINK_TAGS", contains_think_tags)


if __name__ == "__main__":
    main()
