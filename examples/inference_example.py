"""Minimal HF inference example using plain Transformers generate.

Expected: prints two short model completions to stdout.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlfusion.trainers.utils import (
    configure_torch_backends,
    get_device,
    resolve_attention_implementation,
)


def main() -> None:
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    device = get_device()
    if device == "cuda":
        configure_torch_backends()
        device_map = "auto"
    else:
        device_map = device

    attn_implementation = resolve_attention_implementation(device_map)
    model_kwargs: dict[str, object] = {
        "device_map": device_map,
        "attn_implementation": attn_implementation,
    }
    if device == "cuda":
        model_kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device == "mps":
        model_kwargs["dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    prompts = [
        [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Name a primary color."},
        ],
    ]
    formatted_prompts = [
        tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        for prompt in prompts
    ]
    encoded = tokenizer(formatted_prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(next(model.parameters()).device)
    attention_mask = encoded["attention_mask"].to(next(model.parameters()).device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=64,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )

    sequences = outputs.sequences
    prompt_len = input_ids.shape[1]
    for idx in range(sequences.shape[0]):
        completion_ids = sequences[idx, prompt_len:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        print(f"[{idx}] {text}")


if __name__ == "__main__":
    main()
