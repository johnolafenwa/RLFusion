from __future__ import annotations

from typing import Any, cast

import torch

from rlfusion.envs import EnvBase
from rlfusion.trainers.types import GenerateOutput


def sample_completions_batch_hf(
    *,
    model: Any,
    tokenizer: Any,
    envs: list[EnvBase],
    do_sample: bool,
    sampling_temperature: float,
    max_new_tokens: int,
    generation_args: dict[str, Any],
) -> tuple[torch.Tensor, list[str], list[int], list[int]]:
    formatted_prompts = [
        tokenizer.apply_chat_template(
            env.prompt,
            add_generation_prompt=True,
            tokenize=False,
        )
        for env in envs
    ]

    input_tokens = tokenizer(formatted_prompts, return_tensors="pt", padding=True)
    model_device = next(model.parameters()).device
    input_ids = input_tokens["input_ids"].to(model_device)
    attention_mask = input_tokens["attention_mask"].to(model_device)
    prompt_lengths = attention_mask.sum(dim=1).tolist()

    gen_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "return_dict_in_generate": True,
        "output_scores": False,
        "use_cache": False,
    }
    if do_sample:
        gen_kwargs["temperature"] = sampling_temperature
    if generation_args:
        gen_kwargs.update(generation_args)
    gen_kwargs["return_dict_in_generate"] = True

    with torch.no_grad():
        outputs = cast(GenerateOutput, model.generate(**gen_kwargs))

    generated_sequences = outputs.sequences
    ret_texts: list[str] = []
    completion_lengths: list[int] = []
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    input_length = input_ids.shape[1]
    for _i, _prompt_len in enumerate(prompt_lengths):
        output_token_ids = generated_sequences[_i]
        generated_token_ids = output_token_ids[input_length:]
        end_offset = generated_token_ids.shape[0]

        if eos_token_id is not None:
            eos_positions = (generated_token_ids == eos_token_id).nonzero(as_tuple=True)[0]
            if eos_positions.numel() > 0:
                end_offset = min(end_offset, int(eos_positions[0]))

        if pad_token_id is not None:
            pad_positions = (generated_token_ids == pad_token_id).nonzero(as_tuple=True)[0]
            if pad_positions.numel() > 0:
                end_offset = min(end_offset, int(pad_positions[0]))

        completion_token_ids = generated_token_ids[:end_offset]
        ret_texts.append(tokenizer.decode(completion_token_ids, skip_special_tokens=True))
        completion_lengths.append(max(end_offset, 0))

    return generated_sequences, ret_texts, prompt_lengths, completion_lengths
