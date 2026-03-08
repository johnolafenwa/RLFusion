"""HF generation helpers with consistent prompt/completion bookkeeping."""

from __future__ import annotations

from typing import Any, Literal, overload, cast

import torch

from rlfusion.envs import EnvBase
from rlfusion.trainers.types import AttentionMask, GenerateOutput, TokenIds

CompletionBatch = tuple[TokenIds, list[str], list[int], list[int]]
CompletionBatchWithMask = tuple[TokenIds, list[str], list[int], list[int], AttentionMask]


def _resolve_chat_stop_token_ids(tokenizer: Any) -> list[int]:
    stop_token_ids: list[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int) and eos_token_id >= 0:
        stop_token_ids.append(eos_token_id)

    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert_tokens_to_ids):
        im_end_token_id = convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_token_id, int) and im_end_token_id >= 0 and im_end_token_id not in stop_token_ids:
            stop_token_ids.append(im_end_token_id)

    return stop_token_ids


def _merge_stop_token_ids(existing: Any, defaults: list[int]) -> list[int]:
    stop_token_ids: list[int] = []

    def _extend(value: Any) -> None:
        if isinstance(value, int):
            if value >= 0 and value not in stop_token_ids:
                stop_token_ids.append(value)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                if isinstance(item, int) and item >= 0 and item not in stop_token_ids:
                    stop_token_ids.append(item)

    _extend(existing)
    _extend(defaults)
    return stop_token_ids


@overload
def sample_completions_batch_hf(
    *,
    model: Any,
    tokenizer: Any,
    envs: list[EnvBase],
    do_sample: bool,
    sampling_temperature: float,
    max_new_tokens: int,
    generation_args: dict[str, Any],
    return_attention_mask: Literal[False] = False,
) -> CompletionBatch: ...


@overload
def sample_completions_batch_hf(
    *,
    model: Any,
    tokenizer: Any,
    envs: list[EnvBase],
    do_sample: bool,
    sampling_temperature: float,
    max_new_tokens: int,
    generation_args: dict[str, Any],
    return_attention_mask: Literal[True],
) -> CompletionBatchWithMask: ...


def sample_completions_batch_hf(
    *,
    model: Any,
    tokenizer: Any,
    envs: list[EnvBase],
    do_sample: bool,
    sampling_temperature: float,
    max_new_tokens: int,
    generation_args: dict[str, Any],
    return_attention_mask: bool = False,
) -> CompletionBatch | CompletionBatchWithMask:
    generate_model = model.module if hasattr(model, "module") else model

    formatted_prompts = [
        tokenizer.apply_chat_template(
            env.prompt,
            add_generation_prompt=True,
            tokenize=False,
        )
        for env in envs
    ]

    # Decoder-only generation needs left padding for correct batched prompts.
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        input_tokens = tokenizer(formatted_prompts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = original_padding_side
    model_device = next(generate_model.parameters()).device
    input_ids = input_tokens["input_ids"].to(model_device)
    attention_mask = input_tokens["attention_mask"].to(model_device)
    input_length = int(input_ids.shape[1])
    # True prompt lengths (no padding); padded length is input_length.
    prompt_lengths = attention_mask.sum(dim=1).tolist()

    gen_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "return_dict_in_generate": True,
        "output_scores": False,
        # Rollout generation is no-grad, so enable KV caching for speed.
        "use_cache": True,
    }
    if do_sample:
        gen_kwargs["temperature"] = sampling_temperature
    if generation_args:
        gen_kwargs.update(generation_args)
    stop_token_ids = _merge_stop_token_ids(
        gen_kwargs.get("eos_token_id"),
        _resolve_chat_stop_token_ids(tokenizer),
    )
    if stop_token_ids:
        gen_kwargs["eos_token_id"] = stop_token_ids[0] if len(stop_token_ids) == 1 else stop_token_ids
    gen_kwargs["return_dict_in_generate"] = True

    with torch.no_grad():
        outputs = cast(GenerateOutput, generate_model.generate(**gen_kwargs))

    generated_sequences = outputs.sequences
    ret_texts: list[str] = []
    completion_lengths: list[int] = []
    stop_token_ids = _resolve_chat_stop_token_ids(tokenizer)
    pad_token_id = tokenizer.pad_token_id

    input_length = int(input_ids.shape[1])
    for _i, _prompt_len in enumerate(prompt_lengths):
        output_token_ids = generated_sequences[_i]
        generated_token_ids = output_token_ids[input_length:]
        end_offset = generated_token_ids.shape[0]

        for stop_token_id in stop_token_ids:
            stop_positions = (generated_token_ids == stop_token_id).nonzero(as_tuple=True)[0]
            if stop_positions.numel() > 0:
                end_offset = min(end_offset, int(stop_positions[0]))

        if pad_token_id is not None:
            pad_positions = (generated_token_ids == pad_token_id).nonzero(as_tuple=True)[0]
            if pad_positions.numel() > 0:
                end_offset = min(end_offset, int(pad_positions[0]))

        completion_token_ids = generated_token_ids[:end_offset]
        ret_texts.append(tokenizer.decode(completion_token_ids, skip_special_tokens=True))
        completion_lengths.append(max(end_offset, 0))

    if return_attention_mask:
        full_attention_mask = torch.zeros_like(generated_sequences, dtype=torch.long)
        full_attention_mask[:, :input_length] = attention_mask.long()
        for idx, completion_len in enumerate(completion_lengths):
            end = min(input_length + int(completion_len), generated_sequences.shape[1])
            if end > input_length:
                full_attention_mask[idx, input_length:end] = 1
        return generated_sequences, ret_texts, prompt_lengths, completion_lengths, full_attention_mask
    return generated_sequences, ret_texts, prompt_lengths, completion_lengths
