import torch

from rlfusion.trainers.sft_trainer import SFTTrainer


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        tokens = []
        for msg in messages:
            role = msg.get("role", "unknown")
            role_id = {"system": 2, "user": 3, "assistant": 4}.get(role, 9)
            tokens.append(role_id)
            content = str(msg.get("content", ""))
            tokens.extend([(ord(c) % 50) + 10 for c in content])
        if add_generation_prompt:
            tokens.append(99)
        if tokenize:
            return tokens
        return " ".join(str(t) for t in tokens)

    def pad(self, encoded, padding=True, return_tensors="pt", max_length=None):
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        if max_length is None:
            max_length = max(len(ids) for ids in input_ids)
        padded_ids = []
        padded_mask = []
        for ids, mask in zip(input_ids, attention_mask):
            pad_len = max_length - len(ids)
            padded_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_mask.append(mask + [0] * pad_len)
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
        }


def _make_trainer(mask_prompt=True):
    trainer = SFTTrainer.__new__(SFTTrainer)
    trainer.tokenizer = DummyTokenizer()
    trainer.max_seq_len = None
    trainer.mask_prompt = mask_prompt
    trainer.assistant_loss_mode = "all"
    return trainer


def test_sft_masks_user_tokens():
    trainer = _make_trainer(mask_prompt=True)
    prompt = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ]
    response = "A"

    input_ids, attention_mask, labels = trainer._build_batch([prompt], [response])
    assert attention_mask[0].sum().item() == input_ids.shape[1]

    messages = list(prompt) + [{"role": "assistant", "content": response}]
    full_ids, spans = trainer._chat_template_message_spans(messages)
    assert input_ids.shape[1] == len(full_ids)

    for start, end, role in spans:
        for idx in range(start, end):
            if attention_mask[0, idx].item() == 0:
                continue
            if role in {"system", "user"}:
                assert labels[0, idx].item() == -100
            else:
                assert labels[0, idx].item() == input_ids[0, idx].item()


def test_sft_no_mask_when_disabled():
    trainer = _make_trainer(mask_prompt=False)
    prompt = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ]
    response = "A"

    input_ids, attention_mask, labels = trainer._build_batch([prompt], [response])
    for idx in range(input_ids.shape[1]):
        if attention_mask[0, idx].item() == 0:
            assert labels[0, idx].item() == -100
        else:
            assert labels[0, idx].item() == input_ids[0, idx].item()


def test_sft_trains_all_assistant_turns_when_enabled():
    trainer = _make_trainer(mask_prompt=True)
    trainer.assistant_loss_mode = "all"

    prompt = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
        {"role": "assistant", "content": "A2"},
    ]

    input_ids, _attention_mask, labels = trainer._build_batch([prompt], [None])
    messages = list(prompt)
    full_ids, spans = trainer._chat_template_message_spans(messages)
    assert input_ids.shape[1] == len(full_ids)

    for start, end, role in spans:
        for idx in range(start, end):
            if role == "assistant":
                assert labels[0, idx].item() == input_ids[0, idx].item()
            else:
                assert labels[0, idx].item() == -100


def test_sft_trains_only_last_assistant_turn_when_enabled():
    trainer = _make_trainer(mask_prompt=True)
    trainer.assistant_loss_mode = "last"

    prompt = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
        {"role": "assistant", "content": "A2"},
    ]

    input_ids, _attention_mask, labels = trainer._build_batch([prompt], [None])
    messages = list(prompt)
    full_ids, spans = trainer._chat_template_message_spans(messages)
    assert input_ids.shape[1] == len(full_ids)

    assistant_spans = [(s, e) for s, e, r in spans if r == "assistant"]
    assert len(assistant_spans) == 2
    last_start, last_end = assistant_spans[-1]

    for idx in range(input_ids.shape[1]):
        if last_start <= idx < last_end:
            assert labels[0, idx].item() == input_ids[0, idx].item()
        else:
            assert labels[0, idx].item() == -100
