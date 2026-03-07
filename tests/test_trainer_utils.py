import json

from rlfusion.trainers.utils import get_tokenizer_compat_kwargs


def test_get_tokenizer_compat_kwargs_normalizes_extra_special_tokens(tmp_path):
    tokenizer_config = {
        "tokenizer_class": "Qwen2Tokenizer",
        "extra_special_tokens": ["<foo>", "<bar>"],
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))

    compat_kwargs = get_tokenizer_compat_kwargs(str(tmp_path))

    assert compat_kwargs["extra_special_tokens"] == {
        "extra_special_token_0": "<foo>",
        "extra_special_token_1": "<bar>",
    }
    assert compat_kwargs["fix_mistral_regex"] is True


def test_get_tokenizer_compat_kwargs_enables_qwen_regex_fix_for_local_checkpoints(tmp_path):
    tokenizer_config = {
        "tokenizer_class": "Qwen2TokenizerFast",
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))

    compat_kwargs = get_tokenizer_compat_kwargs(str(tmp_path))

    assert compat_kwargs == {"fix_mistral_regex": True}


def test_get_tokenizer_compat_kwargs_returns_empty_for_remote_model_ids():
    assert get_tokenizer_compat_kwargs("Qwen/Qwen3-4B-Instruct-2507") == {}
