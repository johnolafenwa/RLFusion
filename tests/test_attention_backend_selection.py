import rlfusion.trainers.utils as trainer_utils


def _patch_cuda(monkeypatch, capabilities):
    monkeypatch.setattr(trainer_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer_utils.torch.cuda, "device_count", lambda: len(capabilities))
    monkeypatch.setattr(
        trainer_utils.torch.cuda,
        "get_device_capability",
        lambda index=0: capabilities[index],
    )


def test_resolve_attention_implementation_prefers_flash_attention_2_on_ada(monkeypatch):
    _patch_cuda(monkeypatch, [(8, 9), (8, 9)])
    monkeypatch.setattr(trainer_utils, "is_flash_attn_3_available", lambda: False)
    monkeypatch.setattr(trainer_utils, "is_flash_attn_2_available", lambda: True)
    monkeypatch.setattr(trainer_utils, "is_kernels_available", lambda: False)
    monkeypatch.setattr(trainer_utils, "_flash_attention_backend_usable", lambda implementation: True)

    assert trainer_utils.resolve_attention_implementation({"": 0}) == "flash_attention_2"


def test_resolve_attention_implementation_prefers_flash_attention_3_on_hopper(monkeypatch):
    _patch_cuda(monkeypatch, [(9, 0), (9, 0)])
    monkeypatch.setattr(trainer_utils, "is_flash_attn_3_available", lambda: True)
    monkeypatch.setattr(trainer_utils, "is_flash_attn_2_available", lambda: True)
    monkeypatch.setattr(trainer_utils, "is_kernels_available", lambda: False)
    monkeypatch.setattr(trainer_utils, "_flash_attention_backend_usable", lambda implementation: True)

    assert trainer_utils.resolve_attention_implementation("auto") == "flash_attention_3"


def test_resolve_attention_implementation_uses_kernels_fallback(monkeypatch):
    _patch_cuda(monkeypatch, [(8, 9)])
    monkeypatch.setattr(trainer_utils, "is_flash_attn_3_available", lambda: False)
    monkeypatch.setattr(trainer_utils, "is_flash_attn_2_available", lambda: False)
    monkeypatch.setattr(trainer_utils, "is_kernels_available", lambda: True)
    monkeypatch.setattr(trainer_utils, "_flash_attention_backend_usable", lambda implementation: True)

    assert trainer_utils.resolve_attention_implementation("auto") == "flash_attention_2"


def test_resolve_attention_implementation_falls_back_to_sdpa_without_flash_backend(monkeypatch):
    _patch_cuda(monkeypatch, [(8, 9)])
    monkeypatch.setattr(trainer_utils, "is_flash_attn_3_available", lambda: False)
    monkeypatch.setattr(trainer_utils, "is_flash_attn_2_available", lambda: False)
    monkeypatch.setattr(trainer_utils, "is_kernels_available", lambda: False)

    assert trainer_utils.resolve_attention_implementation("auto") == "sdpa"


def test_resolve_attention_implementation_honors_env_override(monkeypatch):
    monkeypatch.setenv("RLFUSION_ATTN_IMPLEMENTATION", "sdpa")
    _patch_cuda(monkeypatch, [(9, 0)])
    monkeypatch.setattr(trainer_utils, "is_flash_attn_3_available", lambda: True)
    monkeypatch.setattr(trainer_utils, "is_flash_attn_2_available", lambda: True)
    monkeypatch.setattr(trainer_utils, "is_kernels_available", lambda: True)
    monkeypatch.setattr(trainer_utils, "_flash_attention_backend_usable", lambda implementation: True)

    assert trainer_utils.resolve_attention_implementation("auto") == "sdpa"


def test_resolve_attention_implementation_falls_back_to_sdpa_when_flash_attn_import_is_broken(monkeypatch):
    _patch_cuda(monkeypatch, [(8, 9)])
    monkeypatch.setattr(trainer_utils, "is_flash_attn_3_available", lambda: False)
    monkeypatch.setattr(trainer_utils, "is_flash_attn_2_available", lambda: True)
    monkeypatch.setattr(trainer_utils, "is_kernels_available", lambda: False)
    monkeypatch.setattr(trainer_utils, "_flash_attention_backend_usable", lambda implementation: False)

    assert trainer_utils.resolve_attention_implementation("auto") == "sdpa"
