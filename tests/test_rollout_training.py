from types import SimpleNamespace

import torch

from rlfusion.envs import EnvBase
from rlfusion.trainers.grpo_trainer import GRPOTrainer
from rlfusion.trainers.onpolicy_distillation_trainer import OnPolicyDistillationTrainer


class _RewardEnv(EnvBase):
    def get_reward(self, prediction: str) -> float:
        return 1.0 if prediction == "good" else 0.0


class _DistillEnv(EnvBase):
    def get_reward(self, prediction: str) -> float:
        return 0.0


class _TrackingTokenizer:
    pad_token_id = 99
    eos_token_id = 98

    def save_pretrained(self, path) -> None:
        _ = path


class _TrackingModel(torch.nn.Module):
    def __init__(self, bias: float = 0.5) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        self.forward_modes: list[bool] = []

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        _ = (attention_mask, use_cache)
        self.forward_modes.append(self.training)
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros((batch_size, seq_len, 2), dtype=torch.float32, device=input_ids.device)
        logits[..., 0] = self.bias
        return SimpleNamespace(logits=logits)

    def save_pretrained(self, path) -> None:
        _ = path


class _CountingScheduler:
    def __init__(self) -> None:
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1


class _CountingOptimizer(torch.optim.SGD):
    def __init__(self, params, lr: float) -> None:
        super().__init__(params, lr=lr)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


def test_grpo_train_rolls_out_in_eval_mode_and_steps_scheduler_per_update(tmp_path):
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.train_dataset = [_RewardEnv(prompt=[{"role": "user", "content": "q"}], answer="x")]
    trainer.eval_steps = None
    trainer.eval_dataset = None
    trainer.model = _TrackingModel()
    trainer.ref_model = None
    trainer.tokenizer = _TrackingTokenizer()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.lr_scheduler = _CountingScheduler()
    trainer.accelerator = None
    trainer.num_epochs = None
    trainer.num_steps = 1
    trainer.batch_size = 1
    trainer.group_size = 2
    trainer.ppo_steps = 3
    trainer.clip_eps = 0.2
    trainer.kl_penalty = 0.0
    trainer.max_error = 100.0
    trainer.invalid_penalty = 1.0
    trainer.max_grad_norm = None
    trainer.logging_steps = 10
    trainer.saving_steps = 10
    trainer.output_dir = tmp_path
    trainer.log_completions = False
    trainer.max_log_chars = 100
    trainer._wandb = None
    trainer.use_vllm = False
    trainer.vllm_enable_sleep = False
    trainer._vllm_dirty = False
    trainer.sampling_temperature = 0.7
    trainer.rollout_modes: list[bool] = []

    def _sample(envs):
        _ = envs
        trainer.rollout_modes.append(trainer.model.training)
        return (
            torch.tensor([[5, 0], [5, 1]], dtype=torch.long),
            ["good", "bad"],
            [1, 1],
            [1, 1],
            torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
        )

    trainer._sample_completions_batch_with_mask = _sample

    trainer.train()

    assert trainer.rollout_modes == [False]
    assert trainer.model.forward_modes[0] is False
    assert trainer.model.forward_modes[1:] == [True, True, True]
    assert trainer.lr_scheduler.step_count == trainer.ppo_steps


def test_grpo_train_skips_all_zero_advantage_batches_without_dirtying_vllm(tmp_path):
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.train_dataset = [_RewardEnv(prompt=[{"role": "user", "content": "q"}], answer="x")]
    trainer.eval_steps = None
    trainer.eval_dataset = None
    trainer.model = _TrackingModel()
    trainer.ref_model = None
    trainer.tokenizer = _TrackingTokenizer()
    trainer.optimizer = _CountingOptimizer(trainer.model.parameters(), lr=0.1)
    trainer.lr_scheduler = _CountingScheduler()
    trainer.accelerator = None
    trainer.num_epochs = None
    trainer.num_steps = 1
    trainer.batch_size = 1
    trainer.group_size = 2
    trainer.ppo_steps = 3
    trainer.clip_eps = 0.2
    trainer.kl_penalty = 0.0
    trainer.max_error = 100.0
    trainer.invalid_penalty = 1.0
    trainer.max_grad_norm = None
    trainer.logging_steps = 1
    trainer.saving_steps = 10
    trainer.output_dir = tmp_path
    trainer.log_completions = False
    trainer.max_log_chars = 100
    trainer._wandb = None
    trainer.use_vllm = True
    trainer.vllm_enable_sleep = False
    trainer._vllm_dirty = False
    trainer.sampling_temperature = 0.7

    def _sample(envs):
        _ = envs
        return (
            torch.tensor([[5, 0], [5, 1]], dtype=torch.long),
            ["bad", "bad"],
            [1, 1],
            [1, 1],
            torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
        )

    trainer._sample_completions_batch_with_mask = _sample

    trainer.train()

    assert trainer.optimizer.step_count == 0
    assert trainer.lr_scheduler.step_count == 0
    assert trainer._vllm_dirty is False
    assert trainer.model.forward_modes == [False]


def test_onpolicy_train_rolls_out_in_eval_mode_and_steps_scheduler_per_update(tmp_path):
    trainer = OnPolicyDistillationTrainer.__new__(OnPolicyDistillationTrainer)
    trainer.train_dataset = [_DistillEnv(prompt=[{"role": "user", "content": "q"}], answer=None)]
    trainer.eval_steps = None
    trainer.eval_dataset = None
    trainer.model = _TrackingModel()
    trainer.teacher_model = _TrackingModel(bias=-0.5)
    trainer.tokenizer = _TrackingTokenizer()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.lr_scheduler = _CountingScheduler()
    trainer.accelerator = None
    trainer.num_epochs = None
    trainer.num_steps = 1
    trainer.batch_size = 1
    trainer.ppo_steps = 2
    trainer.clip_eps = 0.2
    trainer.max_grad_norm = None
    trainer.logging_steps = 10
    trainer.saving_steps = 10
    trainer.output_dir = tmp_path
    trainer.log_completions = False
    trainer.max_log_chars = 100
    trainer._wandb = None
    trainer.use_vllm = False
    trainer.vllm_enable_sleep = False
    trainer._vllm_dirty = False
    trainer.sampling_temperature = 0.7
    trainer.rollout_modes: list[bool] = []

    for parameter in trainer.teacher_model.parameters():
        parameter.requires_grad_(False)

    def _sample(envs):
        _ = envs
        trainer.rollout_modes.append(trainer.model.training)
        return (
            torch.tensor([[5, 0]], dtype=torch.long),
            ["ok"],
            [1],
            [1],
            torch.tensor([[1, 1]], dtype=torch.long),
        )

    trainer._sample_completions_batch_with_mask = _sample

    trainer.train()

    assert trainer.rollout_modes == [False]
    assert trainer.model.forward_modes[0] is False
    assert trainer.model.forward_modes[1:] == [True, True]
    assert trainer.teacher_model.forward_modes == [False]
    assert trainer.lr_scheduler.step_count == trainer.ppo_steps


def test_grpo_vllm_sampling_wakes_syncs_and_sleeps_when_dirty(monkeypatch):
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.use_vllm = True
    trainer.vllm_enable_sleep = True
    trainer._vllm_dirty = True
    trainer._vllm_engine = object()
    trainer.model = object()
    trainer.tokenizer = object()

    def _build_vllm_sampling_params():
        return "params"

    trainer._build_vllm_sampling_params = _build_vllm_sampling_params

    events: list[str] = []

    monkeypatch.setattr(
        "rlfusion.trainers.grpo_trainer.vllm_wake_up",
        lambda engine, tags=None: events.append(f"wake:{tags or 'all'}"),
    )
    monkeypatch.setattr(
        "rlfusion.trainers.grpo_trainer.sync_model_weights_to_vllm",
        lambda model, engine: events.append("sync"),
    )
    monkeypatch.setattr(
        "rlfusion.trainers.grpo_trainer.sample_completions_batch_vllm",
        lambda **kwargs: (
            events.append("sample"),
            (torch.tensor([[1, 2]], dtype=torch.long), ["ok"], [1], [1]),
        )[1],
    )
    monkeypatch.setattr(
        "rlfusion.trainers.grpo_trainer.vllm_sleep",
        lambda engine: events.append("sleep"),
    )

    envs = [_RewardEnv(prompt=[{"role": "user", "content": "q"}], answer="x")]
    _, texts, prompt_lengths, completion_lengths = trainer.sample_completions_batch(envs)

    assert texts == ["ok"]
    assert prompt_lengths == [1]
    assert completion_lengths == [1]
    assert events == ["wake:['weights']", "sync", "wake:['kv_cache']", "sample", "sleep"]
    assert trainer._vllm_dirty is False


def test_onpolicy_vllm_sampling_wakes_syncs_and_sleeps_when_dirty(monkeypatch):
    trainer = OnPolicyDistillationTrainer.__new__(OnPolicyDistillationTrainer)
    trainer.use_vllm = True
    trainer.vllm_enable_sleep = True
    trainer._vllm_dirty = True
    trainer._vllm_engine = object()
    trainer.model = object()
    trainer.tokenizer = object()

    def _build_vllm_sampling_params():
        return "params"

    trainer._build_vllm_sampling_params = _build_vllm_sampling_params

    events: list[str] = []

    monkeypatch.setattr(
        "rlfusion.trainers.onpolicy_distillation_trainer.vllm_wake_up",
        lambda engine, tags=None: events.append(f"wake:{tags or 'all'}"),
    )
    monkeypatch.setattr(
        "rlfusion.trainers.onpolicy_distillation_trainer.sync_model_weights_to_vllm",
        lambda model, engine: events.append("sync"),
    )
    monkeypatch.setattr(
        "rlfusion.trainers.onpolicy_distillation_trainer.sample_completions_batch_vllm",
        lambda **kwargs: (
            events.append("sample"),
            (torch.tensor([[1, 2]], dtype=torch.long), ["ok"], [1], [1]),
        )[1],
    )
    monkeypatch.setattr(
        "rlfusion.trainers.onpolicy_distillation_trainer.vllm_sleep",
        lambda engine: events.append("sleep"),
    )

    envs = [_DistillEnv(prompt=[{"role": "user", "content": "q"}], answer=None)]
    _, texts, prompt_lengths, completion_lengths = trainer.sample_completions_batch(envs)

    assert texts == ["ok"]
    assert prompt_lengths == [1]
    assert completion_lengths == [1]
    assert events == ["wake:['weights']", "sync", "wake:['kv_cache']", "sample", "sleep"]
    assert trainer._vllm_dirty is False
