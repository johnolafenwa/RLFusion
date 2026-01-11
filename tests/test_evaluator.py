import json
import sys
from types import ModuleType, SimpleNamespace

import torch

from rlfusion.evaluation.evaluator import Evaluator
from rlfusion.envs import EnvBase


class DummyEnv(EnvBase):
    def get_reward(self, prediction: str) -> float:
        return 1.0 if prediction == "completion" else 0.0


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"

    def apply_chat_template(self, prompt, add_generation_prompt=True, tokenize=False):
        return " ".join(msg.get("content", "") for msg in prompt)

    def __call__(self, prompts, return_tensors="pt", padding=True):
        batch_size = len(prompts)
        input_ids = torch.ones((batch_size, 1), dtype=torch.long)
        attention_mask = torch.ones((batch_size, 1), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids, skip_special_tokens=True):
        return "completion"


class FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return FakeTokenizer()


class PaddingTokenizer(FakeTokenizer):
    def __call__(self, prompts, return_tensors="pt", padding=True):
        lengths = [max(len(prompt.split()), 1) for prompt in prompts]
        max_len = max(lengths)
        input_ids = torch.full((len(prompts), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, length in enumerate(lengths):
            input_ids[i, :length] = 2
            attention_mask[i, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class PaddingAutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return PaddingTokenizer()


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(use_cache=True)
        self.training = False

    def parameters(self):
        return iter([torch.zeros(1)])

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def generate(self, input_ids, attention_mask, **kwargs):
        batch_size, seq_len = input_ids.shape
        sequences = torch.full((batch_size, seq_len + 2), 2, dtype=torch.long)
        sequences[:, :seq_len] = input_ids
        sequences[:, -1] = 1
        return SimpleNamespace(sequences=sequences)


class FakeAutoModelForCausalLM:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return FakeModel()


def test_evaluator_runs_and_writes_metrics(tmp_path, monkeypatch):
    dataset = [
        DummyEnv(prompt=[{"role": "user", "content": "Hi"}]),
        DummyEnv(prompt=[{"role": "user", "content": "Hello"}]),
    ]

    monkeypatch.setattr("rlfusion.evaluation.evaluator.AutoModelForCausalLM", FakeAutoModelForCausalLM)
    monkeypatch.setattr("rlfusion.evaluation.evaluator.AutoTokenizer", FakeAutoTokenizer)

    evaluator = Evaluator(
        model="fake",
        dataset=dataset,
        output_dir=str(tmp_path),
        batch_size=2,
        max_new_tokens=2,
    )
    metrics = evaluator.evaluate()

    assert metrics["reward_mean"] == 1.0
    assert metrics["reward_std"] == 0.0
    assert metrics["completion_tokens_mean"] == 1.0
    assert (tmp_path / "metrics.json").exists()
    results_path = tmp_path / "results.jsonl"
    assert results_path.exists()
    lines = results_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(dataset)
    first_record = json.loads(lines[0])
    assert first_record["prompt"] == dataset[0].prompt
    assert first_record["answer"] is None
    assert first_record["generated_answer"] == "completion"
    assert first_record["reward"] == 1.0


def test_sample_completions_batch_ignores_input_padding(monkeypatch):
    dataset = [
        DummyEnv(prompt=[{"role": "user", "content": "short"}]),
        DummyEnv(prompt=[{"role": "user", "content": "long long long"}]),
    ]

    monkeypatch.setattr("rlfusion.evaluation.evaluator.AutoModelForCausalLM", FakeAutoModelForCausalLM)
    monkeypatch.setattr("rlfusion.evaluation.evaluator.AutoTokenizer", PaddingAutoTokenizer)

    evaluator = Evaluator(
        model="fake",
        dataset=dataset,
        batch_size=2,
        max_new_tokens=2,
    )

    _, _, _, completion_lengths = evaluator.sample_completions_batch(dataset)

    assert completion_lengths == [1, 1]


def test_evaluator_vllm_engine_uses_generation_outputs(monkeypatch):
    class FakeSamplingParams:
        def __init__(self, max_tokens=16, temperature=1.0, top_p=1.0, top_k=-1):
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.top_p = top_p
            self.top_k = top_k

    class FakeCompletionOutput:
        def __init__(self, text, token_ids):
            self.text = text
            self.token_ids = token_ids

    class FakeRequestOutput:
        def __init__(self):
            self.prompt_token_ids = [9, 8]
            self.outputs = [FakeCompletionOutput("completion", [2, 1])]

    class FakeLLM:
        def __init__(self, model, **kwargs):
            self.model = model

        def generate(self, prompts, sampling_params):
            return [FakeRequestOutput() for _ in prompts]

    fake_vllm = ModuleType("vllm")
    setattr(fake_vllm, "LLM", FakeLLM)
    setattr(fake_vllm, "SamplingParams", FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr("rlfusion.evaluation.evaluator.AutoTokenizer", FakeAutoTokenizer)

    dataset = [DummyEnv(prompt=[{"role": "user", "content": "Hi"}])]
    evaluator = Evaluator(
        model="fake",
        dataset=dataset,
        engine="vllm",
        max_new_tokens=2,
    )

    _, texts, prompt_lengths, completion_lengths = evaluator.sample_completions_batch(dataset)

    assert texts == ["completion"]
    assert prompt_lengths == [2]
    assert completion_lengths == [1]
