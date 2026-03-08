import importlib.util
from pathlib import Path


def _load_qwen_grpo_example():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "grpo_qwen3_4b_instruct_2507.py"
    spec = importlib.util.spec_from_file_location("grpo_qwen3_4b_instruct_2507", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reasoning_reward_scores_format_and_answer_components() -> None:
    module = _load_qwen_grpo_example()
    env = module.ReasoningRLEnv(prompt=[{"role": "user", "content": "q"}], answer="4")

    assert env.get_reward("<think>reasoning</think>") == 0.3
    assert env.get_reward("<think>reasoning</think>\n\\boxed{5}") == 0.5
    assert env.get_reward("<think>reasoning</think>\n\\boxed{4}") == 1.0
    assert env.get_reward("Final answer: \\boxed{4}") == 0.7
