from rlfusion.datasets.capybara_sft import CapyBaraEnv


def test_capybara_env_reward_contract():
    env = CapyBaraEnv(prompt=[{"role": "user", "content": "hello"}])
    reward = env.get_reward("world")
    assert isinstance(reward, float)
    assert reward == 0.0
