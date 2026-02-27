"""Минимальный пример обучения PPO."""

from stable_baselines3 import PPO

from isaac_env_final import IsaacEnv


def main() -> None:
    env = IsaacEnv(manual_reset=False)
    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)
    model.save("isaac_ppo")
    env.close()


if __name__ == "__main__":
    main()
