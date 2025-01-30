from typing import Any

import d4rl
import gymnasium as gym
import numpy as np

D4RL_MUJOCO = [
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-expert-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-medium-expert-v2",
]


def make_d4rl_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make("GymV26Environment-v0", env_id=env_name)
    env.reset(seed=seed)
    return env


def make_d4rl_dataset(env_name: str) -> list[dict[str, Any]]:
    env = make_d4rl_env(env_name, seed=0)

    # extract dataset
    dataset = env.env.env.gym_env.get_dataset()
    timesteps = []
    total_steps = dataset["rewards"].shape[0]
    for i in range(total_steps - 1):
        obs = dataset["observations"][i]
        action = dataset["actions"][i]
        reward = dataset["rewards"][i]
        terminated = dataset["terminals"][i]
        truncated = dataset["timeouts"][i]
        if terminated:
            next_obs = np.zeros_like(obs)
        elif truncated:
            continue
        else:
            next_obs = dataset["observations"][i + 1]
        timestep = {
            "observation": np.expand_dims(obs, axis=0),
            "action": np.expand_dims(action, axis=0),
            "reward": np.array([reward]),
            "terminated": np.array([terminated]),
            "truncated": np.array([truncated]),
            "next_observation": np.expand_dims(next_obs, axis=0),
        }
        timesteps.append(timestep)

    return timesteps


def get_d4rl_normalized_score(env_name: str, unnormalized_score: float) -> float:
    return 100 * d4rl.get_normalized_score(env_name, unnormalized_score)
