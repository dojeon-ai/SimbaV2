import gymnasium as gym

MUJOCO_ALL = [
    "HalfCheetah-v4",
    "Hopper-v4",
    "Walker2d-v4",
    "Ant-v4",
    "Humanoid-v4",
]

MUJOCO_RANDOM_SCORE = {
    "HalfCheetah-v4": -289.415,
    "Hopper-v4": 18.791,
    "Walker2d-v4": 2.791,
    "Ant-v4": -70.288,
    "Humanoid-v4": 120.423,
}

MUJOCO_TD3_SCORE = {
    "HalfCheetah-v4": 10574,
    "Hopper-v4": 3226,
    "Walker2d-v4": 3946,
    "Ant-v4": 3942,
    "Humanoid-v4": 5165,
}


def make_mujoco_env(
    env_name: str,
    seed: int,
) -> gym.Env:
    env = gym.make(env_name, render_mode="rgb_array")

    return env
