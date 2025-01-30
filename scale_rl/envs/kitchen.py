import gymnasium as gym

from scale_rl.envs.wrappers import DoNotTerminate

KITCHEN_ALL = [
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
]


class KitchenGymnasiumVersionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.unwrapped_env = env.unwrapped

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward > 0.0:
            info["success"] = True
        return obs, reward, terminated, truncated, info


def make_kitchen_env(
    env_name: str,
    seed: int,
    render_mode="rgb_array",
    render_width=256,
    render_height=256,
) -> gym.Env:
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    env = gym.make(
        "FrankaKitchen-v1",
        tasks_to_complete=[env_name],
        render_mode=render_mode,
        width=render_width,
        height=render_height,
    )
    env = KitchenGymnasiumVersionWrapper(env)

    return env
