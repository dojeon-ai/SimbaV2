import gymnasium as gym
import numpy as np
from gymnasium import spaces

from scale_rl.envs.wrappers import EpisodeMonitor

#################################
#
#      Original Humanoid
#
#################################

# 14 tasks
HB_LOCOMOTION = [
    "h1hand-walk-v0",
    "h1hand-stand-v0",
    "h1hand-run-v0",
    "h1hand-reach-v0",
    "h1hand-hurdle-v0",
    "h1hand-crawl-v0",
    "h1hand-maze-v0",
    "h1hand-sit_simple-v0",
    "h1hand-sit_hard-v0",
    "h1hand-balance_simple-v0",
    "h1hand-balance_hard-v0",
    "h1hand-stair-v0",
    "h1hand-slide-v0",
    "h1hand-pole-v0",
]

# 8 tasks
HB_LOCOMOTION_SMALL = [
    "h1hand-walk-v0",
    "h1hand-stand-v0",
    "h1hand-run-v0",
    "h1hand-reach-v0",
    "h1hand-hurdle-v0",
    "h1hand-crawl-v0",
    "h1hand-sit_simple-v0",
    "h1hand-balance_simple-v0",
]

# 17 tasks
HB_MANIPULATION = [
    "h1hand-push-v0",
    "h1hand-cabinet-v0",
    "h1strong-highbar_hard-v0",  # Make hands stronger to be able to hang from the high bar
    "h1hand-door-v0",
    "h1hand-truck-v0",
    "h1hand-cube-v0",
    "h1hand-bookshelf_simple-v0",
    "h1hand-bookshelf_hard-v0",
    "h1hand-basketball-v0",
    "h1hand-window-v0",
    "h1hand-spoon-v0",
    "h1hand-kitchen-v0",
    "h1hand-package-v0",
    "h1hand-powerlift-v0",
    "h1hand-room-v0",
    "h1hand-insert_small-v0",
    "h1hand-insert_normal-v0",
]


#################################
#
#      No Hand Humanoid
#
#################################

HB_LOCOMOTION_NOHAND = [
    "h1-walk-v0",
    "h1-stand-v0",
    "h1-run-v0",
    "h1-reach-v0",
    "h1-hurdle-v0",
    "h1-crawl-v0",
    "h1-maze-v0",
    "h1-sit_simple-v0",
    "h1-sit_hard-v0",
    "h1-balance_simple-v0",
    "h1-balance_hard-v0",
    "h1-stair-v0",
    "h1-slide-v0",
    "h1-pole-v0",
]

#################################
#
#      Task Success scores
#
#################################

# 10 seeds, 10 eval envs per 1 seed
HB_RANDOM_SCORE = {
    'h1-walk-v0': 2.377,
    'h1-stand-v0': 10.545,
    'h1-run-v0': 2.02,
    'h1-reach-v0': 260.302,
    'h1-hurdle-v0': 2.214,
    'h1-crawl-v0': 272.658,
    'h1-maze-v0': 106.441,
    'h1-sit-simple-v0': 9.393,
    'h1-sit-hard-v0': 2.448,
    'h1-balance-simple-v0': 9.391,
    'h1-balance-hard-v0': 9.044,
    'h1-stair-v0': 3.112,
    'h1-slide-v0': 3.191,
    'h1-pole-v0': 20.09,
    'h1hand-push-v0': -526.8,
    'h1hand-cabinet-v0': 37.733,
    'h1strong-highbar-hard-v0': 0.178,
    'h1hand-door-v0': 2.771,
    'h1hand-truck-v0': 562.419,
    'h1hand-cube-v0': 4.787,
    'h1hand-bookshelf-simple-v0': 16.777,
    'h1hand-bookshelf-hard-v0': 14.848,
    'h1hand-basketball-v0': 8.979,
    'h1hand-window-v0': 2.713,
    'h1hand-spoon-v0': 4.661,
    'h1hand-kitchen-v0': 0.0,
    'h1hand-package-v0': -10040.932,
    'h1hand-powerlift-v0': 17.638,
    'h1hand-room-v0': 3.018,
    'h1hand-insert-small-v0': 1.653,
    'h1hand-insert-normal-v0': 1.673,
    'h1hand-walk-v0': 2.505,
    'h1hand-stand-v0': 11.973,
    'h1hand-run-v0': 1.927,
    'h1hand-reach-v0': -50.024,
    'h1hand-hurdle-v0': 2.371,
    'h1hand-crawl-v0': 278.868,
    'h1hand-maze-v0': 106.233,
    'h1hand-sit-simple-v0': 10.768,
    'h1hand-sit-hard-v0': 2.477,
    'h1hand-balance-simple-v0': 10.17,
    'h1hand-balance-hard-v0': 10.032,
    'h1hand-stair-v0': 3.161,
    'h1hand-slide-v0': 3.142,
    'h1hand-pole-v0': 19.721
}

HB_SUCCESS_SCORE = {
    "h1_walk_v0": 700.0,
    "h1_stand_v0": 800.0,
    "h1_run_v0": 700.0,
    "h1_reach_v0": 12000.0,
    "h1_hurdle_v0": 700.0,
    "h1_crawl_v0": 700.0,
    "h1_maze_v0": 1200.0,
    "h1_sit_simple_v0": 750.0,
    "h1_sit_hard_v0": 750.0,
    "h1_balance_simple_v0": 800.0,
    "h1_balance_hard_v0": 800.0,
    "h1_stair_v0": 700.0,
    "h1_slide_v0": 700.0,
    "h1_pole_v0": 700.0,
    "h1_push_v0": 700.0,
    "h1_cabinet_v0": 2500.0,
    "h1_highbar_v0": 750.0,
    "h1_door_v0": 600.0,
    "h1_truck_v0": 3000.0,
    "h1_cube_v0": 370.0,
    "h1_bookshelf_simple_v0": 2000.0,
    "h1_bookshelf_hard_v0": 2000.0,
    "h1_basketball_v0": 1200.0,
    "h1_window_v0": 650.0,
    "h1_spoon_v0": 650.0,
    "h1_kitchen_v0": 4.0,
    "h1_package_v0": 1500.0,
    "h1_powerlift_v0": 800.0,
    "h1_room_v0": 400.0,
    "h1_insert_small_v0": 350.0,
    "h1_insert_normal_v0": 350.0,
}


class HBGymnasiumVersionWrapper(gym.Wrapper):
    """
    humanoid bench originally requires gymnasium==0.29.1
    however, we are currently using  gymnasium==1.0.0a2,
    hence requiring some minor fix to the rendering function
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.task = env.unwrapped.task

    def render(self):
        return self.task._env.mujoco_renderer.render(self.task._env.render_mode)


def make_humanoid_env(
    env_name: str,
    seed: int,
    monitor_episode: bool = True,
) -> gym.Env:
    import humanoid_bench

    additional_kwargs = {}
    if env_name == "h1hand-package-v0":
        additional_kwargs = {"policy_path": None}
    env = gym.make(env_name, **additional_kwargs)
    env = HBGymnasiumVersionWrapper(env)

    if monitor_episode:
        env = EpisodeMonitor(env)

    return env
