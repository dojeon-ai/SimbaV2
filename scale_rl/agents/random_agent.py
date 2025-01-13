from typing import Dict, TypeVar

import gymnasium as gym
import numpy as np

from scale_rl.agents.base_agent import BaseAgent

Config = TypeVar("Config")


class RandomAgent(BaseAgent):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        cfg: Config,
    ):
        """
        An agent that randomly selects actions without training.
        Useful for collecting baseline results and for debugging purposes.
        """
        super(RandomAgent, self).__init__(
            observation_space,
            action_space,
            cfg,
        )

    def sample_actions(
        self,
        interaction_step: int,
        prev_timestep: Dict[str, np.ndarray],
        training: bool,
    ) -> np.ndarray:
        num_envs = prev_timestep["next_observation"].shape[0]
        actions = []
        for _ in range(num_envs):
            actions.append(self._action_space.sample())

        actions = np.stack(actions).reshape(-1)
        return actions

    def update(self, update_step: int, batch: Dict[str, np.ndarray]) -> Dict:
        update_info = {}
        return update_info

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
