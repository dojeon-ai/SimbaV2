from __future__ import annotations

from typing import Any, Tuple

from gymnasium.core import ObsType
from gymnasium.spaces.utils import flatten, flatten_space
from gymnasium.vector.utils.space_utils import batch_space
from numpy.typing import NDArray

from scale_rl.envs.wrappers.vector.vector_env import (
    VectorEnv,
    VectorObservationWrapper,
)

__all__ = ["FlattenObservation"]


class FlattenObservation(VectorObservationWrapper):
    """
    This wrapper will flatten observations.
    """

    def __init__(self, env: VectorEnv):
        VectorObservationWrapper.__init__(self, env)
        _single_observation_space = flatten_space(self.observation_space)
        self.observation_space = batch_space(_single_observation_space, n=self.num_envs)

    def observations(self, observations: ObsType) -> ObsType:
        """Defines the vector observation normalization function.

        Args:
            observations: A vector observation from the environment

        Returns:
            the normalized observation
        """
        _observation = flatten(self.env.observation_space, observations)
        return _observation.reshape(self.env.num_envs, -1)

    def step(
        self, actions: NDArray[Any]
    ) -> Tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        """Steps through the environment, normalizing the reward returned."""
        obs, reward, terminated, truncated, info = super().step(actions)
        if "final_observation" in info:
            final_observations = info["final_observation"]
            final_observation_list = []
            for final_observation in final_observations:
                final_observation_list.append(
                    flatten(self.env.observation_space, final_observation)
                )
            info["final_observation"] = final_observation_list

        return obs, reward, terminated, truncated, info
