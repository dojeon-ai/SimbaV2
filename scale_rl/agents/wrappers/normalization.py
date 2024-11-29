from typing import Dict

import numpy as np

from scale_rl.agents.base_agent import AgentWrapper, BaseAgent
from scale_rl.agents.wrappers.utils import RunningMeanStd


class ObservationNormalizer(AgentWrapper):
    """
    This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Observation statistics is updated only on sample_actions with training==True
    """

    def __init__(self, agent: BaseAgent, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            agent (BaseAgent): The agent to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        AgentWrapper.__init__(self, agent)

        self.obs_rms = RunningMeanStd(
            shape=self.agent._observation_space.shape,
            dtype=np.float32,
        )
        self.epsilon = epsilon

    def _normalize(self, observations):
        return (observations - self.obs_rms.mean) / np.sqrt(
            self.obs_rms.var + self.epsilon
        )

    def sample_actions(
        self,
        interaction_step: int,
        prev_timestep: Dict[str, np.ndarray],
        training: bool,
    ) -> np.ndarray:
        """
        Defines the sample action function with normalized observation.
        """

        observations = prev_timestep["next_observation"]
        if training:
            self.obs_rms.update(observations)
        prev_timestep["next_observation"] = self._normalize(observations)

        return self.agent.sample_actions(
            interaction_step=interaction_step,
            prev_timestep=prev_timestep,
            training=training,
        )

    def update(self, update_step: int, batch: Dict[str, np.ndarray]):
        batch["observation"] = self._normalize(batch["observation"])
        batch["next_observation"] = self._normalize(batch["next_observation"])
        return self.agent.update(
            update_step=update_step,
            batch=batch,
        )


class StreamRewardScaler(AgentWrapper):
    """
    This wrapper will scale rewards using the variance of a running estimate of the discounted returns. In policy gradient methods, the update rule often involves the term ∇log ⁡π(a|s)⋅G_t, where G_t is the return from time t. Scaling G_t to have unit variance can be an effective variance reduction technique.

    Return statistics is updated only on sample_actions with training == True
    """

    def __init__(self, agent: BaseAgent, gamma: float, epsilon: float = 1e-8):
        """This wrapper will scale rewards using the variance of a running estimate of the discounted returns.

        Args:
            agent (BaseAgent): The agent to apply the wrapper
            gamma: Discount factor
            epsilon: A stability parameter that is used when scaling the rewards.
        """
        AgentWrapper.__init__(self, agent)
        self.G = 0.0  # running estimate of the discounted return
        self.G_rms = RunningMeanStd(
            shape=1,
            dtype=np.float32,
        )
        self.gamma = gamma
        self.epsilon = epsilon

    def _scale_reward(self, rewards):
        """
        Stream-ScaleReward: https://arxiv.org/abs/2410.14606
        """
        return rewards / np.sqrt(self.G_rms.var + self.epsilon)

    def sample_actions(
        self,
        interaction_step: int,
        prev_timestep: Dict[str, np.ndarray],
        training: bool,
    ) -> np.ndarray:
        """
        Defines the sample action function with updating statistics.
        """
        if training:
            self.G = (
                self.gamma * (1 - prev_timestep["terminated"]) * self.G
                + prev_timestep["reward"]
            )
            self.G_rms.update(self.G)

        return self.agent.sample_actions(
            interaction_step=interaction_step,
            prev_timestep=prev_timestep,
            training=training,
        )

    def update(self, update_step: int, batch: Dict[str, np.ndarray]):
        batch["reward"] = self._scale_reward(batch["reward"])
        return self.agent.update(
            update_step=update_step,
            batch=batch,
        )
