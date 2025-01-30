from typing import Dict

import numpy as np
from flax.training import checkpoints

from scale_rl.agents.base_agent import AgentWrapper, BaseAgent
from scale_rl.agents.wrappers.utils import RunningMeanStd


class ObservationNormalizer(AgentWrapper):
    """
    This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Observation statistics is updated only on sample_actions with training==True
    """

    def __init__(self, agent: BaseAgent, load_rms: bool = True, epsilon: float = 1e-8):
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
        self.load_rms = load_rms
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

    def save(self, path: str) -> None:
        """
        Save both the wrapped agent and this wrapper's running statistics.
        """
        # 1. Save the underlying agent’s checkpoint
        self.agent.save(path)

        # 2. Save the wrapper’s statistics in a separate file
        ckpt = {
            "obs_rms_mean": self.obs_rms.mean,
            "obs_rms_var": self.obs_rms.var,
            "obs_rms_count": self.obs_rms.count,
        }
        checkpoints.save_checkpoint(
            ckpt_dir=path + "/obs_norm",
            target=ckpt,
            step=0,
            overwrite=True,
            keep=1,
        )

    def load(self, path: str):
        """
        Load both the wrapped agent and the wrapper’s running statistics.
        """
        # 1. Load the underlying agent
        self.agent.load(path)

        # 2. Load the wrapper’s statistics
        if self.load_rms:
            ckpt = checkpoints.restore_checkpoint(
                ckpt_dir=path + "/obs_norm", target=None
            )
            self.obs_rms.mean = ckpt["obs_rms_mean"]
            self.obs_rms.var = ckpt["obs_rms_var"]
            self.obs_rms.count = ckpt["obs_rms_count"]


class RewardNormalizer(AgentWrapper):
    """
    This wrapper will scale rewards using the variance of a running estimate of the discounted returns. In policy gradient methods, the update rule often involves the term ∇log ⁡π(a|s)⋅G_t, where G_t is the return from time t. Scaling G_t to have unit variance can be an effective variance reduction technique.

    Return statistics is updated only on sample_actions with training == True
    """

    def __init__(
        self,
        agent: BaseAgent,
        gamma: float,
        g_max: float = 10.0,
        load_rms: bool = True,
        epsilon: float = 1e-8,
    ):
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
        self.G_r_max = 0.0  # running-max
        self.gamma = gamma
        self.g_max = g_max
        self.load_rms = load_rms
        self.epsilon = epsilon

    def _scale_reward(self, rewards):
        """
        https://gymnasium.farama.org/api/wrappers/reward_wrappers/#gymnasium.wrappers.NormalizeReward
        """
        var_denominator = np.sqrt(self.G_rms.var + self.epsilon)
        min_required_denominator = self.G_r_max / self.g_max
        denominator = max(var_denominator, min_required_denominator)

        return rewards / denominator

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
            reward = prev_timestep["reward"]
            terminated = prev_timestep["terminated"]
            self.G = self.gamma * (1 - terminated) * self.G + reward
            self.G_rms.update(self.G)
            self.G_r_max = max(self.G_r_max, max(abs(self.G)))

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

    def save(self, path: str) -> None:
        """
        Save both the wrapped agent and this wrapper's running statistics.
        """
        # 1. Save the underlying agent’s checkpoint
        self.agent.save(path)

        # 2. Save the wrapper’s statistics in a separate file
        ckpt = {
            "G": self.G,
            "G_rms_mean": self.G_rms.mean,
            "G_rms_var": self.G_rms.var,
            "G_rms_count": self.G_rms.count,
            "G_r_max": self.G_r_max,
        }
        checkpoints.save_checkpoint(
            ckpt_dir=path + "/rew_norm",
            target=ckpt,
            step=0,
            overwrite=True,
            keep=1,
        )

    def load(self, path: str):
        """
        Load both the wrapped agent and the wrapper’s running statistics.
        """
        # 1. Load the underlying agent
        self.agent.load(path)

        # 2. Load the wrapper’s statistics
        if self.load_rms:
            ckpt = checkpoints.restore_checkpoint(
                ckpt_dir=path + "/rew_norm", target=None
            )
            self.G = ckpt["G"]
            self.G_rms.mean = ckpt["G_rms_mean"]
            self.G_rms.var = ckpt["G_rms_var"]
            self.G_rms.count = ckpt["G_rms_count"]
            self.G_r_max = ckpt["G_r_max"]
