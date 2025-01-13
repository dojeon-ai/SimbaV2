import functools
from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import dynamic_scale

from scale_rl.agents.base_agent import BaseAgent
from scale_rl.agents.hyper_simba.hyper_simba_network import (
    HyperSimbaActor,
    HyperSimbaCritic,
    HyperSimbaDoubleCritic,
    HyperSimbaTemperature,
)
from scale_rl.agents.hyper_simba.hyper_simba_update import (
    l2normalize_network,
    update_actor,
    update_critic,
    update_target_network,
    update_temperature,
)
from scale_rl.buffers.base_buffer import Batch
from scale_rl.networks.trainer import PRNGKey, Trainer

"""
The @dataclass decorator must have `frozen=True` to ensure the instance is immutable,
allowing it to be treated as a static variable in JAX.
"""


@dataclass(frozen=True)
class HyperSimbaConfig:
    seed: int
    num_train_envs: int
    max_episode_steps: int
    normalize_observation: bool
    normalize_reward: bool
    normalized_g_max: float

    load_only_param: bool
    load_param_key: bool
    load_observation_normalizer: bool
    load_reward_normalizer: bool

    learning_rate_init: float
    learning_rate_end: float
    learning_rate_decay_rate: float
    learning_rate_decay_step: int

    actor_num_blocks: int
    actor_hidden_dim: int
    actor_scaler_init: float
    actor_scaler_scale: float
    actor_alpha_init: float
    actor_alpha_scale: float

    critic_use_cdq: bool
    critic_num_blocks: int
    critic_hidden_dim: int
    critic_num_bins: int
    critic_min_v: float
    critic_max_v: float
    critic_scaler_init: float
    critic_scaler_scale: float
    critic_alpha_init: float
    critic_alpha_scale: float

    target_tau: float

    temp_initial_value: float
    temp_target_entropy: float
    temp_target_entropy_coef: float

    gamma: float
    n_step: int


@functools.partial(
    jax.jit,
    static_argnames=(
        "observation_dim",
        "action_dim",
        "cfg",
    ),
)
def _init_hyper_simba_networks(
    observation_dim: int,
    action_dim: int,
    cfg: HyperSimbaConfig,
) -> Tuple[PRNGKey, Trainer, Trainer, Trainer, Trainer]:
    fake_observations = jnp.zeros((1, observation_dim))
    fake_actions = jnp.zeros((1, action_dim))

    rng = jax.random.PRNGKey(cfg.seed)
    rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
    compute_dtype = jnp.float32

    # When initializing the network in the flax.nn.Module class, rng_key should be passed as rngs.
    actor = Trainer.create(
        network_def=HyperSimbaActor(
            num_blocks=cfg.actor_num_blocks,
            hidden_dim=cfg.actor_hidden_dim,
            scaler_init=cfg.actor_scaler_init,
            scaler_scale=cfg.actor_scaler_scale,
            alpha_init=cfg.actor_alpha_init,
            alpha_scale=cfg.actor_alpha_scale,
            action_dim=action_dim,
            dtype=compute_dtype,
        ),
        network_inputs={"rngs": actor_key, "observations": fake_observations},
        tx=optax.adam(
            learning_rate=optax.linear_schedule(
                init_value=cfg.learning_rate_init,
                end_value=cfg.learning_rate_end,
                transition_steps=cfg.learning_rate_decay_step,
            ),
        ),
    )

    if cfg.critic_use_cdq:
        critic_network_def = HyperSimbaDoubleCritic(
            num_blocks=cfg.critic_num_blocks,
            hidden_dim=cfg.critic_hidden_dim,
            min_v=cfg.critic_min_v,
            max_v=cfg.critic_max_v,
            num_bins=cfg.critic_num_bins,
            scaler_init=cfg.critic_scaler_init,
            scaler_scale=cfg.critic_scaler_scale,
            alpha_init=cfg.critic_alpha_init,
            alpha_scale=cfg.critic_alpha_scale,
            dtype=compute_dtype,
        )
    else:
        critic_network_def = HyperSimbaCritic(
            num_blocks=cfg.critic_num_blocks,
            hidden_dim=cfg.critic_hidden_dim,
            min_v=cfg.critic_min_v,
            max_v=cfg.critic_max_v,
            num_bins=cfg.critic_num_bins,
            scaler_init=cfg.critic_scaler_init,
            scaler_scale=cfg.critic_scaler_scale,
            alpha_init=cfg.critic_alpha_init,
            alpha_scale=cfg.critic_alpha_scale,
            dtype=compute_dtype,
        )

    critic = Trainer.create(
        network_def=critic_network_def,
        network_inputs={
            "rngs": critic_key,
            "observations": fake_observations,
            "actions": fake_actions,
        },
        tx=optax.adam(
            learning_rate=optax.linear_schedule(
                init_value=cfg.learning_rate_init,
                end_value=cfg.learning_rate_end,
                transition_steps=cfg.learning_rate_decay_step,
            ),
        ),
    )

    # we set target critic's parameters identical to critic by using same rng.
    target_network_def = critic_network_def
    target_critic = Trainer.create(
        network_def=target_network_def,
        network_inputs={
            "rngs": critic_key,
            "observations": fake_observations,
            "actions": fake_actions,
        },
        tx=None,
    )

    bin_values = jnp.linspace(
        cfg.critic_min_v,
        cfg.critic_max_v,
        cfg.critic_num_bins,
        dtype=compute_dtype,
    ).reshape(1, -1)

    temperature = Trainer.create(
        network_def=HyperSimbaTemperature(cfg.temp_initial_value),
        network_inputs={
            "rngs": temp_key,
        },
        tx=optax.adam(
            learning_rate=optax.linear_schedule(
                init_value=cfg.learning_rate_init,
                end_value=cfg.learning_rate_end,
                transition_steps=cfg.learning_rate_decay_step,
            ),
        ),
    )

    # l2-normalize the network after initialization
    actor = l2normalize_network(actor)
    critic = l2normalize_network(critic)
    target_critic = l2normalize_network(target_critic)

    return rng, actor, critic, target_critic, bin_values, temperature


@jax.jit
def _sample_hyper_simba_actions(
    rng: PRNGKey,
    actor: Trainer,
    observations: jnp.ndarray,
    temperature: float = 1.0,
) -> Tuple[PRNGKey, jnp.ndarray]:
    rng, key = jax.random.split(rng)
    dist, _ = actor(observations=observations, temperature=temperature)
    actions = dist.sample(seed=key)

    return rng, actions


@functools.partial(
    jax.jit,
    static_argnames=(
        "gamma",
        "n_step",
        "critic_use_cdq",
        "critic_min_v",
        "critic_max_v",
        "critic_num_bins",
        "target_tau",
        "temp_target_entropy",
    ),
)
def _update_hyper_simba_networks(
    rng: PRNGKey,
    actor: Trainer,
    critic: Trainer,
    target_critic: Trainer,
    temperature: Trainer,
    batch: Batch,
    gamma: float,
    n_step: int,
    critic_use_cdq: bool,
    critic_min_v: float,
    critic_max_v: float,
    critic_num_bins: int,
    critic_bin_values: jnp.ndarray,
    target_tau: float,
    temp_target_entropy: float,
) -> Tuple[PRNGKey, Trainer, Trainer, Trainer, Trainer, Dict[str, float]]:
    rng, actor_key, critic_key = jax.random.split(rng, 3)

    new_actor, actor_info = update_actor(
        key=actor_key,
        actor=actor,
        critic=critic,
        temperature=temperature,
        batch=batch,
        critic_use_cdq=critic_use_cdq,
        bin_values=critic_bin_values,
    )

    new_temperature, temperature_info = update_temperature(
        temperature=temperature,
        entropy=actor_info["actor/entropy"],
        target_entropy=temp_target_entropy,
    )

    new_critic, critic_info = update_critic(
        key=critic_key,
        actor=new_actor,
        critic=critic,
        target_critic=target_critic,
        temperature=new_temperature,
        batch=batch,
        critic_use_cdq=critic_use_cdq,
        min_v=critic_min_v,
        max_v=critic_max_v,
        num_bins=critic_num_bins,
        bin_values=critic_bin_values,
        gamma=gamma,
        n_step=n_step,
    )

    new_target_critic, target_critic_info = update_target_network(
        network=new_critic,
        target_network=target_critic,
        target_tau=target_tau,
    )

    info = {
        **actor_info,
        **critic_info,
        **target_critic_info,
        **temperature_info,
    }

    return (rng, new_actor, new_critic, new_target_critic, new_temperature, info)


class HyperSimbaAgent(BaseAgent):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        cfg: HyperSimbaConfig,
    ):
        """
        An agent that randomly selects actions without training.
        Useful for collecting baseline results and for debugging purposes.
        """

        self._observation_dim = observation_space.shape[-1]
        self._action_dim = action_space.shape[-1]

        cfg["temp_target_entropy"] = cfg["temp_target_entropy_coef"] * self._action_dim

        super(HyperSimbaAgent, self).__init__(
            observation_space,
            action_space,
            cfg,
        )

        # map dictionary to dataclass
        self._cfg = HyperSimbaConfig(**cfg)

        self._init_network()

    def _init_network(self):
        (
            self._rng,
            self._actor,
            self._critic,
            self._target_critic,
            self._bin_values,
            self._temperature,
        ) = _init_hyper_simba_networks(
            self._observation_dim, self._action_dim, self._cfg
        )

    def sample_actions(
        self,
        interaction_step: int,
        prev_timestep: Dict[str, np.ndarray],
        training: bool,
    ) -> np.ndarray:
        if training:
            temperature = 1.0
        else:
            temperature = 0.0

        # current timestep observation is "next" observations from the previous timestep
        observations = jnp.asarray(prev_timestep["next_observation"])

        self._rng, actions = _sample_hyper_simba_actions(
            self._rng, self._actor, observations, temperature
        )
        actions = np.array(actions)

        return actions

    def update(self, update_step: int, batch: Dict[str, np.ndarray]) -> Dict:
        for key, value in batch.items():
            batch[key] = jnp.asarray(value)

        (
            self._rng,
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
            update_info,
        ) = _update_hyper_simba_networks(
            rng=self._rng,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            batch=batch,
            gamma=self._cfg.gamma,
            n_step=self._cfg.n_step,
            critic_use_cdq=self._cfg.critic_use_cdq,
            critic_min_v=self._cfg.critic_min_v,
            critic_max_v=self._cfg.critic_max_v,
            critic_num_bins=self._cfg.critic_num_bins,
            critic_bin_values=self._bin_values,
            target_tau=self._cfg.target_tau,
            temp_target_entropy=self._cfg.temp_target_entropy,
        )

        for key, value in update_info.items():
            if isinstance(value, dict):
                continue
            update_info[key] = float(value)

        return update_info

    def save(self, path: str) -> None:
        self._actor.save(path + "/actor")
        self._critic.save(path + "/critic")
        self._target_critic.save(path + "/target_critic")
        self._temperature.save(path + "/temperature")

    def load(self, path: str) -> None:
        only_param = self._cfg.load_only_param
        param_key = self._cfg.load_param_key

        self._actor = self._actor.load(path + "/actor", param_key, only_param)
        self._critic = self._critic.load(path + "/critic", param_key, only_param)
        self._target_critic = self._target_critic.load(
            path + "/target_critic", param_key, only_param
        )
        self._temperature = self._temperature.load(
            path + "/temperature", None, only_param
        )
