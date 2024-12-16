import functools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import dynamic_scale

from scale_rl.agents.base_agent import BaseAgent
from scale_rl.agents.hyper_sac_dev.hyper_sac_dev_network import (
    HyperSACDevActor,
    HyperSACDevCategoricalCritic,
    HyperSACDevCategoricalDoubleCritic,
    HyperSACDevClippedDoubleCritic,
    HyperSACDevCritic,
    HyperSACDevTemperature,
)
from scale_rl.agents.hyper_sac_dev.hyper_sac_dev_update import (
    l2normalize_network,
    update_actor,
    update_actor_with_categorical_critic,
    update_categorical_critic,
    update_critic,
    update_target_network,
    update_temperature,
)
from scale_rl.buffers.base_buffer import Batch
from scale_rl.networks.metrics import (
    get_critic_featdot,
    get_dormant_ratio,
    get_feature_norm,
    get_rank,
    get_scaler_stat,
    get_weight_norm,
)
from scale_rl.networks.trainer import PRNGKey, Trainer

"""
The @dataclass decorator must have `frozen=True` to ensure the instance is immutable,
allowing it to be treated as a static variable in JAX.
"""


@dataclass(frozen=True)
class HyperSACDevConfig:
    seed: int
    num_train_envs: int
    max_episode_steps: int
    normalize_observation: bool
    normalize_reward: bool
    normalized_g_max: float

    actor_num_blocks: int
    actor_hidden_dim: int
    actor_learning_rate_init: float
    actor_learning_rate_end: float
    actor_learning_rate_decay_rate: float
    actor_learning_rate_decay_step: int
    actor_weight_decay: float
    actor_input_process_type: str
    actor_input_projection_type: str
    actor_input_projection_constant: float
    actor_scale_input_dense: bool
    actor_alpha_init: float
    actor_alpha_scale: float
    actor_output_separate_mean_std: bool
    actor_output_project_in: bool
    actor_output_hidden_dim: int
    actor_output_use_scaler: bool
    actor_output_non_linear: bool
    actor_output_use_bias: bool

    critic_num_blocks: int
    critic_hidden_dim: int
    critic_learning_rate_init: float
    critic_learning_rate_end: float
    critic_learning_rate_decay_rate: float
    critic_learning_rate_decay_step: int
    critic_weight_decay: float
    critic_use_cdq: bool
    critic_input_process_type: str
    critic_input_projection_type: str
    critic_input_projection_constant: float
    critic_scale_input_dense: bool
    critic_alpha_init: float
    critic_alpha_scale: float
    critic_output_project_in: bool
    critic_output_hidden_dim: int
    critic_output_use_scaler: bool
    critic_output_non_linear: bool
    critic_output_use_bias: bool

    critic_use_categorical: bool
    critic_num_bins: int
    categorical_min_v: float
    categorical_max_v: float

    temp_target_entropy: float
    temp_target_entropy_coef: float
    temp_initial_value: float
    temp_learning_rate: float
    temp_weight_decay: float

    target_copy_type: str
    target_copy_every: int
    target_tau: float
    target_normalize_weight: bool
    gamma: float
    n_step: int

    mixed_precision: bool


@functools.partial(
    jax.jit,
    static_argnames=(
        "observation_dim",
        "action_dim",
        "cfg",
    ),
)
def _init_hyper_sac_dev_networks(
    observation_dim: int,
    action_dim: int,
    cfg: HyperSACDevConfig,
) -> Tuple[PRNGKey, Trainer, Trainer, Trainer, Trainer]:
    fake_observations = jnp.zeros((1, observation_dim))
    fake_actions = jnp.zeros((1, action_dim))

    rng = jax.random.PRNGKey(cfg.seed)
    rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
    compute_dtype = jnp.float16 if cfg.mixed_precision else jnp.float32

    # When initializing the network in the flax.nn.Module class, rng_key should be passed as rngs.
    actor = Trainer.create(
        network_def=HyperSACDevActor(
            num_blocks=cfg.actor_num_blocks,
            hidden_dim=cfg.actor_hidden_dim,
            action_dim=action_dim,
            input_process_type=cfg.actor_input_process_type,
            input_projection_type=cfg.actor_input_projection_type,
            input_projection_constant=cfg.actor_input_projection_constant,
            scale_input_dense=cfg.actor_scale_input_dense,
            alpha_init=cfg.actor_alpha_init,
            alpha_scale=cfg.actor_alpha_scale,
            dtype=compute_dtype,
            output_separate_mean_std=cfg.actor_output_separate_mean_std,
            output_project_in=cfg.actor_output_project_in,
            output_hidden_dim=cfg.actor_output_hidden_dim,
            output_use_scaler=cfg.actor_output_use_scaler,
            output_non_linear=cfg.actor_output_non_linear,
            output_use_bias=cfg.actor_output_use_bias,
        ),
        network_inputs={"rngs": actor_key, "observations": fake_observations},
        tx=optax.adamw(
            learning_rate=optax.linear_schedule(
                init_value=cfg.actor_learning_rate_init,
                end_value=cfg.actor_learning_rate_end,
                transition_steps=cfg.actor_learning_rate_decay_step,
            ),
            weight_decay=cfg.actor_weight_decay,
        ),
        dynamic_scale=dynamic_scale.DynamicScale() if cfg.mixed_precision else None,
    )

    if cfg.critic_use_cdq:
        if cfg.critic_use_categorical:
            critic_network_def = HyperSACDevCategoricalDoubleCritic(
                num_blocks=cfg.critic_num_blocks,
                hidden_dim=cfg.critic_hidden_dim,
                input_process_type=cfg.critic_input_process_type,
                input_projection_type=cfg.critic_input_projection_type,
                input_projection_constant=cfg.critic_input_projection_constant,
                scale_input_dense=cfg.critic_scale_input_dense,
                alpha_init=cfg.critic_alpha_init,
                alpha_scale=cfg.critic_alpha_scale,
                dtype=compute_dtype,
                output_project_in=cfg.critic_output_project_in,
                output_hidden_dim=cfg.critic_output_hidden_dim,
                output_use_scaler=cfg.critic_output_use_scaler,
                output_non_linear=cfg.critic_output_non_linear,
                output_use_bias=cfg.critic_output_use_bias,
                num_bins=cfg.critic_num_bins,
            )
        else:
            critic_network_def = HyperSACDevClippedDoubleCritic(
                num_blocks=cfg.critic_num_blocks,
                hidden_dim=cfg.critic_hidden_dim,
                input_process_type=cfg.critic_input_process_type,
                input_projection_type=cfg.critic_input_projection_type,
                input_projection_constant=cfg.critic_input_projection_constant,
                scale_input_dense=cfg.critic_scale_input_dense,
                alpha_init=cfg.critic_alpha_init,
                alpha_scale=cfg.critic_alpha_scale,
                dtype=compute_dtype,
                output_project_in=cfg.critic_output_project_in,
                output_hidden_dim=cfg.critic_output_hidden_dim,
                output_use_scaler=cfg.critic_output_use_scaler,
                output_non_linear=cfg.critic_output_non_linear,
                output_use_bias=cfg.critic_output_use_bias,
            )
    else:
        if cfg.critic_use_categorical:
            critic_network_def = HyperSACDevCategoricalCritic(
                num_blocks=cfg.critic_num_blocks,
                hidden_dim=cfg.critic_hidden_dim,
                input_process_type=cfg.critic_input_process_type,
                input_projection_type=cfg.critic_input_projection_type,
                input_projection_constant=cfg.critic_input_projection_constant,
                scale_input_dense=cfg.critic_scale_input_dense,
                alpha_init=cfg.critic_alpha_init,
                alpha_scale=cfg.critic_alpha_scale,
                dtype=compute_dtype,
                output_project_in=cfg.critic_output_project_in,
                output_hidden_dim=cfg.critic_output_hidden_dim,
                output_use_scaler=cfg.critic_output_use_scaler,
                output_non_linear=cfg.critic_output_non_linear,
                output_use_bias=cfg.critic_output_use_bias,
                num_bins=cfg.critic_num_bins,
            )
        else:
            critic_network_def = HyperSACDevCritic(
                num_blocks=cfg.critic_num_blocks,
                hidden_dim=cfg.critic_hidden_dim,
                input_process_type=cfg.critic_input_process_type,
                input_projection_type=cfg.critic_input_projection_type,
                input_projection_constant=cfg.critic_input_projection_constant,
                scale_input_dense=cfg.critic_scale_input_dense,
                alpha_init=cfg.critic_alpha_init,
                alpha_scale=cfg.critic_alpha_scale,
                dtype=compute_dtype,
                output_project_in=cfg.critic_output_project_in,
                output_hidden_dim=cfg.critic_output_hidden_dim,
                output_use_scaler=cfg.critic_output_use_scaler,
                output_non_linear=cfg.critic_output_non_linear,
                output_use_bias=cfg.critic_output_use_bias,
            )

    critic = Trainer.create(
        network_def=critic_network_def,
        network_inputs={
            "rngs": critic_key,
            "observations": fake_observations,
            "actions": fake_actions,
        },
        tx=optax.adamw(
            learning_rate=optax.linear_schedule(
                init_value=cfg.critic_learning_rate_init,
                end_value=cfg.critic_learning_rate_end,
                transition_steps=cfg.critic_learning_rate_decay_step,
            ),
            weight_decay=cfg.critic_weight_decay,
        ),
        dynamic_scale=dynamic_scale.DynamicScale() if cfg.mixed_precision else None,
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

    temperature = Trainer.create(
        network_def=HyperSACDevTemperature(cfg.temp_initial_value),
        network_inputs={
            "rngs": temp_key,
        },
        tx=optax.adamw(
            learning_rate=cfg.temp_learning_rate,
            weight_decay=cfg.temp_weight_decay,
        ),
    )

    # l2-normalize the network after initialization
    actor = l2normalize_network(actor)
    critic = l2normalize_network(critic)
    target_critic = l2normalize_network(target_critic)

    return rng, actor, critic, target_critic, temperature


@jax.jit
def _sample_hyper_sac_dev_actions(
    rng: PRNGKey,
    actor: Trainer,
    observations: jnp.ndarray,
    temperature: float = 1.0,
) -> Tuple[PRNGKey, jnp.ndarray]:
    rng, key = jax.random.split(rng)
    dist, _ = actor(observations=observations, temperature=temperature)
    actions = dist.sample(seed=key)
    actions = jnp.clip(actions, -1.0, 1.0)

    return rng, actions


@functools.partial(
    jax.jit,
    static_argnames=(
        "gamma",
        "n_step",
        "critic_use_cdq",
        "critic_use_categorical",
        "categorical_min_v",
        "categorical_max_v",
        "target_copy_type",
        "target_copy",
        "target_tau",
        "target_normalize_weight",
        "temp_target_entropy",
    ),
)
def _update_hyper_sac_dev_networks(
    rng: PRNGKey,
    actor: Trainer,
    critic: Trainer,
    target_critic: Trainer,
    temperature: Trainer,
    batch: Batch,
    gamma: float,
    n_step: int,
    critic_use_cdq: bool,
    critic_use_categorical: bool,
    categorical_min_v: float,
    categorical_max_v: float,
    target_copy_type: str,
    target_copy: bool,
    target_tau: float,
    target_normalize_weight: bool,
    temp_target_entropy: float,
) -> Tuple[PRNGKey, Trainer, Trainer, Trainer, Trainer, Dict[str, float]]:
    rng, actor_key, critic_key = jax.random.split(rng, 3)

    if critic_use_categorical:
        new_actor, actor_info = update_actor_with_categorical_critic(
            key=actor_key,
            actor=actor,
            critic=critic,
            temperature=temperature,
            batch=batch,
            min_v=categorical_min_v,
            max_v=categorical_max_v,
            critic_use_cdq=critic_use_cdq,
        )
    else:
        new_actor, actor_info = update_actor(
            key=actor_key,
            actor=actor,
            critic=critic,
            temperature=temperature,
            batch=batch,
            critic_use_cdq=critic_use_cdq,
        )

    new_temperature, temperature_info = update_temperature(
        temperature=temperature,
        entropy=actor_info["actor/entropy"],
        target_entropy=temp_target_entropy,
    )

    if critic_use_categorical:
        new_critic, critic_info = update_categorical_critic(
            key=critic_key,
            actor=new_actor,
            critic=critic,
            target_critic=target_critic,
            temperature=new_temperature,
            batch=batch,
            gamma=gamma,
            n_step=n_step,
            min_v=categorical_min_v,
            max_v=categorical_max_v,
            critic_use_cdq=critic_use_cdq,
        )
    else:
        new_critic, critic_info = update_critic(
            key=critic_key,
            actor=new_actor,
            critic=critic,
            target_critic=target_critic,
            temperature=new_temperature,
            batch=batch,
            gamma=gamma,
            n_step=n_step,
            critic_use_cdq=critic_use_cdq,
        )

    new_target_critic, target_critic_info = update_target_network(
        network=new_critic,
        target_network=target_critic,
        target_copy_type=target_copy_type,
        target_copy=target_copy,
        target_tau=target_tau,
        normalize_weight=target_normalize_weight,
    )

    info = {
        **actor_info,
        **critic_info,
        **target_critic_info,
        **temperature_info,
    }

    return (rng, new_actor, new_critic, new_target_critic, new_temperature, info)


############
# Metrics
def _get_metrics_hyper_sac_dev_networks(
    rng: PRNGKey,
    actor: Trainer,
    critic: Trainer,
    target_critic: Trainer,
    batch: Batch,
) -> Tuple[PRNGKey, Trainer, Trainer, Trainer, Trainer, Trainer, Dict[str, float]]:
    """
    get_metrics currently measures
        1) Dormant Ratio
        2) Zero activation ratio
        3) Feature norm
        4) Weight norm
        5) Srank
        6) Smooth rank
        7) Feature coadaptation (DR3 Kumar et al.)
    """

    _, actor_info = actor(observations=batch["observation"])
    actor_metrics_info = {
        **get_dormant_ratio(actor_info, prefix="actor", tau=0.1),
        **get_dormant_ratio(actor_info, prefix="actor", tau=0.0),
        **get_feature_norm(actor_info, prefix="actor"),
        **get_rank(actor_info, prefix="actor"),
        **get_weight_norm(actor.params, prefix="actor"),
        **get_scaler_stat(actor.params, prefix="actor"),
    }

    _, critic_info = critic(observations=batch["observation"], actions=batch["action"])
    critic_metrics_info = {
        **get_dormant_ratio(critic_info, prefix="critic", tau=0.1),
        **get_dormant_ratio(critic_info, prefix="critic", tau=0.0),
        **get_feature_norm(critic_info, prefix="critic"),
        **get_rank(critic_info, prefix="critic"),
        **get_weight_norm(critic.params, prefix="critic"),
        **get_scaler_stat(critic.params, prefix="critic"),
    }

    _, target_critic_info = target_critic(
        observations=batch["observation"], actions=batch["action"]
    )
    target_critic_metrics_info = {
        **get_dormant_ratio(target_critic_info, prefix="target_critic", tau=0.1),
        **get_dormant_ratio(target_critic_info, prefix="target_critic", tau=0.0),
        **get_feature_norm(target_critic_info, prefix="target_critic"),
        **get_rank(target_critic_info, prefix="target_critic"),
        **get_weight_norm(target_critic.params, prefix="target_critic"),
        **get_scaler_stat(target_critic.params, prefix="target_critic"),
    }

    new_rng, dr3_info = get_critic_featdot(
        rng=rng, actor=actor, critic=critic, batch=batch
    )

    metrics_info = {
        **actor_metrics_info,
        **critic_metrics_info,
        **target_critic_metrics_info,
        **dr3_info,
    }

    return new_rng, metrics_info


############
class HyperSACDevAgent(BaseAgent):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        cfg: HyperSACDevConfig,
    ):
        """
        An agent that randomly selects actions without training.
        Useful for collecting baseline results and for debugging purposes.
        """

        self._observation_dim = observation_space.shape[-1]
        self._action_dim = action_space.shape[-1]

        cfg["temp_target_entropy"] = cfg["temp_target_entropy_coef"] * self._action_dim

        super(HyperSACDevAgent, self).__init__(
            observation_space,
            action_space,
            cfg,
        )

        # map dictionary to dataclass
        self._cfg = HyperSACDevConfig(**cfg)

        self._init_network()

    def _init_network(self):
        (
            self._rng,
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
        ) = _init_hyper_sac_dev_networks(
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

        self._rng, actions = _sample_hyper_sac_dev_actions(
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
        ) = _update_hyper_sac_dev_networks(
            rng=self._rng,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            batch=batch,
            gamma=self._cfg.gamma,
            n_step=self._cfg.n_step,
            critic_use_cdq=self._cfg.critic_use_cdq,
            critic_use_categorical=self._cfg.critic_use_categorical,
            categorical_min_v=self._cfg.categorical_min_v,
            categorical_max_v=self._cfg.categorical_max_v,
            target_copy_type=self._cfg.target_copy_type,
            target_copy=(update_step % self._cfg.target_copy_every == 0),
            target_tau=self._cfg.target_tau,
            target_normalize_weight=self._cfg.target_normalize_weight,
            temp_target_entropy=self._cfg.temp_target_entropy,
        )

        for key, value in update_info.items():
            update_info[key] = float(value)

        return update_info

    def get_metrics(self, batch: Dict[str, np.ndarray]) -> Dict:
        for key, value in batch.items():
            batch[key] = jnp.asarray(value)
        (
            self._rng,
            metrics_info,
        ) = _get_metrics_hyper_sac_dev_networks(
            rng=self._rng,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            batch=batch,
        )

        for key, value in metrics_info.items():
            metrics_info[key] = float(value)

        return metrics_info
