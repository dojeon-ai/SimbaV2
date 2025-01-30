import functools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import dynamic_scale

from scale_rl.agents.base_agent import BaseAgent
from scale_rl.agents.simba.simba_network import (
    SimbaActor,
    SimbaCategoricalCritic,
    SimbaCategoricalDoubleCritic,
    SimbaClippedDoubleCritic,
    SimbaCritic,
    SimbaTemperature,
)
from scale_rl.agents.simba.simba_update import (
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
    get_effective_lr,
    get_feature_norm,
    get_gnorm,
    get_num_parameters_dict,
    get_pnorm,
    get_rank,
)
from scale_rl.networks.trainer import PRNGKey, Trainer

"""
The @dataclass decorator must have `frozen=True` to ensure the instance is immutable,
allowing it to be treated as a static variable in JAX.
"""


@dataclass(frozen=True)
class SimbaConfig:
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

    actor_block_type: str
    actor_num_blocks: int
    actor_hidden_dim: int
    actor_learning_rate_init: float
    actor_learning_rate_end: float
    actor_learning_rate_decay_rate: float
    actor_learning_rate_decay_step: int
    actor_weight_decay: float

    critic_block_type: str
    critic_num_blocks: int
    critic_hidden_dim: int
    critic_learning_rate_init: float
    critic_learning_rate_end: float
    critic_learning_rate_decay_rate: float
    critic_learning_rate_decay_step: int
    critic_weight_decay: float
    critic_use_cdq: bool

    critic_use_categorical: bool
    critic_num_bins: int
    categorical_min_v: float
    categorical_max_v: float

    temp_target_entropy: float
    temp_target_entropy_coef: float
    temp_initial_value: float
    temp_learning_rate: float
    temp_weight_decay: float

    target_tau: float
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
def _init_simba_networks(
    observation_dim: int,
    action_dim: int,
    cfg: SimbaConfig,
) -> Tuple[PRNGKey, Trainer, Trainer, Trainer, Trainer]:
    fake_observations = jnp.zeros((1, observation_dim))
    fake_actions = jnp.zeros((1, action_dim))

    rng = jax.random.PRNGKey(cfg.seed)
    rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
    compute_dtype = jnp.float16 if cfg.mixed_precision else jnp.float32

    # When initializing the network in the flax.nn.Module class, rng_key should be passed as rngs.
    actor = Trainer.create(
        network_def=SimbaActor(
            block_type=cfg.actor_block_type,
            num_blocks=cfg.actor_num_blocks,
            hidden_dim=cfg.actor_hidden_dim,
            action_dim=action_dim,
            dtype=compute_dtype,
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
            critic_network_def = SimbaCategoricalDoubleCritic(
                block_type=cfg.critic_block_type,
                num_blocks=cfg.critic_num_blocks,
                hidden_dim=cfg.critic_hidden_dim,
                num_bins=cfg.critic_num_bins,
                dtype=compute_dtype,
            )
        else:
            critic_network_def = SimbaClippedDoubleCritic(
                block_type=cfg.critic_block_type,
                num_blocks=cfg.critic_num_blocks,
                hidden_dim=cfg.critic_hidden_dim,
                dtype=compute_dtype,
            )
    else:
        if cfg.critic_use_categorical:
            critic_network_def = SimbaCategoricalCritic(
                block_type=cfg.critic_block_type,
                num_blocks=cfg.critic_num_blocks,
                hidden_dim=cfg.critic_hidden_dim,
                num_bins=cfg.critic_num_bins,
                dtype=compute_dtype,
            )
        else:
            critic_network_def = SimbaCritic(
                block_type=cfg.critic_block_type,
                num_blocks=cfg.critic_num_blocks,
                hidden_dim=cfg.critic_hidden_dim,
                dtype=compute_dtype,
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
        network_def=SimbaTemperature(cfg.temp_initial_value),
        network_inputs={
            "rngs": temp_key,
        },
        tx=optax.adamw(
            learning_rate=cfg.temp_learning_rate,
            weight_decay=cfg.temp_weight_decay,
        ),
    )

    return rng, actor, critic, target_critic, temperature


@jax.jit
def _sample_simba_actions(
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
        "critic_use_categorical",
        "categorical_min_v",
        "categorical_max_v",
        "categorical_num_bins",
        "target_tau",
        "temp_target_entropy",
    ),
)
def _update_simba_networks(
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
    categorical_num_bins: int,
    target_tau: float,
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
            num_bins=categorical_num_bins,
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
            num_bins=categorical_num_bins,
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
        target_tau=target_tau,
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


def _get_metrics_simba_networks(
    rng: PRNGKey,
    actor: Trainer,
    critic: Trainer,
    critic_use_cdq: bool,
    target_critic: Trainer,
    actor_pcount_dict: Dict[str, int],
    critic_pcount_dict: Dict[str, int],
    batch: Batch,
    update_info: Dict[str, Any],
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
    # Actor
    _, actor_info = actor(observations=batch["observation"])
    actor_pnorm_dict = get_pnorm(actor.params, actor_pcount_dict, prefix="actor")
    actor_gnorm_dict = get_gnorm(
        update_info.pop("actor/_grads"), actor_pcount_dict, prefix="actor"
    )
    actor_effective_lr_dict = get_effective_lr(
        actor_gnorm_dict, actor_pnorm_dict, actor_pcount_dict, prefix="actor"
    )
    actor_metrics_info = {
        **get_dormant_ratio(actor_info, prefix="actor", tau=0.1),
        **get_dormant_ratio(actor_info, prefix="actor", tau=0.0),
        **get_feature_norm(actor_info, prefix="actor"),
        **get_rank(actor_info, prefix="actor"),
        **actor_pnorm_dict,
        **actor_gnorm_dict,
        **actor_effective_lr_dict,
    }

    # Critic
    _, critic_info = critic(observations=batch["observation"], actions=batch["action"])
    critic_params = critic.params
    critic_grads = update_info.pop("critic/_grads")
    # Remove Vmap module
    if critic_use_cdq:
        # Elements (e.g. pnorm, gnorm) of vmapped functions (multi-head Q-networks) are summed to a single value
        (_, critic_params), = critic_params.items()
        (_, critic_grads), = critic_grads.items()
    critic_pnorm_dict = get_pnorm(critic_params, critic_pcount_dict, prefix="critic")
    critic_gnorm_dict = get_gnorm(critic_grads, critic_pcount_dict, prefix="critic")
    critic_effective_lr_dict = get_effective_lr(
        critic_gnorm_dict, critic_pnorm_dict, critic_pcount_dict, prefix="critic"
    )
    critic_metrics_info = {
        **get_dormant_ratio(critic_info, prefix="critic", tau=0.1),
        **get_dormant_ratio(critic_info, prefix="critic", tau=0.0),
        **get_feature_norm(critic_info, prefix="critic"),
        **get_rank(critic_info, prefix="critic"),
        **critic_pnorm_dict,
        **critic_gnorm_dict,
        **critic_effective_lr_dict,
    }
    new_rng, dr3_info = get_critic_featdot(
        rng=rng, actor=actor, critic=critic, batch=batch
    )

    # Target Critic
    _, target_critic_info = target_critic(
        observations=batch["observation"], actions=batch["action"]
    )
    target_critic_metrics_info = {
        **get_dormant_ratio(target_critic_info, prefix="target_critic", tau=0.1),
        **get_dormant_ratio(target_critic_info, prefix="target_critic", tau=0.0),
        **get_feature_norm(target_critic_info, prefix="target_critic"),
        **get_rank(target_critic_info, prefix="target_critic"),
    }

    metrics_info = {
        **actor_metrics_info,
        **critic_metrics_info,
        **target_critic_metrics_info,
        **dr3_info,
    }

    return new_rng, metrics_info


############
class SimbaAgent(BaseAgent):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        cfg: SimbaConfig,
    ):
        """
        An agent that randomly selects actions without training.
        Useful for collecting baseline results and for debugging purposes.
        """

        self._observation_dim = observation_space.shape[-1]
        self._action_dim = action_space.shape[-1]

        cfg["temp_target_entropy"] = cfg["temp_target_entropy_coef"] * self._action_dim

        super(SimbaAgent, self).__init__(
            observation_space,
            action_space,
            cfg,
        )

        # map dictionary to dataclass
        self._cfg = SimbaConfig(**cfg)

        self._init_network()

        # to measure effective learning rate
        self._actor_pcount_dict = get_num_parameters_dict(
            self._actor.params, prefix="actor"
        )

        critic_param = self._critic.params
        # taking off Vmap module
        if self._cfg.critic_use_cdq:
            (_, critic_param),  = self._critic.params.items()
        self._critic_pcount_dict = get_num_parameters_dict(
            critic_param, prefix="critic"
        )

    def _init_network(self):
        (
            self._rng,
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
        ) = _init_simba_networks(self._observation_dim, self._action_dim, self._cfg)

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

        self._rng, actions = _sample_simba_actions(
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
        ) = _update_simba_networks(
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
            categorical_num_bins=self._cfg.critic_num_bins,
            target_tau=self._cfg.target_tau,
            temp_target_entropy=self._cfg.temp_target_entropy,
        )

        for key, value in update_info.items():
            if isinstance(value, dict):
                continue
            else:
                update_info[key] = float(value)

        return update_info

    def get_metrics(
        self,
        batch: Dict[str, np.ndarray],
        update_info: Dict[str, Any],
    ) -> Dict:
        for key, value in batch.items():
            batch[key] = jnp.asarray(value)
        (
            self._rng,
            metrics_info,
        ) = _get_metrics_simba_networks(
            rng=self._rng,
            actor=self._actor,
            critic=self._critic,
            critic_use_cdq=self._cfg.critic_use_cdq,
            target_critic=self._target_critic,
            actor_pcount_dict=self._actor_pcount_dict,
            critic_pcount_dict=self._critic_pcount_dict,
            batch=batch,
            update_info=update_info,
        )

        for key, value in metrics_info.items():
            if isinstance(value, dict):
                continue
            metrics_info[key] = float(value)

        return metrics_info

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
