from typing import Any, Dict, Tuple

import flax
import jax
import jax.numpy as jnp

from scale_rl.buffers import Batch
from scale_rl.networks.critics import (
    compute_categorical_loss,
    compute_categorical_value,
)
from scale_rl.networks.projection_utils import l2normalize
from scale_rl.networks.trainer import PRNGKey, Trainer
from scale_rl.networks.utils import tree_map_until_match, tree_norm


def l2normalize_layer(tree):
    """
    apply l2-normalization to the all leaf nodes
    """
    if len(tree["kernel"].shape) == 2:
        axis = 0
    elif len(tree["kernel"].shape) == 3:
        axis = 1
    else:
        raise ValueError
    return jax.tree.map(f=lambda x: l2normalize(x, axis=axis), tree=tree)


def l2normalize_network(
    network: Trainer,
    regex: str = "hyper_dense",
) -> Trainer:
    params = network.params
    new_params = tree_map_until_match(
        f=lambda x: l2normalize_layer(x), tree=params, target_re=regex, keep_values=True
    )
    network = network.replace(params=new_params)
    return network


def update_actor(
    key: PRNGKey,
    actor: Trainer,
    critic: Trainer,  # SACDoubleCritic
    temperature: Trainer,
    batch: Batch,
    bin_values: jnp.ndarray,
    critic_use_cdq: bool,
) -> Tuple[Trainer, Dict[str, float]]:
    def actor_loss_fn(
        actor_params: flax.core.FrozenDict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        dist, _ = actor.apply(
            variables={"params": actor_params},
            observations=batch["observation"],
        )
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)

        if critic_use_cdq:
            (q_log_probs_1, q_log_probs_2), _ = critic(
                observations=batch["observation"], actions=actions
            )
            q1 = compute_categorical_value(q_log_probs_1, bin_values)
            q2 = compute_categorical_value(q_log_probs_2, bin_values)
            q = jnp.minimum(q1, q2).reshape(-1)  # (n, 1) -> (n, )
        else:
            q_log_probs, _ = critic(
                observations=batch["observation"],
                actions=actions,
            )
            q = compute_categorical_value(q_log_probs, bin_values)

        actor_loss = (log_probs * temperature() - q).mean()
        actor_info = {
            "actor/loss": actor_loss,
            "actor/entropy": -log_probs.mean(),
            "actor/mean_action": jnp.mean(jnp.abs(actions)),
            # "actor/total_pnorm": tree_norm(actor_params),
        }
        return actor_loss, actor_info

    actor, info = actor.apply_gradient(actor_loss_fn, get_info=False)
    actor = l2normalize_network(actor)

    return actor, info


def update_critic(
    key: PRNGKey,
    actor: Trainer,
    critic: Trainer,
    target_critic: Trainer,
    temperature: Trainer,
    batch: Batch,
    critic_use_cdq: bool,
    min_v: float,
    max_v: float,
    num_bins: int,
    bin_values: jnp.ndarray,
    gamma: float,
    n_step: int,
) -> Tuple[Trainer, Dict[str, float]]:
    # compute the target q-value
    next_dist, _ = actor(observations=batch["next_observation"])
    next_actions = next_dist.sample(seed=key)
    next_log_probs = next_dist.log_prob(next_actions)

    if critic_use_cdq:
        (next_q_log_probs_1, next_q_log_probs_2), _ = target_critic(
            observations=batch["next_observation"], actions=next_actions
        )
        next_q1 = compute_categorical_value(next_q_log_probs_1, bin_values)
        next_q2 = compute_categorical_value(next_q_log_probs_2, bin_values)
        min_indices = jnp.concat([next_q1, next_q2], axis=1).argmin(axis=1)
        stacked_log_probs = jnp.stack([next_q_log_probs_1, next_q_log_probs_2], axis=1)
        next_q_log_probs = jax.vmap(lambda a, b: a[b])(stacked_log_probs, min_indices)
    else:
        next_q_log_probs, _ = target_critic(
            observations=batch["next_observation"],
            actions=next_actions,
        )

    def critic_loss_fn(
        critic_params: flax.core.FrozenDict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        # compute predicted q-value
        if critic_use_cdq:
            (pred_log_probs_1, pred_log_probs_2), _ = critic.apply(
                variables={"params": critic_params},
                observations=batch["observation"],
                actions=batch["action"],
            )
            pred_log_probs = pred_log_probs_1
            loss_1 = compute_categorical_loss(
                log_probs=pred_log_probs_1,
                gamma=gamma**n_step,
                reward=batch["reward"].reshape((-1, 1)),
                done=batch["terminated"].reshape((-1, 1)),
                target_log_probs=next_q_log_probs,
                entropy=(temperature() * next_log_probs).reshape((-1, 1)),
                bin_values=bin_values,
                num_bins=num_bins,
                min_v=min_v,
                max_v=max_v,
            )
            loss_2 = compute_categorical_loss(
                log_probs=pred_log_probs_2,
                gamma=gamma**n_step,
                reward=batch["reward"].reshape((-1, 1)),
                done=batch["terminated"].reshape((-1, 1)),
                target_log_probs=next_q_log_probs,
                entropy=(temperature() * next_log_probs).reshape((-1, 1)),
                bin_values=bin_values,
                num_bins=num_bins,
                min_v=min_v,
                max_v=max_v,
            )
            # compute mse loss
            critic_loss = (loss_1 + loss_2).mean()

        else:
            pred_log_probs, _ = critic.apply(
                variables={"params": critic_params},
                observations=batch["observation"],
                actions=batch["action"],
            )
            loss = compute_categorical_loss(
                log_probs=pred_log_probs,
                gamma=gamma**n_step,
                reward=batch["reward"].reshape((-1, 1)),
                done=batch["terminated"].reshape((-1, 1)),
                target_log_probs=next_q_log_probs,
                entropy=(temperature() * next_log_probs).reshape((-1, 1)),
                bin_values=bin_values,
                num_bins=num_bins,
                min_v=min_v,
                max_v=max_v,
            )

            # compute mse loss
            critic_loss = loss.mean()

        critic_info = {
            "critic/loss": critic_loss,
            "critic/batch_rew_min": batch["reward"].min(),
            "critic/batch_rew_mean": batch["reward"].mean(),
            "critic/batch_rew_max": batch["reward"].max(),
            # "critic/total_pnorm": tree_norm(critic_params),
        }

        return critic_loss, critic_info

    critic, info = critic.apply_gradient(critic_loss_fn, get_info=False)
    critic = l2normalize_network(critic)

    return critic, info


def update_target_network(
    network: Trainer,  # SACDoubleCritic
    target_network: Trainer,
    target_tau: bool,
) -> Tuple[Trainer, Dict[str, float]]:
    new_target_params = jax.tree_map(
        lambda p, tp: p * target_tau + tp * (1 - target_tau),
        network.params,
        target_network.params,
    )
    target_network = target_network.replace(params=new_target_params)

    info = {}

    return target_network, info


def update_temperature(
    temperature: Trainer, entropy: float, target_entropy: float
) -> Tuple[Trainer, Dict[str, float]]:
    def temperature_loss_fn(
        temperature_params: flax.core.FrozenDict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        temperature_value = temperature.apply({"params": temperature_params})
        temperature_loss = temperature_value * (entropy - target_entropy).mean()
        temperature_info = {
            "temperature/value": temperature_value,
            "temperature/loss": temperature_loss,
        }

        return temperature_loss, temperature_info

    temperature, info = temperature.apply_gradient(temperature_loss_fn, get_info=False)

    return temperature, info
