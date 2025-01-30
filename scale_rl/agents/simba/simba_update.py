from typing import Any, Dict, Tuple

import flax
import jax
import jax.numpy as jnp

from scale_rl.buffers import Batch
from scale_rl.networks.critics import (
    compute_categorical_loss,
    compute_categorical_value,
)
from scale_rl.networks.trainer import PRNGKey, Trainer
from scale_rl.networks.utils import tree_norm


def update_actor(
    key: PRNGKey,
    actor: Trainer,
    critic: Trainer,  # SACDoubleCritic
    temperature: Trainer,
    batch: Batch,
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
            (q1, q2), _ = critic(observations=batch["observation"], actions=actions)
            q = jnp.minimum(q1, q2).reshape(-1)  # (n, 1) -> (n, )
        else:
            q, _ = critic(observations=batch["observation"], actions=actions)
            q = q.reshape(-1)  # (n, 1) -> (n, )

        actor_loss = (log_probs * temperature() - q).mean()
        actor_info = {
            "actor/loss": actor_loss,
            "actor/entropy": -log_probs.mean(),  # not exactly entropy, just calculating randomness
            "actor/mean_action": jnp.mean(jnp.abs(actions)),
            "actor/total_pnorm": tree_norm(actor_params),
        }

        return actor_loss, actor_info

    actor, info = actor.apply_gradient(actor_loss_fn)
    info["actor/_grads"] = info.pop("_grads")

    return actor, info


def update_actor_with_categorical_critic(
    key: PRNGKey,
    actor: Trainer,
    critic: Trainer,  # SACDoubleCritic
    temperature: Trainer,
    batch: Batch,
    min_v: float,
    max_v: float,
    num_bins: int,
    critic_use_cdq: bool,
) -> Tuple[Trainer, Dict[str, float]]:
    bin_values = jnp.linspace(min_v, max_v, num_bins)

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
            "actor/entropy": -log_probs.mean(),  # not exactly entropy, just calculating randomness
            "actor/mean_action": jnp.mean(jnp.abs(actions)),
            "actor/total_pnorm": tree_norm(actor_params),
        }
        return actor_loss, actor_info

    actor, info = actor.apply_gradient(actor_loss_fn)
    info["actor/_grads"] = info.pop("_grads")

    return actor, info


def update_critic(
    key: PRNGKey,
    actor: Trainer,
    critic: Trainer,
    target_critic: Trainer,
    temperature: Trainer,
    batch: Batch,
    gamma: float,
    n_step: int,
    critic_use_cdq: bool,
) -> Tuple[Trainer, Dict[str, float]]:
    # compute the target q-value
    next_dist, _ = actor(observations=batch["next_observation"])
    next_actions = next_dist.sample(seed=key)
    next_log_probs = next_dist.log_prob(next_actions)
    if critic_use_cdq:
        (next_q1, next_q2), _ = target_critic(
            observations=batch["next_observation"], actions=next_actions
        )
        next_q = jnp.minimum(next_q1, next_q2).reshape(-1)
    else:
        next_q, _ = target_critic(
            observations=batch["next_observation"],
            actions=next_actions,
        )
        next_q = next_q.reshape(-1)
    
    # compute the td-target, incorporating the n-step accumulated reward
    # https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
    target_q = batch["reward"] + (gamma**n_step) * (1 - batch["terminated"]) * next_q
    target_q -= (
        (gamma**n_step) * (1 - batch["terminated"]) * temperature() * next_log_probs
    )

    def critic_loss_fn(
        critic_params: flax.core.FrozenDict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        # compute predicted q-value
        if critic_use_cdq:
            (pred_q1, pred_q2), _ = critic.apply(
                variables={"params": critic_params},
                observations=batch["observation"],
                actions=batch["action"],
            )
            pred_q1 = pred_q1.reshape(-1)
            pred_q2 = pred_q2.reshape(-1)

            # compute mse loss
            critic_loss = ((pred_q1 - target_q) ** 2 + (pred_q2 - target_q) ** 2).mean()
        else:
            pred_q, _ = critic.apply(
                variables={"params": critic_params},
                observations=batch["observation"],
                actions=batch["action"],
            )
            pred_q = pred_q.reshape(-1)
            pred_q1 = pred_q2 = pred_q

            # compute mse loss
            critic_loss = ((pred_q - target_q) ** 2).mean()

        critic_info = {
            "critic/loss": critic_loss,
            "critic/pred_q1_mean": pred_q1.mean(),
            "critic/pred_q2_mean": pred_q2.mean(),
            "critic/batch_rew_mean": batch["reward"].mean(),
            "critic/total_pnorm": tree_norm(critic_params),
        }

        return critic_loss, critic_info

    critic, info = critic.apply_gradient(critic_loss_fn)
    info["critic/_grads"] = info.pop("_grads")

    return critic, info


def update_categorical_critic(
    key: PRNGKey,
    actor: Trainer,
    critic: Trainer,
    target_critic: Trainer,
    temperature: Trainer,
    batch: Batch,
    gamma: float,
    n_step: int,
    min_v: float,
    max_v: float,
    num_bins: int,
    critic_use_cdq: bool,
) -> Tuple[Trainer, Dict[str, float]]:
    # compute the target q-value
    next_dist, _ = actor(observations=batch["next_observation"])
    next_actions = next_dist.sample(seed=key)
    next_log_probs = next_dist.log_prob(next_actions)

    bin_values = jnp.linspace(min_v, max_v, num_bins)
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
            "critic/total_pnorm": tree_norm(critic_params),
        }

        return critic_loss, critic_info

    critic, info = critic.apply_gradient(critic_loss_fn)
    info["critic/_grads"] = info.pop("_grads")

    return critic, info


def update_target_network(
    network: Trainer,  # SACDoubleCritic
    target_network: Trainer,
    target_tau: float,
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
            "temperature": temperature_value,
            "temperature_loss": temperature_loss,
        }

        return temperature_loss, temperature_info

    temperature, info = temperature.apply_gradient(temperature_loss_fn)
    info.pop("_grads")  # do not track temperature gradient

    return temperature, info
