import re
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

from scale_rl.networks.trainer import PRNGKey, Trainer
from scale_rl.networks.utils import tree_filter, tree_map_until_match

# Additional typings
Params = flax.core.FrozenDict[str, Any]
Array = Union[np.ndarray, jnp.ndarray]
Data = Union[Array, Dict[str, "Data"]]
Batch = Dict[str, Data]


# rephrase each key
def flatten_dict(d, parent_key="", sep="_"):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) or isinstance(v, FrozenDict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def add_prefix_to_dict(d: dict, prefix: str = None, sep="/") -> dict:
    new_dict = {}
    for key, value in d.items():
        new_dict[prefix + sep + key] = value
    return new_dict


def sum_all_values_in_pytree(pytree) -> float:
    # Flatten the pytree to get all leaves (individual values)
    leaves = jax.tree_util.tree_leaves(pytree)

    # Sum all leaves
    total_sum = sum(jnp.sum(leaf) for leaf in leaves)

    return total_sum


def get_weight_norm(
    param_dict: Params,
    prefix: str,
) -> Dict[str, jnp.ndarray]:
    """
    param_dict is a frozen dictionary which contains the values of each individual parameter

    Return:
        param value norm dictionary
        (Caution : norm values for vmapped functions (multi-head Q-networks) are summed to a single value)
    """
    param_norm_dict = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), param_dict)

    updated_params = add_all_key(param_norm_dict)
    squared_param_norm_dict = jax.tree_util.tree_map(
        lambda x: jnp.square(x), param_norm_dict
    )
    updated_params["total"] = jnp.sqrt(
        sum_all_values_in_pytree(squared_param_norm_dict)
    )

    return add_prefix_to_dict(
        flatten_dict(updated_params), prefix + "/weightnorm", sep="_"
    )


def get_grad_norm(
    grad_dict: Params,
    prefix: str,
) -> Dict[str, jnp.ndarray]:
    """
    grad_dict is a frozen dictionary which contains the gradients of each individual parameter

    Return:
        param gradient norm dictionary
        (Caution : norm values for vmapped functions (multi-head Q-networks) are summed to a single value)
    """
    grad_norm_dict = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), grad_dict)
    updated_params = add_all_key(grad_norm_dict)
    squared_param_norm_dict = jax.tree_util.tree_map(
        lambda x: jnp.square(x), grad_norm_dict
    )

    updated_params["total"] = jnp.sqrt(
        sum_all_values_in_pytree(squared_param_norm_dict)
    )

    return add_prefix_to_dict(
        flatten_dict(updated_params), prefix + "/gradnorm", sep="_"
    )


def get_effective_lr(
    grad_dict: Params,
    param_dict: Params,
    prefix: str,
) -> Dict[str, jnp.ndarray]:
    """
    grad_dict is a frozen dictionary which contains the gradients of each individual parameter

    Return:
        param gradient norm dictionary
        (Caution : norm values for vmapped functions (multi-head Q-networks) are summed to a single value)
    """
    effective_lr_dict = {}
    for _k, _g in grad_dict.items():
        # remove prefix
        _layer = _k.replace(prefix + "/gradnorm_", "")
        _p = param_dict[prefix + "/weightnorm_" + _layer]
        effective_lr_dict[prefix + "/effective_lr_" + _layer] = _g / _p
    return effective_lr_dict


def get_scaler_stat(
    param_dict: Params,
    prefix: str,
) -> Dict[str, jnp.ndarray]:
    """
    param_dict is a frozen dictionary which contains the gradients/values of each individual parameter

    Return:
        param gradient/value norm dictionary
        (Caution : norm values for vmapped functions (multi-head Q-networks) are summed to a single value)
    """
    regex = "scaler"
    mean = tree_filter(f=lambda x: jnp.mean(x), tree=param_dict, target_re=regex)
    var = tree_filter(f=lambda x: jnp.var(x), tree=param_dict, target_re=regex)
    mean_dict = add_prefix_to_dict(flatten_dict(mean), prefix + "/mean", sep="_")
    var_dict = add_prefix_to_dict(flatten_dict(var), prefix + "/var", sep="_")

    info = {}
    info.update(mean_dict)
    info.update(var_dict)

    return info


def add_all_key(d):
    new_dict = {}
    for key, value in d.items():
        if isinstance(value, dict) or isinstance(value, FrozenDict):
            new_dict[key] = add_all_key(value)
            if "kernel" in new_dict[key] and "bias" in new_dict[key]:
                kernel_norm = jnp.square(new_dict[key]["kernel"])
                bias_norm = jnp.square(new_dict[key]["bias"])
                # Integrated Norm
                new_dict[key + "_kernel+bias"] = jnp.sqrt(kernel_norm + bias_norm)
                # Separated Norm
                new_dict[key + "_kernel"] = jnp.sqrt(kernel_norm)
                new_dict[key + "_bias"] = jnp.sqrt(bias_norm)
        else:
            new_dict[key] = jnp.linalg.norm(value)
    return new_dict


def get_dormant_ratio(
    activations: Dict[str, List[jnp.ndarray]], prefix: str, tau: float = 0.1
) -> Dict[str, jnp.ndarray]:
    """
    Compute the dormant mask for a given set of activations.

    Args:
        activations: A dictionary of activations.
        prefix: A string prefix for naming.
        tau: The threshold for determining dormancy.

    Returns:
        A dictionary of dormancy ratios for each layer and the total.

    Source : https://github.com/timoklein/redo/blob/dcaeff1c6afd0f1615a21da5beda870487b2ed15/src/redo.py#L215
    """
    key = "dormant" if tau > 0.0 else "zeroactiv"
    ratios = {}
    total_activs = []

    for sub_layer_name, activs in list(activations.items()):
        layer_name = f"{prefix}_{sub_layer_name}"

        # For double critics, lets just stack them into one batch
        if len(activs.shape) > 2:
            activs = activs.reshape(-1, activs.shape[-1])

        # Taking the mean here conforms to the expectation under D in the main paper's formula
        score = jnp.abs(activs).mean(axis=0)
        # Divide by activation mean to make the threshold independent of the layer size
        # see https://github.com/google/dopamine/blob/ce36aab6528b26a699f5f1cefd330fdaf23a5d72/dopamine/labs/redo/weight_recyclers.py#L314
        # https://github.com/google/dopamine/issues/209
        normalized_score = score / (score.mean() + 1e-9)

        if tau > 0.0:
            layer_mask = jnp.where(normalized_score <= tau, 1, 0)
        else:
            layer_mask = jnp.where(
                jnp.isclose(normalized_score, jnp.zeros_like(normalized_score)), 1, 0
            )

        ratios[f"{prefix}/{key}_{layer_name}"] = (
            jnp.sum(layer_mask) / layer_mask.size
        ) * 100
        total_activs.append(layer_mask)

    # aggregated mask of entire network
    total_mask = jnp.concatenate(total_activs)

    ratios[f"{prefix}/{key}_total"] = (jnp.sum(total_mask) / total_mask.size) * 100

    return ratios


# source: https://github.com/CLAIRE-Labo/no-representation-no-trust/blob/52a785da4aee93b569d87a289b1f5865271aedfe/src/po_dynamics/modules/metrics.py#L9
def get_rank(
    activations: Dict[str, List[jnp.ndarray]], prefix: str, tau: float = 0.01
) -> Dict[str, jnp.ndarray]:
    """
    Computes different approximations of the rank for a given set of activations.

    Args:
        activations: A dictionary of activations.
        prefix: A string prefix for naming.
        tau: cutoff parameter. not used in (1), 1 - 99% in (2), delta in (3), epsilon in (4).

    Returns:
        (1) Effective rank.
        A continuous approximation of the rank of a matrix.
        Definition 2.1. in Roy & Vetterli, (2007) https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7098875
        Also used in Huh et al. (2023) https://arxiv.org/pdf/2103.10427.pdf

        (2) Approximate rank.
        Threshold at the dimensions explaining 99% of the variance in a PCA analysis.
        Section 2 in Yang et al. (2020) https://arxiv.org/pdf/1909.12255.pdf

        (3) srank.
        Another version of (2).
        Section 3 in Kumar et al. https://arxiv.org/pdf/2010.14498.pdf

        (4) Feature rank.
        A threshold rank: normalize by dim size and discard dimensions with singular values below 0.01.
        Equations (4) and (5). Lyle et al. (2022) https://arxiv.org/pdf/2204.09560.pdf

        (5) Jnp rank.
        Rank defined in jnp. A reasonable value for cutoff parameter is chosen based the floating point precision of the input.
    """
    threshold = 1 - tau
    ranks = {}

    for sub_layer_name, feature in list(activations.items()):
        layer_name = f"{prefix}_{sub_layer_name}"

        # For double critics, lets just stack them into one batch
        if len(feature.shape) > 2:
            feature = feature.reshape(-1, feature.shape[-1])

        # Compute the L2 norm for all examples in the batch at once
        svals = jnp.linalg.svdvals(feature)

        # (1) Effective rank.
        sval_sum = jnp.sum(svals)
        sval_dist = svals / sval_sum
        # Replace 0 with 1. This is a safe trick to avoid log(0) = -inf
        # as Roy & Vetterli assume 0*log(0) = 0 = 1*log(1).
        sval_dist_fixed = jnp.where(sval_dist == 0, jnp.ones_like(sval_dist), sval_dist)
        effective_ranks = jnp.exp(-jnp.sum(sval_dist_fixed * jnp.log(sval_dist_fixed)))

        # (2) Approximate rank. PCA variance. Yang et al. (2020)
        sval_squares = svals**2
        sval_squares_sum = jnp.sum(sval_squares)
        cumsum_squares = jnp.cumsum(sval_squares)
        threshold_crossed = cumsum_squares >= (threshold * sval_squares_sum)
        approximate_ranks = (~threshold_crossed).sum() + 1

        # (3) srank. Weird. Kumar et al. (2020)
        cumsum = jnp.cumsum(svals)
        threshold_crossed = cumsum >= threshold * sval_sum
        sranks = (~threshold_crossed).sum() + 1

        # (4) Feature rank. Most basic. Lyle et al. (2022)
        n_obs = feature.shape[0]
        svals_of_normalized = svals / jnp.sqrt(n_obs)
        over_cutoff = svals_of_normalized > tau
        feature_ranks = over_cutoff.sum()

        # (5) jnp rank.
        # Note that this determines the matrix rank same with (4), but some reasonable tau is chosen automatically based on the floating point precision of the input.
        jnp_ranks = jnp.linalg.matrix_rank(feature)

        ranks.update(
            {
                f"{prefix}/effective_rank_vetterli_{layer_name}": effective_ranks,
                f"{prefix}/approximate_rank_pca_{layer_name}": approximate_ranks,
                f"{prefix}/srank_kumar_{layer_name}": sranks,
                f"{prefix}/feature_rank_lyle_{layer_name}": feature_ranks,
                f"{prefix}/matrix_rank_{layer_name}": jnp_ranks,
            }
        )

    return ranks


def get_feature_norm(
    activations: Dict[str, List[jnp.ndarray]], prefix: str
) -> Dict[str, jnp.ndarray]:
    """
    Computes the feature norm for a given set of activations.
    """
    norms = {}
    total_norm = 0.0
    for sub_layer_name, activs in list(activations.items()):
        layer_name = f"{prefix}_{sub_layer_name}"

        # For double critics, lets just stack them into one batch
        if len(activs.shape) > 2:
            activs = activs.reshape(-1, activs.shape[-1])

        # Compute the L2 norm for all examples in the batch at once
        batch_norms = jnp.linalg.norm(activs, ord=2, axis=-1)

        # Compute the expected (mean) L2 norm across the batch
        expected_norm = jnp.mean(batch_norms)

        norms[f"{prefix}/featnorm_{layer_name}"] = expected_norm
        total_norm += expected_norm

    norms[f"{prefix}/featnorm_total"] = total_norm

    return norms


# source : https://arxiv.org/pdf/2112.04716
def get_critic_featdot(
    rng: PRNGKey, actor: Trainer, critic: Trainer, batch: Batch, sample: bool = True
) -> Tuple[PRNGKey, Dict[str, jnp.ndarray]]:
    new_rng, cur_sample_key, next_sample_key = jax.random.split(rng, 3)

    if sample:
        dist, _ = actor(observations=batch["observation"])
        cur_actions = dist.sample(seed=cur_sample_key)

        next_dist, _ = actor(observations=batch["next_observation"])
        next_actions = next_dist.sample(seed=next_sample_key)
    else:
        cur_actions, _ = actor(observations=batch["observation"])
        next_actions, _ = actor(observations=batch["next_observation"])

    _, cur_critic_info = critic(observations=batch["observation"], actions=cur_actions)

    final_cur_critic_feat = cur_critic_info[get_last_layer(cur_critic_info)]
    if len(final_cur_critic_feat.shape) > 2:
        final_cur_critic_feat = final_cur_critic_feat.reshape(
            -1, final_cur_critic_feat.shape[-1]
        )

    _, next_critic_info = critic(
        observations=batch["next_observation"], actions=next_actions
    )

    final_next_critic_feat = next_critic_info[get_last_layer(next_critic_info)]
    if len(final_next_critic_feat.shape) > 2:
        final_next_critic_feat = final_next_critic_feat.reshape(
            -1, final_next_critic_feat.shape[-1]
        )

    # Compute mean dot product of the batch
    # Don't do cosine similarity, it has to be dot product according to the paper
    result = jnp.mean(
        jnp.sum(final_cur_critic_feat * final_next_critic_feat, axis=1), axis=0
    )

    return new_rng, {"critic/DR3_featdot": result}


def get_last_layer(layer_dict):
    def extract_number(key):
        match = re.search(r"\d+$", key)
        return int(match.group()) if match else -1

    # Sort keys based on the numeric suffix
    sorted_keys = sorted(layer_dict.keys(), key=extract_number, reverse=True)

    # Return the first key (which will be the one with the highest number)
    return sorted_keys[0] if sorted_keys else None


def print_num_parameters(
    pytree_dict_list: Sequence[FrozenDict], network_type: str
) -> int:
    """
    Return number of trainable parameters
    """
    total_params = 0

    for pytree_dict in pytree_dict_list:
        leaf_nodes = jax.tree_util.tree_leaves(pytree_dict)
        for leaf in leaf_nodes:
            total_params += np.prod(leaf.shape)

    # Format the total_params to a human-readable string
    if total_params >= 1e6:
        return print(f"{network_type} total params: {total_params / 1e6:.2f}M")
    elif total_params >= 1e3:
        return print(f"{network_type} total params: {total_params / 1e3:.2f}K")
    else:
        return print(f"{network_type} total params: {total_params}")
