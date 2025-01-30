import copy
import os
from functools import partial
from typing import Any, Optional, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import checkpoints
from flax.training import dynamic_scale as dynamic_scale_lib
from orbax.checkpoint import CheckpointManager, checkpoint_utils

from scale_rl.networks.utils import tree_map_until_match, tree_norm

PRNGKey = jnp.ndarray


@flax.struct.dataclass
class Trainer:
    network_def: nn.Module = flax.struct.field(pytree_node=False)
    params: flax.core.FrozenDict[str, Any]
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None
    update_step: int = 0
    dynamic_scale: Optional[dynamic_scale_lib.DynamicScale] = None
    """
    dataclass decorator makes custom class to be passed safely to Jax.
    https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html

    Trainer class wraps network & optimizer to easily optimize the network under the hood.

    args:
        network_def:
        params: network parameters.
        tx: optimizer (e.g., optax.Adam).
        opt_state: current state of the optimizer (e.g., beta_1 in Adam).
        update_step: number of update step so far.
    """

    @classmethod
    def create(
        cls,
        network_def: nn.Module,
        network_inputs: flax.core.FrozenDict[str, jnp.ndarray],
        tx: Optional[optax.GradientTransformation] = None,
        dynamic_scale: Optional[dynamic_scale_lib.DynamicScale] = None,
    ) -> "Trainer":
        variables = network_def.init(**network_inputs)
        params = variables.pop("params")

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        network = cls(
            network_def=network_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            dynamic_scale=dynamic_scale,
        )

        return network

    def __call__(self, *args, **kwargs):
        return self.network_def.apply({"params": self.params}, *args, **kwargs)

    def save(self, path: str, step: int = 0, keep: int = 1) -> None:
        """
        Save parameters, optimizer state, and other metadata to the given path.
        """
        ckpt = {
            "params": self.params,
            "opt_state": self.opt_state,
            "update_step": self.update_step,
            "dynamic_scale": self.dynamic_scale,
        }
        checkpoints.save_checkpoint(
            ckpt_dir=path,
            target=ckpt,
            step=step,
            overwrite=True,
            keep=keep,
        )

    def load(self, path: str, param_key: str = None, only_param: bool = False) -> None:
        """
        Load parameters, optimizer state, and other metadata from the given path.
        args:
            path (str): The path to the checkpoint directory.
            param_key (str): If specified, only the subset of parameters is loaded.
            only_param (bool): If True, only the parameters are loaded.
        """
        ckpt = checkpoints.restore_checkpoint(ckpt_dir=path, target=None)

        def _key_exists(d, key):
            """
            Recursively check if key exists in dictionary d.
            """
            if key in d:
                return True
            return any(_key_exists(v, key) for v in d.values() if isinstance(v, dict))

        def _recursive_replace(source, target, key):
            """
            Recursively replace the value of key in source from target.
            """
            for k, v in source.items():
                if k == key:
                    source[k] = target[k]
                elif (
                    isinstance(v, dict) and k in source and isinstance(target[k], dict)
                ):
                    _recursive_replace(source[k], target[k], key)
            return source

        if param_key:
            if not _key_exists(self.params, param_key):
                raise ValueError(f"The key '{param_key}' is missing")
            new_params = copy.deepcopy(self.params)
            new_params = _recursive_replace(new_params, ckpt["params"], param_key)
        else:
            new_params = ckpt["params"]

        if only_param:
            network = self.replace(params=new_params)
        else:
            # self.opt_state: named_tuple
            # ckpt['opt_state]: dictionary
            new_opt_state = jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(self.opt_state),
                jax.tree_util.tree_leaves(ckpt["opt_state"]),
            )

            network = self.replace(
                params=new_params,
                opt_state=new_opt_state,
                update_step=ckpt["update_step"],
                dynamic_scale=ckpt["dynamic_scale"],
            )

        return network

    def apply(self, *args, **kwargs):
        return self.network_def.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn, get_info=True) -> Tuple[Any, "Trainer"]:
        if self.dynamic_scale:
            grad_fn = self.dynamic_scale.value_and_grad(loss_fn, has_aux=True)
            dynamic_scale, is_fin, (_, info), grads = grad_fn(self.params)
        else:
            grad_fn = jax.grad(loss_fn, has_aux=True)
            grads, info = grad_fn(self.params)
            dynamic_scale = None
            is_fin = True
        # grad_norm = tree_norm(grads)
        # info["grad_norm"] = grad_norm
        info["_grads"] = grads

        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        network = self.replace(
            params=jax.tree_util.tree_map(
                partial(jnp.where, is_fin), new_params, self.params
            ),
            opt_state=jax.tree_util.tree_map(
                partial(jnp.where, is_fin), new_opt_state, self.opt_state
            ),
            update_step=self.update_step + 1,
            dynamic_scale=dynamic_scale,
        )

        return network, info
