from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from jax.lax import convert_element_type
from tensorflow_probability.substrates import jax as tfp

from scale_rl.agents.hyper_sac.hyper_sac_layer import (
    HyperEncoder,
    HyperNormalTanhPolicy, 
    HyperCategoricalValue,
)

tfd = tfp.distributions
tfb = tfp.bijectors


class HyperSACActor(nn.Module):
    num_blocks: int
    hidden_dim: int
    action_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    dtype: Any

    def setup(self):
        self.encoder = HyperEncoder(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            alpha_init=self.alpha_init,
            alpha_scale=self.alpha_scale,
            dtype=self.dtype,
        )
        self.predictor = HyperNormalTanhPolicy(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
        )

    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
    ) -> tfd.Distribution:
        observations = convert_element_type(observations, self.dtype)
        z = self.encoder(observations)
        dist = self.predictor(z, temperature)
        
        info = {}
        return dist, info


class HyperSACCritic(nn.Module):
    num_blocks: int
    hidden_dim: int
    min_v: float
    max_v: float
    num_bins: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    dtype: Any

    def setup(self):
        self.encoder = HyperEncoder(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            alpha_init=self.alpha_init,
            alpha_scale=self.alpha_scale,
            dtype=self.dtype,
        )

        self.predictor = HyperCategoricalValue(
            hidden_dim=self.hidden_dim,
            num_bins=self.num_bins,
            dtype=self.dtype,
        )

    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        inputs = jnp.concatenate((observations, actions), axis=1)
        inputs = convert_element_type(inputs, self.dtype)
        z = self.encoder(inputs)
        q_log_prob = self.predictor(z)
        info = {}
        return q_log_prob, info


class HyperSACDoubleCritic(nn.Module):
    """
    Vectorized Double-Q for Clipped Double Q-learning.
    https://arxiv.org/pdf/1802.09477v3
    """

    num_blocks: int
    min_v: float
    max_v: float
    num_bins: int    
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    dtype: Any

    num_qs: int = 2

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        VmapCritic = nn.vmap(
            HyperSACCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )

        q_log_probs = VmapCritic(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            min_v=self.min_v,
            max_v=self.max_v,
            num_bins=self.num_bins,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            alpha_init=self.alpha_init,
            alpha_scale=self.alpha_scale,
            dtype=self.dtype,
        )(observations, actions)

        return q_log_probs


class HyperSACTemperature(nn.Module):
    initial_value: float = 0.01

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            name="log_temp",
            init_fn=lambda key: jnp.full(
                shape=(), fill_value=jnp.log(self.initial_value)
            ),
        )
        return jnp.exp(log_temp)
