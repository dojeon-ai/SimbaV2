import flax.linen as nn
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from scale_rl.agents.simba.simba_layer import (
    LinearCritic,
    NormalTanhPolicy,
    PreLNResidualBlock,
)

tfd = tfp.distributions
tfb = tfp.bijectors


class SimbaActor(nn.Module):
    num_blocks: int
    hidden_dim: int
    action_dim: int

    def setup(self):
        self.embedder = nn.Dense(
            self.hidden_dim, kernel_init=nn.initializers.orthogonal(1.0)
        )
        self.encoder = nn.Sequential(
            [
                *[PreLNResidualBlock(hidden_dim=self.hidden_dim) 
                  for _ in range(self.num_blocks)], 
                nn.LayerNorm(),
            ]
        )
        self.predictor = NormalTanhPolicy(self.action_dim)

    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
    ) -> tfd.Distribution:
        x = observations
        y = self.embedder(x)
        z = self.encoder(y)
        dist, info = self.predictor(z, temperature)
        return dist, info


class SimbaCritic(nn.Module):
    num_blocks: int
    hidden_dim: int

    def setup(self):
        self.embedder = nn.Dense(
            self.hidden_dim, kernel_init=nn.initializers.orthogonal(1.0)
        )
        self.encoder = nn.Sequential(
            [
                *[PreLNResidualBlock(hidden_dim=self.hidden_dim) 
                  for _ in range(self.num_blocks)], 
                nn.LayerNorm(),
            ]
        )
        self.predictor = LinearCritic()

    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        x = jnp.concatenate((observations, actions), axis=1)
        y = self.embedder(x)
        z = self.encoder(y)
        q, info = self.predictor(z)
        return q, info


class SimbaDoubleCritic(nn.Module):
    """
    Vectorized Double-Q for Clipped Double Q-learning.
    https://arxiv.org/pdf/1802.09477v3
    """

    num_blocks: int
    hidden_dim: int

    num_qs: int = 2

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        VmapCritic = nn.vmap(
            SimbaCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )

        qs, infos = VmapCritic(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
        )(observations, actions)

        return qs, infos


class SimbaTemperature(nn.Module):
    initial_value: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            name="log_temp",
            init_fn=lambda key: jnp.full(
                shape=(), fill_value=jnp.log(self.initial_value)
            ),
        )
        return jnp.exp(log_temp)
