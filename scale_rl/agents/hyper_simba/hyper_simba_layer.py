from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from jax.lax import convert_element_type
from tensorflow_probability.substrates import jax as tfp

from scale_rl.agents.hyper_simba.hyper_simba_update import l2normalize
from scale_rl.networks.projection_utils import project_to_hypersphere
from scale_rl.networks.utils import orthogonal_init

tfd = tfp.distributions
tfb = tfp.bijectors


class Scale(nn.Module):
    dim: int
    init: float = 1.0
    scale: float = 1.0

    """
    init: value one would like to multiply to the input x.
    scale: value to control the scale of the gradient.
    """

    def setup(self):
        self.scaler = self.param(
            "scaler",
            nn.initializers.constant(1.0 * self.scale),
            self.dim,
        )
        self.forward_scaler = self.init / self.scale

    def __call__(self, x):
        return self.scaler * self.forward_scaler * x


class HyperDense(nn.Module):
    hidden_dim: int
    dtype: Any
    use_scaler: bool
    scaler_init: float = 1.0
    scaler_scale: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(
            name="hyper_dense",
            features=self.hidden_dim,
            kernel_init=orthogonal_init(scale=1.0, axis=0),
            use_bias=False,  # mandatory
            dtype=self.dtype,
        )(x)
        if self.use_scaler:
            x = Scale(self.hidden_dim, init=self.scaler_init, scale=self.scaler_scale)(
                x
            )

        return x


class HyperFeedForward(nn.Module):
    hidden_dim: int
    out_dim: int
    scaler_init: float
    scaler_scale: float
    dtype: Any
    eps: float = 1e-8

    def setup(self):
        self.w1 = HyperDense(
            self.hidden_dim,
            use_scaler=True,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            dtype=self.dtype,
        )
        self.w2 = HyperDense(self.out_dim, use_scaler=False, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.w1(x)
        # `eps` is required to prevent zero vector.
        x = nn.relu(x) + self.eps
        x = self.w2(x)
        x = l2normalize(x, axis=-1)
        return x


class HyperResidualBlock(nn.Module):
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    dtype: Any

    def setup(self):
        self.ff = HyperFeedForward(
            hidden_dim=self.hidden_dim * 4,
            out_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            dtype=self.dtype,
        )
        self.alpha_scaler = Scale(
            self.hidden_dim,
            init=self.alpha_init,
            scale=self.alpha_scale,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = self.ff(x)
        x = residual + self.alpha_scaler(x - residual)
        x = l2normalize(x, axis=-1)
        return x


class HyperEncoder(nn.Module):
    num_blocks: int
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    dtype: Any

    input_projection_type: str = "shift"
    input_projection_constant: float = 3.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = project_to_hypersphere(
            x=x,
            projection_type=self.input_projection_type,
            constant=self.input_projection_constant,
        )
        x = HyperDense(
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
            use_scaler=True,
        )(x)
        x = l2normalize(x, axis=-1)

        for _ in range(self.num_blocks):
            x = HyperResidualBlock(
                hidden_dim=self.hidden_dim,
                scaler_init=self.scaler_init,
                scaler_scale=self.scaler_scale,
                alpha_init=self.alpha_init,
                alpha_scale=self.alpha_scale,
                dtype=self.dtype,
            )(x)

        return x


class HyperNormalTanhPolicy(nn.Module):
    hidden_dim: int
    action_dim: int
    kernel_init_scale: float = 1.0
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        temperature: float = 1.0,
    ) -> tfd.Distribution:
        # mean
        means = HyperDense(
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
            use_scaler=True,
        )(inputs)

        log_stds = HyperDense(
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
            use_scaler=True,
        )(inputs)

        means = HyperDense(
            hidden_dim=self.action_dim,
            dtype=self.dtype,
            use_scaler=False,
        )(means)

        log_stds = HyperDense(
            hidden_dim=self.action_dim,
            dtype=self.dtype,
            use_scaler=False,
        )(log_stds)

        mean_bias = self.param("mean_bias", nn.initializers.zeros, (self.action_dim,))
        log_std_bias = self.param(
            "log_std_bias", nn.initializers.zeros, (self.action_dim,)
        )

        means = means + mean_bias

        log_stds = log_stds + log_std_bias

        # normalize log-stds for stability
        log_stds = convert_element_type(log_stds, jnp.float32)
        log_stds = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
            1 + nn.tanh(log_stds)
        )

        # N(mu, exp(log_sigma))
        dist = tfd.MultivariateNormalDiag(
            loc=convert_element_type(means, jnp.float32),
            scale_diag=jnp.exp(log_stds) * temperature,
        )

        # tanh(N(mu, sigma))
        dist = tfd.TransformedDistribution(distribution=dist, bijector=tfb.Tanh())

        return dist


class HyperCategoricalValue(nn.Module):
    hidden_dim: int
    num_bins: int = 101
    kernel_init_scale: float = 1.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
    ) -> jnp.ndarray:
        value = HyperDense(
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
            use_scaler=True,
        )(inputs)

        value = HyperDense(
            hidden_dim=self.num_bins,
            dtype=self.dtype,
            use_scaler=False,
        )(value)

        bias = self.param("value_bias", nn.initializers.zeros, (self.num_bins,))
        value = value + bias

        # return log probability of bins
        return nn.log_softmax(value, axis=1)
