from typing import Any

import math
import flax.linen as nn
import jax.numpy as jnp
from jax.lax import convert_element_type
from tensorflow_probability.substrates import jax as tfp

from scale_rl.agents.hyper_sac.hyper_sac_update import l2normalize
from scale_rl.networks.critics import LinearCritic
from scale_rl.networks.layers import MLPBlock, ResidualBlock
from scale_rl.networks.policies import NormalTanhPolicy
from scale_rl.networks.projection_utils import project_to_hypersphere
from scale_rl.networks.utils import he_normal_init, orthogonal_init

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
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(
            name="hyper_dense",
            features=self.hidden_dim,
            kernel_init=orthogonal_init(scale=1.0, axis=0),
            use_bias=self.use_bias,
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
            dtype=self.dtype
        )
        self.w2 = HyperDense(
            self.out_dim, 
            use_scaler=False, 
            dtype=self.dtype
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.w1(x)
        # note: when using relu,
        # it should be `x = nn.relu(x) + eps` to prevent zero vector.
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
            dtype=self.dtype
        )
        self.scaler = Scale(
            self.hidden_dim, 
            init=self.alpha_init, 
            scale=self.alpha_scale,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = self.ff(x)
        x = residual + self.scaler(x - residual)
        x = l2normalize(x, axis=-1)
        return x


class HyperEncoder(nn.Module):
    num_blocks: int
    hidden_dim: int
    input_process_type: str
    input_projection_type: str
    input_projection_constant: float
    scale_input_dense: bool
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    dtype: Any

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        info = {}
        layer_idx = 0

        if self.input_process_type == "concat_norm_linear_norm":
            x = project_to_hypersphere(
                x=x,
                projection_type=self.input_projection_type,
                constant=self.input_projection_constant,
            )
            info[f"proj_in_{layer_idx}"] = x

            x = HyperDense(
                hidden_dim=self.hidden_dim,
                dtype=self.dtype,
                use_scaler=self.scale_input_dense,
            )(x)

            x = l2normalize(x, axis=-1)
            info[f"proj_out_{layer_idx}"] = x
            layer_idx += 1

        elif self.input_process_type == "linear_concat_norm":
            x = HyperDense(
                hidden_dim=self.hidden_dim - 1,
                dtype=self.dtype,
                use_scaler=self.scale_input_dense,
            )(x)
            info[f"proj_in_{layer_idx}"] = x

            x = project_to_hypersphere(
                x=x,
                projection_type=self.input_projection_type,
                constant=self.input_projection_constant,
            )
            info[f"proj_out_{layer_idx}"] = x
            layer_idx += 1

        elif self.input_process_type == "mlp_concat_norm":
            x = HyperDense(
                hidden_dim=self.hidden_dim - 1,
                dtype=self.dtype,
                use_scaler=self.scale_input_dense,
            )(x)
            info[f"proj_in_{layer_idx}"] = x

            x = nn.relu(x) + 1e-5
            x = HyperDense(
                hidden_dim=self.hidden_dim - 1,
                dtype=self.dtype,
                use_scaler=False,
            )(x)

            x = project_to_hypersphere(
                x=x,
                projection_type=self.input_projection_type,
                constant=self.input_projection_constant,
            )
            info[f"proj_out_{layer_idx}"] = x
            layer_idx += 1

        else:
            raise ValueError

        for _ in range(self.num_blocks):
            x = HyperResidualBlock(
                hidden_dim=self.hidden_dim,
                scaler_init=self.scaler_init,
                scaler_scale=self.scaler_scale,
                alpha_init=self.alpha_init,
                alpha_scale=self.alpha_scale,
                dtype=self.dtype,
            )(x)
            info[f"res_block_{layer_idx}"] = x
            layer_idx += 1

        return x, info


class HyperNormalTanhPolicy(nn.Module):
    action_dim: int
    kernel_init_scale: float = 1.0
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    dtype: Any = jnp.float32

    separate_mean_std: bool = True
    project_in: bool = True
    hidden_dim: int = 128
    use_scaler: bool = True
    non_linear: bool = False
    use_bias: bool = True

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        temperature: float = 1.0,
    ) -> tfd.Distribution:
        if self.separate_mean_std:
            if self.project_in:
                means = HyperDense(
                    hidden_dim=self.hidden_dim,
                    dtype=self.dtype,
                    use_scaler=False,
                )(inputs)

                log_stds = HyperDense(
                    hidden_dim=self.hidden_dim,
                    dtype=self.dtype,
                    use_scaler=False,
                )(inputs)
            else:
                means = log_stds = inputs

        else:
            if self.project_in:
                means = log_stds = HyperDense(
                    hidden_dim=self.hidden_dim,
                    dtype=self.dtype,
                    use_scaler=False,
                )(inputs)

            else:
                means = log_stds = inputs

        if self.use_scaler:
            means = Scale(self.hidden_dim)(means)
            log_stds = Scale(self.hidden_dim)(log_stds)

        if self.non_linear:
            means = nn.relu(means)
            log_stds = nn.relu(log_stds)

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

        if self.use_bias:
            mean_bias = self.param(
                "mean_bias", nn.initializers.zeros, (self.action_dim,)
            )
            log_std_bias = self.param(
                "log_std_bias", nn.initializers.zeros, (self.action_dim,)
            )

            means = means + mean_bias
            log_stds = log_stds + log_std_bias

        log_stds = convert_element_type(log_stds, jnp.float32)

        # suggested by Ilya for stability
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


class HyperCritic(nn.Module):
    kernel_init_scale: float = 1.0
    dtype: Any = jnp.float32

    project_in: bool = True
    hidden_dim: int = 512
    use_scaler: bool = True
    non_linear: bool = False
    use_bias: bool = True

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
    ) -> jnp.ndarray:
        value = inputs
        if self.project_in:
            value = HyperDense(
                hidden_dim=self.hidden_dim,
                dtype=self.dtype,
                use_scaler=False,
            )(value)

        if self.use_scaler:
            value = Scale(self.hidden_dim)(value)

        if self.non_linear:
            value = nn.relu(value)

        value = HyperDense(
            hidden_dim=1,
            dtype=self.dtype,
            use_scaler=False,
        )(value)

        if self.use_bias:
            bias = self.param("value_bias", nn.initializers.zeros, (1,))
            value = value + bias

        return value


class HyperCategoricalCritic(nn.Module):
    kernel_init_scale: float = 1.0
    dtype: Any = jnp.float32

    project_in: bool = True
    hidden_dim: int = 512
    use_scaler: bool = True
    non_linear: bool = True
    use_bias: bool = True

    num_bins: int = 51

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
    ) -> jnp.ndarray:
        value = inputs
        if self.project_in:
            value = HyperDense(
                hidden_dim=self.hidden_dim,
                dtype=self.dtype,
                use_scaler=False,
            )(value)

        if self.use_scaler:
            value = Scale(self.hidden_dim)(value)

        if self.non_linear:
            value = nn.relu(value)

        value = HyperDense(
            hidden_dim=self.num_bins,
            dtype=self.dtype,
            use_scaler=False,
        )(value)

        if self.use_bias:
            bias = self.param("value_bias", nn.initializers.zeros, (self.num_bins,))
            value = value + bias

        # return log probability of bins
        return nn.log_softmax(value, axis=1)


class HyperSACDevActor(nn.Module):
    num_blocks: int
    hidden_dim: int
    action_dim: int
    input_process_type: str
    input_projection_type: str
    input_projection_constant: float
    scale_input_dense: bool
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    dtype: Any

    output_separate_mean_std: bool
    output_project_in: bool
    output_hidden_dim: int
    output_use_scaler: bool
    output_non_linear: bool
    output_use_bias: bool

    def setup(self):
        self.encoder = HyperEncoder(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            input_process_type=self.input_process_type,
            input_projection_type=self.input_projection_type,
            input_projection_constant=self.input_projection_constant,
            scale_input_dense=self.scale_input_dense,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            alpha_init=self.alpha_init,
            alpha_scale=self.alpha_scale,
            dtype=self.dtype,
        )

        self.predictor = HyperNormalTanhPolicy(
            action_dim=self.action_dim,
            dtype=self.dtype,
            separate_mean_std=self.output_separate_mean_std,
            project_in=self.output_project_in,
            hidden_dim=self.output_hidden_dim,
            use_scaler=self.output_use_scaler,
            non_linear=self.output_non_linear,
            use_bias=self.output_use_bias,
        )

    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
    ) -> tfd.Distribution:
        observations = convert_element_type(observations, self.dtype)
        z, info = self.encoder(observations)
        dist = self.predictor(z, temperature)

        return dist, info


class HyperSACDevCritic(nn.Module):
    num_blocks: int
    hidden_dim: int
    input_process_type: str
    input_projection_type: str
    input_projection_constant: float
    scale_input_dense: bool
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    dtype: Any

    output_project_in: bool
    output_hidden_dim: int
    output_use_scaler: bool
    output_non_linear: bool
    output_use_bias: bool

    def setup(self):
        self.encoder = HyperEncoder(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            input_process_type=self.input_process_type,
            input_projection_type=self.input_projection_type,
            input_projection_constant=self.input_projection_constant,
            scale_input_dense=self.scale_input_dense,
            alpha_init=self.alpha_init,
            alpha_scale=self.alpha_scale,
            dtype=self.dtype,
        )

        self.predictor = HyperCritic(
            dtype=self.dtype,
            project_in=self.output_project_in,
            hidden_dim=self.output_hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            use_scaler=self.output_use_scaler,
            non_linear=self.output_non_linear,
            use_bias=self.output_use_bias,
        )

    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        inputs = jnp.concatenate((observations, actions), axis=1)
        inputs = convert_element_type(inputs, self.dtype)
        z, info = self.encoder(inputs)
        q = self.predictor(z)
        return q, info


class HyperSACDevClippedDoubleCritic(nn.Module):
    """
    Vectorized Double-Q for Clipped Double Q-learning.
    https://arxiv.org/pdf/1802.09477v3
    """

    num_blocks: int
    hidden_dim: int
    input_process_type: str
    input_projection_type: str
    input_projection_constant: float
    scale_input_dense: bool
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    dtype: Any

    output_project_in: bool
    output_hidden_dim: int
    output_use_scaler: bool
    output_non_linear: bool
    output_use_bias: bool

    num_qs: int = 2

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        VmapCritic = nn.vmap(
            HyperSACDevCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )

        qs, info = VmapCritic(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            input_process_type=self.input_process_type,
            input_projection_type=self.input_projection_type,
            input_projection_constant=self.input_projection_constant,
            scale_input_dense=self.scale_input_dense,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            alpha_init=self.alpha_init,
            alpha_scale=self.alpha_scale,
            dtype=self.dtype,
            output_project_in=self.output_project_in,
            output_hidden_dim=self.output_hidden_dim,
            output_use_scaler=self.output_use_scaler,
            output_non_linear=self.output_non_linear,
            output_use_bias=self.output_use_bias,
        )(observations, actions)

        return qs, info


class HyperSACDevCategoricalCritic(nn.Module):
    num_blocks: int
    hidden_dim: int
    input_process_type: str
    input_projection_type: str
    input_projection_constant: float
    scale_input_dense: bool
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    dtype: Any

    output_project_in: bool
    output_hidden_dim: int
    output_use_scaler: bool
    output_non_linear: bool
    output_use_bias: bool

    num_bins: int

    def setup(self):
        self.encoder = HyperEncoder(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            input_process_type=self.input_process_type,
            input_projection_type=self.input_projection_type,
            input_projection_constant=self.input_projection_constant,
            scale_input_dense=self.scale_input_dense,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            alpha_init=self.alpha_init,
            alpha_scale=self.alpha_scale,
            dtype=self.dtype,
        )

        self.predictor = HyperCategoricalCritic(
            dtype=self.dtype,
            project_in=self.output_project_in,
            hidden_dim=self.output_hidden_dim,
            use_scaler=self.output_use_scaler,
            non_linear=self.output_non_linear,
            use_bias=self.output_use_bias,
            num_bins=self.num_bins,
        )

    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        inputs = jnp.concatenate((observations, actions), axis=1)
        inputs = convert_element_type(inputs, self.dtype)
        z, info = self.encoder(inputs)
        q_log_probs = self.predictor(z)
        return q_log_probs, info


class HyperSACDevCategoricalDoubleCritic(nn.Module):
    """
    Vectorized Double-C51 for Clipped Double Q-learning.
    https://arxiv.org/pdf/1802.09477v3
    """

    num_blocks: int
    hidden_dim: int
    input_process_type: str
    input_projection_type: str
    input_projection_constant: float
    scale_input_dense: bool
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    dtype: Any

    output_project_in: bool
    output_hidden_dim: int
    output_use_scaler: bool
    output_non_linear: bool
    output_use_bias: bool

    num_bins: int

    num_qs: int = 2

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        VmapCritic = nn.vmap(
            HyperSACDevCategoricalCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )

        qs = VmapCritic(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            input_process_type=self.input_process_type,
            input_projection_type=self.input_projection_type,
            input_projection_constant=self.input_projection_constant,
            scale_input_dense=self.scale_input_dense,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            alpha_init=self.alpha_init,
            alpha_scale=self.alpha_scale,
            dtype=self.dtype,
            output_project_in=self.output_project_in,
            output_hidden_dim=self.output_hidden_dim,
            output_use_scaler=self.output_use_scaler,
            output_non_linear=self.output_non_linear,
            output_use_bias=self.output_use_bias,
            num_bins=self.num_bins,
        )(observations, actions)

        return qs


class HyperSACDevTemperature(nn.Module):
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
