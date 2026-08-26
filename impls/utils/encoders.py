import functools
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import MLP


class ResnetStack(nn.Module):
    """ResNet stack module."""

    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
        )(x)

        if self.max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out


class TokenPooler(nn.Module):
    """Attention pooler over gaussian-token sets (mirror of the nwm torch pooler).

    Accepts either a single token set (last dim `token_dim`) or GCEncoder's channel-concatenated obs+goal pair
    (last dim `2 * token_dim`), which is split back into two sets, tagged with a type embedding, and pooled jointly.

    Attributes:
        token_dim: Per-token feature width of one set.
        d_model: Width of the shared projection and of the attention.
        num_queries: Number of learned pooling queries.
        num_heads: Number of attention heads.
        out_dim: Output width.
    """

    token_dim: int = 14
    d_model: int = 128
    num_queries: int = 8
    num_heads: int = 4
    out_dim: int = 512

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        num_sets, rem = divmod(x.shape[-1], self.token_dim)
        assert rem == 0 and num_sets in (1, 2), f'expected 1 or 2 sets of {self.token_dim} channels, got {x.shape[-1]}'
        x = x.astype(jnp.float32)

        proj = nn.Dense(self.d_model, name='proj')
        type_emb = self.param('type_emb', nn.initializers.normal(0.02), (2, self.d_model))
        sets = [proj(x[..., i * self.token_dim : (i + 1) * self.token_dim]) + type_emb[i] for i in range(num_sets)]
        kv = nn.LayerNorm(name='kv_norm')(jnp.concatenate(sets, axis=-2))

        queries = self.param('queries', nn.initializers.normal(0.02), (self.num_queries, self.d_model))
        queries = jnp.broadcast_to(queries, (*kv.shape[:-2], self.num_queries, self.d_model))
        out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            name='attn',
        )(queries, kv)

        out = out.reshape((*out.shape[:-2], self.num_queries * self.d_model))
        return nn.gelu(nn.Dense(self.out_dim, name='out')(out))


class GCEncoder(nn.Module):
    """Helper module to handle inputs to goal-conditioned networks.

    It takes in observations (s) and goals (g) and returns the concatenation of `state_encoder(s)`, `goal_encoder(g)`,
    and `concat_encoder([s, g])`. It ignores the encoders that are not provided. This way, the module can handle both
    early and late fusion (or their variants) of state and goal information.
    """

    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None
    concat_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations, goals=None, goal_encoded=False):
        """Returns the representations of observations and goals.

        If `goal_encoded` is True, `goals` is assumed to be already encoded representations. In this case, either
        `goal_encoder` or `concat_encoder` must be None.
        """
        reps = []
        if self.state_encoder is not None:
            reps.append(self.state_encoder(observations))
        if goals is not None:
            if goal_encoded:
                # Can't have both goal_encoder and concat_encoder in this case.
                assert self.goal_encoder is None or self.concat_encoder is None
                reps.append(goals)
            else:
                if self.goal_encoder is not None:
                    reps.append(self.goal_encoder(goals))
                if self.concat_encoder is not None:
                    reps.append(self.concat_encoder(jnp.concatenate([observations, goals], axis=-1)))
        reps = jnp.concatenate(reps, axis=-1)
        return reps


encoder_modules = {
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
    # Token modality: GCDataset.augment only touches ndim==4 arrays, so token batches skip random cropping anyway;
    # still run these with --agent.p_aug=0.0 so the augmentation branch never fires.
    'token_pooler': TokenPooler,
}
