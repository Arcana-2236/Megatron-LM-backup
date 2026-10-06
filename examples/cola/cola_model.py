# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Isolated TP=1 CoLA layer spec, adapted from ATC-Megatron's nonlinear bottlenecks."""

import torch
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules


def _check_config(config, rank, is_expert):
    if (
        any(
            size != 1
            for size in (
                config.tensor_model_parallel_size,
                config.pipeline_model_parallel_size,
                config.context_parallel_size,
            )
        )
        or config.sequence_parallel
    ):
        raise ValueError("CoLA currently requires TP=PP=CP=1 and no sequence parallelism.")
    if is_expert or config.num_moe_experts is not None:
        raise ValueError("CoLA currently supports dense layers only.")
    if not config.gated_linear_unit or config.activation_func is not F.silu:
        raise ValueError("CoLA requires SwiGLU (--swiglu).")
    if config.normalization != "RMSNorm" or config.add_bias_linear or config.add_qkv_bias:
        raise ValueError("CoLA requires RMSNorm and bias-free projections.")
    for feature in (
        "fp8",
        "fp4",
        "multi_latent_attention",
        "attention_output_gate",
        "qk_layernorm",
        "qk_l2_norm",
        "use_te_activation_func",
        "use_kitchen",
        "delay_wgrad_compute",
    ):
        if getattr(config, feature, False):
            raise ValueError(f"CoLA does not support {feature}.")
    rank = config.hidden_size // 4 if rank is None else rank
    if rank <= 0:
        raise ValueError("CoLA bottleneck rank must be positive.")
    return rank


def _column(input_size, output_size, config, init_method, tp_group):
    return ColumnParallelLinear(
        input_size,
        output_size,
        config=config,
        init_method=init_method,
        bias=False,
        gather_output=False,
        skip_bias_add=False,
        tp_group=tp_group,
    )


class _BranchedProjection(MegatronModule):
    """A fused input bottleneck followed by independently parameterized branches."""

    def __init__(self, input_size, output_sizes, *, config, init_method, rank, tp_group):
        super().__init__(config=config)
        self.input_projection = _column(
            input_size, len(output_sizes) * rank, config, init_method, tp_group
        )
        self.output_projections = torch.nn.ModuleList(
            _column(rank, size, config, init_method, tp_group) for size in output_sizes
        )

    def _branches(self, x):
        hidden, _ = self.input_projection(x)
        inputs = F.silu(hidden).chunk(len(self.output_projections), dim=-1)
        return [projection(part)[0] for projection, part in zip(self.output_projections, inputs)]

    def backward_dw(self):
        for projection in reversed(self.output_projections):
            projection.backward_dw()
        self.input_projection.backward_dw()


class CoLAQKVLinear(_BranchedProjection):
    """Q/K/V bottlenecks; return the query-group-interleaved layout Megatron consumes."""

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config,
        init_method,
        bias,
        rank=None,
        tp_group=None,
        is_expert=False,
        **kwargs,
    ):
        rank = _check_config(config, rank, is_expert)
        if bias:
            raise ValueError("CoLA projections are bias-free.")
        q_size = config.num_attention_heads * config.kv_channels
        kv_size = config.num_query_groups * config.kv_channels
        if output_size != q_size + 2 * kv_size:
            raise ValueError("Unexpected CoLA QKV projection size.")
        super().__init__(
            input_size,
            (q_size, kv_size, kv_size),
            config=config,
            init_method=init_method,
            rank=rank,
            tp_group=tp_group,
        )
        self.num_query_groups = config.num_query_groups

    def forward(self, x):
        query, key, value = self._branches(x)
        grouped_shape = (*x.shape[:-1], self.num_query_groups, -1)
        # Each group contains its query heads, then one K head and one V head.
        packed = torch.cat([part.reshape(grouped_shape) for part in (query, key, value)], dim=-1)
        return packed.flatten(-2), None


class CoLAFC1Linear(_BranchedProjection):
    """Separate gate/up expansions after a fused, SiLU-activated input bottleneck."""

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config,
        init_method,
        bias,
        rank=None,
        tp_group=None,
        is_expert=False,
        **kwargs,
    ):
        rank = _check_config(config, rank, is_expert)
        if bias or output_size % 2:
            raise ValueError("CoLA FC1 requires bias=False and an even output size.")
        super().__init__(
            input_size,
            (output_size // 2, output_size // 2),
            config=config,
            init_method=init_method,
            rank=rank,
            tp_group=tp_group,
        )

    def forward(self, x):
        # The existing MLP applies the outer SwiGLU to this gate/up concatenation.
        return torch.cat(self._branches(x), dim=-1), None


class CoLAOutputLinear(MegatronModule):
    """Input -> rank -> SiLU -> output, for attention output or FFN down projection."""

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config,
        init_method,
        bias,
        rank=None,
        tp_group=None,
        is_expert=False,
        **kwargs,
    ):
        rank = _check_config(config, rank, is_expert)
        if bias:
            raise ValueError("CoLA projections are bias-free.")
        super().__init__(config=config)
        self.input_projection = RowParallelLinear(
            input_size,
            rank,
            config=config,
            init_method=init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            tp_group=tp_group,
        )
        self.output_projection = _column(rank, output_size, config, init_method, tp_group)

    def forward(self, x):
        hidden, _ = self.input_projection(x)
        return self.output_projection(F.silu(hidden))

    def backward_dw(self):
        self.output_projection.backward_dw()
        self.input_projection.backward_dw()


def get_cola_layer_spec(attn_rank=None, mlp_rank=None):
    """Reuse GPT/TE attention and MLP logic, replacing only their linear projections."""
    for rank in (attn_rank, mlp_rank):
        if rank is not None and rank <= 0:
            raise ValueError("CoLA bottleneck rank must be positive.")
    spec = get_gpt_layer_with_transformer_engine_spec()
    layer = spec.submodules
    # Original TE QKV/FC1 combine norm+linear. Restore explicit norms before replacing them.
    layer.input_layernorm = TENorm
    layer.pre_mlp_layernorm = TENorm
    attention = layer.self_attention.submodules
    attention.linear_qkv = ModuleSpec(CoLAQKVLinear, params={"rank": attn_rank})
    attention.linear_proj = ModuleSpec(CoLAOutputLinear, params={"rank": attn_rank})
    mlp = get_submodules(layer.mlp)
    mlp.linear_fc1 = ModuleSpec(CoLAFC1Linear, params={"rank": mlp_rank})
    mlp.linear_fc2 = ModuleSpec(CoLAOutputLinear, params={"rank": mlp_rank})
    layer.sharded_state_dict_keys_map = {}
    return spec


# Also usable directly with pretrain_gpt.py --spec examples.cola.cola_model layer_spec.
layer_spec = get_cola_layer_spec()
