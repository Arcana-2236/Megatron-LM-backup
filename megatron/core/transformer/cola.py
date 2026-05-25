# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import torch
import torch.nn.functional as F

from megatron.core import tensor_parallel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import Attention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_range_pop, nvtx_range_push


def _default_rank(config: TransformerConfig, attr: str) -> int:
    rank = getattr(config, attr, None)
    return rank if rank is not None else config.hidden_size // 4


class CoLAMLP(MegatronModule):
    """Low-rank CoLA MLP for SwiGLU GPT layers."""

    def __init__(
        self,
        config: TransformerConfig,
        *,
        pg_collection: ProcessGroupCollection,
        is_expert: bool = False,
        name: str | None = None,
    ):
        super().__init__(config=config)
        if is_expert:
            raise NotImplementedError("CoLA MLP does not support MoE experts.")
        if config.sequence_parallel:
            raise NotImplementedError("CoLA MLP does not support sequence parallelism.")
        if not config.gated_linear_unit or config.activation_func is not F.silu:
            raise ValueError("CoLA MLP requires llama-style SwiGLU: pass --swiglu.")

        self.config = config
        self.mlp_rank = _default_rank(config, "mlp_rank")
        self.tp_group = pg_collection.tp
        bias = config.add_bias_linear

        self.cola_h_to_2r = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            2 * self.mlp_rank,
            config=config,
            init_method=config.init_method,
            bias=bias,
            gather_output=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="cola_fc1_down",
            tp_group=self.tp_group,
            name=(name + ".cola_h_to_2r") if name is not None else None,
        )
        self.cola_gate_r_to_dff = tensor_parallel.ColumnParallelLinear(
            self.mlp_rank,
            config.ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=bias,
            gather_output=False,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="cola_gate_up",
            tp_group=self.tp_group,
            name=(name + ".cola_gate_r_to_dff") if name is not None else None,
        )
        self.cola_up_r_to_dff = tensor_parallel.ColumnParallelLinear(
            self.mlp_rank,
            config.ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=bias,
            gather_output=False,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="cola_up_up",
            tp_group=self.tp_group,
            name=(name + ".cola_up_r_to_dff") if name is not None else None,
        )
        self.cola_dff_to_r = tensor_parallel.RowParallelLinear(
            config.ffn_hidden_size,
            self.mlp_rank,
            config=config,
            init_method=config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="cola_fc2_down",
            tp_group=self.tp_group,
            name=(name + ".cola_dff_to_r") if name is not None else None,
        )
        self.cola_r_to_h = tensor_parallel.ColumnParallelLinear(
            self.mlp_rank,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=bias,
            gather_output=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="cola_fc2_up",
            tp_group=self.tp_group,
            name=(name + ".cola_r_to_h") if name is not None else None,
        )

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        del kwargs
        nvtx_range_push(suffix="cola_mlp_up")
        rank_hidden, rank_bias = self.cola_h_to_2r(hidden_states)
        if rank_bias is not None:
            rank_hidden = rank_hidden + rank_bias
        gate_rank, up_rank = torch.chunk(F.silu(rank_hidden), 2, dim=-1)

        gate, gate_bias = self.cola_gate_r_to_dff(gate_rank)
        up, up_bias = self.cola_up_r_to_dff(up_rank)
        if gate_bias is not None:
            gate = gate + gate_bias
        if up_bias is not None:
            up = up + up_bias
        intermediate_parallel = F.silu(gate) * up
        nvtx_range_pop(suffix="cola_mlp_up")

        nvtx_range_push(suffix="cola_mlp_down")
        down_rank, down_rank_bias = self.cola_dff_to_r(intermediate_parallel)
        if down_rank_bias is not None:
            down_rank = down_rank + down_rank_bias
        output, output_bias = self.cola_r_to_h(F.silu(down_rank))
        nvtx_range_pop(suffix="cola_mlp_down")
        if output_bias is not None:
            output = output + output_bias
            output_bias = None
        return output, output_bias

    @classmethod
    def as_mlp_submodule(
        cls,
        *,
        config: TransformerConfig,
        pg_collection: ProcessGroupCollection,
        is_mtp_layer: bool,
        is_expert: bool = False,
        input_size: int | None = None,
        ffn_hidden_size: int | None = None,
        name: str | None = None,
    ) -> "CoLAMLP":
        del is_mtp_layer, input_size, ffn_hidden_size
        return cls(config=config, pg_collection=pg_collection, is_expert=is_expert, name=name)


class CoLAAttentionOutput(MegatronModule):
    """Low-rank attention output projection with the standard linear_proj interface."""

    def __init__(
        self,
        query_projection_size: int,
        hidden_size: int,
        *,
        config: TransformerConfig,
        pg_collection: ProcessGroupCollection,
        name: str | None = None,
    ):
        super().__init__(config=config)
        if query_projection_size != hidden_size:
            raise NotImplementedError("CoLA attention output expects query projection size == hidden size.")
        self.attn_rank = _default_rank(config, "attn_rank")
        self.cola_out_d_to_r = tensor_parallel.RowParallelLinear(
            query_projection_size,
            self.attn_rank,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="cola_attn_out_down",
            tp_group=pg_collection.tp,
            name=(name + ".cola_out_d_to_r") if name is not None else None,
        )
        self.cola_out_r_to_d = tensor_parallel.ColumnParallelLinear(
            self.attn_rank,
            hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            gather_output=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="cola_attn_out_up",
            tp_group=pg_collection.tp,
            name=(name + ".cola_out_r_to_d") if name is not None else None,
        )

    def forward(self, hidden_states: torch.Tensor):
        down_rank, down_rank_bias = self.cola_out_d_to_r(hidden_states)
        if down_rank_bias is not None:
            down_rank = down_rank + down_rank_bias
        output, output_bias = self.cola_out_r_to_d(F.silu(down_rank))
        if output_bias is not None:
            output = output + output_bias
            output_bias = None
        return output, output_bias

    def backward_dw(self) -> None:
        self.cola_out_r_to_d.backward_dw()
        self.cola_out_d_to_r.backward_dw()


class CoLASelfAttention(Attention):
    """Low-rank CoLA self-attention for the DP-only benchmark path."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        pp_layer_offset: int | None = None,
        name: str | None = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            name=name,
        )
        if self.world_size != 1:
            raise NotImplementedError("CoLA attention currently supports tensor parallel size 1.")
        if config.num_query_groups != config.num_attention_heads:
            raise NotImplementedError("CoLA attention currently supports MHA only, not GQA.")
        if config.sequence_parallel:
            raise NotImplementedError("CoLA attention does not support sequence parallelism.")
        self.q_layernorm = None
        self.k_layernorm = None

        self.attn_rank = _default_rank(config, "attn_rank")
        bias = config.add_bias_linear or config.add_qkv_bias
        self.linear_proj = CoLAAttentionOutput(
            self.query_projection_size,
            config.hidden_size,
            config=config,
            pg_collection=self.pg_collection,
            name=(name + ".linear_proj") if name is not None else None,
        )

        self.cola_d_to_3r = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            3 * self.attn_rank,
            config=config,
            init_method=config.init_method,
            bias=bias,
            gather_output=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="cola_qkv_down",
            tp_group=self.pg_collection.tp,
            name=(name + ".cola_d_to_3r") if name is not None else None,
        )
        self.cola_q_r_to_proj = tensor_parallel.ColumnParallelLinear(
            self.attn_rank,
            self.query_projection_size,
            config=config,
            init_method=config.init_method,
            bias=bias,
            gather_output=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="cola_q_up",
            tp_group=self.pg_collection.tp,
            name=(name + ".cola_q_r_to_proj") if name is not None else None,
        )
        self.cola_k_r_to_kv = tensor_parallel.ColumnParallelLinear(
            self.attn_rank,
            self.kv_projection_size,
            config=config,
            init_method=config.init_method,
            bias=bias,
            gather_output=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="cola_k_up",
            tp_group=self.pg_collection.tp,
            name=(name + ".cola_k_r_to_kv") if name is not None else None,
        )
        self.cola_v_r_to_kv = tensor_parallel.ColumnParallelLinear(
            self.attn_rank,
            self.kv_projection_size,
            config=config,
            init_method=config.init_method,
            bias=bias,
            gather_output=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="cola_v_up",
            tp_group=self.pg_collection.tp,
            name=(name + ".cola_v_r_to_kv") if name is not None else None,
        )

    def get_query_key_value_tensors(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        output_gate: bool = False,
        split_qkv: bool = True,
    ):
        if key_value_states is not None:
            raise NotImplementedError("CoLA self-attention does not support cross-attention.")
        if output_gate:
            raise NotImplementedError("CoLA attention does not support attention output gates.")
        if not split_qkv:
            raise NotImplementedError("CoLA attention does not support fused qkv+rope.")

        qkv_rank, qkv_rank_bias = self.cola_d_to_3r(hidden_states)
        if qkv_rank_bias is not None:
            qkv_rank = qkv_rank + qkv_rank_bias
        q_rank, k_rank, v_rank = torch.chunk(F.silu(qkv_rank), 3, dim=-1)

        query, query_bias = self.cola_q_r_to_proj(q_rank)
        key, key_bias = self.cola_k_r_to_kv(k_rank)
        value, value_bias = self.cola_v_r_to_kv(v_rank)
        if query_bias is not None:
            query = query + query_bias
        if key_bias is not None:
            key = key + key_bias
        if value_bias is not None:
            value = value + value_bias

        query = query.view(
            query.size(0),
            query.size(1),
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        key = key.view(
            key.size(0),
            key.size(1),
            self.num_query_groups_per_partition,
            self.hidden_size_per_attention_head,
        )
        value = value.view(
            value.size(0),
            value.size(1),
            self.num_query_groups_per_partition,
            self.hidden_size_per_attention_head,
        )
        return query, key, value

    def backward_dw(self) -> None:
        self.linear_proj.backward_dw()
