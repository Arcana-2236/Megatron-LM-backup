# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Initial CoLA linear wrappers for MCore GPT layers."""

from typing import Callable

import torch
import torch.nn.functional as F

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_pg_size

COLA_LINEAR_PLACEMENT_PATTERN = "column_gather-column_shard-row-column_gather"


def _check_cola_config(config: TransformerConfig, tp_group: torch.distributed.ProcessGroup) -> None:
    if config.cola_linear_placement_pattern != COLA_LINEAR_PLACEMENT_PATTERN:
        raise NotImplementedError(
            "Initial CoLA support only implements "
            f"{COLA_LINEAR_PLACEMENT_PATTERN}, got {config.cola_linear_placement_pattern}."
        )
    if get_pg_size(tp_group) != 1:
        raise NotImplementedError("Initial CoLA support is limited to tensor_model_parallel_size=1.")


def _cola_rank(rank: int | None, hidden_size: int, name: str) -> int:
    if rank is None:
        rank = hidden_size // 4
    if rank <= 0:
        raise ValueError(f"{name} must be positive, got {rank}.")
    return rank


def _add_bias(output: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    return output if bias is None else output + bias


class CoLAFC1Linear(torch.nn.Module):
    """CoLA replacement for MLP ``linear_fc1`` in the TP=1 first pass."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        bias: bool,
        gather_output: bool,
        skip_bias_add: bool,
        is_expert: bool = False,
        tp_comm_buffer_name: str | None = None,
        tp_group: torch.distributed.ProcessGroup | None = None,
        stride: int = 1,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__()
        del gather_output, skip_bias_add, tp_comm_buffer_name, stride, name, kwargs
        if is_expert:
            raise NotImplementedError("Initial CoLA support does not cover MoE experts.")
        if not config.gated_linear_unit:
            raise ValueError("CoLA MLP requires gated_linear_unit=True, e.g. pass --swiglu.")
        if output_size % 2 != 0:
            raise ValueError(f"CoLA MLP fc1 output_size must be even, got {output_size}.")

        rank = _cola_rank(config.cola_mlp_rank, config.hidden_size, "cola_mlp_rank")
        ffn_hidden_size = output_size // 2

        self.cola_h_to_2r = ColumnParallelLinear(
            input_size,
            2 * rank,
            config=config,
            init_method=init_method,
            bias=bias,
            gather_output=True,
            skip_bias_add=True,
            tp_group=tp_group,
        )
        _check_cola_config(config, self.cola_h_to_2r.tp_group)
        self.cola_gate_r_to_dff = ColumnParallelLinear(
            rank,
            ffn_hidden_size,
            config=config,
            init_method=init_method,
            bias=bias,
            gather_output=False,
            skip_bias_add=True,
            tp_group=tp_group,
        )
        self.cola_up_r_to_dff = ColumnParallelLinear(
            rank,
            ffn_hidden_size,
            config=config,
            init_method=init_method,
            bias=bias,
            gather_output=False,
            skip_bias_add=True,
            tp_group=tp_group,
        )

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply hidden -> rank -> SwiGLU FC1 projections."""
        del weight, runtime_gather_output, kwargs
        rank_hidden, rank_bias = self.cola_h_to_2r(input_)
        rank_hidden = F.silu(_add_bias(rank_hidden, rank_bias))
        gate_rank, up_rank = torch.chunk(rank_hidden, 2, dim=-1)

        gate, gate_bias = self.cola_gate_r_to_dff(gate_rank)
        up, up_bias = self.cola_up_r_to_dff(up_rank)
        return torch.cat((_add_bias(gate, gate_bias), _add_bias(up, up_bias)), dim=-1), None

    def backward_dw(self) -> None:
        """Compute delayed weight gradients for child linears if enabled."""
        self.cola_up_r_to_dff.backward_dw()
        self.cola_gate_r_to_dff.backward_dw()
        self.cola_h_to_2r.backward_dw()


class CoLARowThenColumnLinear(torch.nn.Module):
    """CoLA row -> column(gather) projection used by MLP and attention outputs."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool = False,
        tp_comm_buffer_name: str | None = None,
        tp_group: torch.distributed.ProcessGroup | None = None,
        stride: int = 1,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__()
        del input_is_parallel, tp_comm_buffer_name, stride, name, kwargs
        if is_expert:
            raise NotImplementedError("Initial CoLA support does not cover MoE experts.")

        rank = _cola_rank(config.cola_mlp_rank, config.hidden_size, "cola_mlp_rank")
        self.cola_in_to_rank = RowParallelLinear(
            input_size,
            rank,
            config=config,
            init_method=init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=True,
            tp_group=tp_group,
        )
        _check_cola_config(config, self.cola_in_to_rank.tp_group)
        self.cola_rank_to_out = ColumnParallelLinear(
            rank,
            output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            gather_output=True,
            skip_bias_add=skip_bias_add,
            tp_group=tp_group,
        )

    def forward(self, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply row projection, SiLU rank activation, then gathered column projection."""
        rank_hidden, rank_bias = self.cola_in_to_rank(input_)
        rank_hidden = F.silu(_add_bias(rank_hidden, rank_bias))
        return self.cola_rank_to_out(rank_hidden)

    def backward_dw(self) -> None:
        """Compute delayed weight gradients for child linears if enabled."""
        self.cola_rank_to_out.backward_dw()
        self.cola_in_to_rank.backward_dw()


class CoLAQKVLinear(torch.nn.Module):
    """CoLA replacement for self-attention QKV projection in the TP=1 first pass."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        bias: bool,
        gather_output: bool,
        skip_bias_add: bool,
        is_expert: bool = False,
        tp_comm_buffer_name: str | None = None,
        tp_group: torch.distributed.ProcessGroup | None = None,
        stride: int = 1,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__()
        del gather_output, skip_bias_add, is_expert, tp_comm_buffer_name, stride, name, kwargs
        if config.attention_output_gate:
            raise NotImplementedError("Initial CoLA support does not cover attention_output_gate.")

        rank = _cola_rank(config.cola_attn_rank, config.hidden_size, "cola_attn_rank")
        query_projection_size = config.kv_channels * config.num_attention_heads
        kv_projection_size = (output_size - query_projection_size) // 2
        if output_size != query_projection_size + 2 * kv_projection_size:
            raise ValueError(
                f"CoLA QKV output size {output_size} is not compatible with Q + 2KV projection."
            )

        self.cola_h_to_3r = ColumnParallelLinear(
            input_size,
            3 * rank,
            config=config,
            init_method=init_method,
            bias=bias,
            gather_output=True,
            skip_bias_add=True,
            tp_group=tp_group,
        )
        _check_cola_config(config, self.cola_h_to_3r.tp_group)
        self.cola_q_r_to_proj = ColumnParallelLinear(
            rank,
            query_projection_size,
            config=config,
            init_method=init_method,
            bias=bias,
            gather_output=False,
            skip_bias_add=True,
            tp_group=tp_group,
        )
        self.cola_k_r_to_kv = ColumnParallelLinear(
            rank,
            kv_projection_size,
            config=config,
            init_method=init_method,
            bias=bias,
            gather_output=False,
            skip_bias_add=True,
            tp_group=tp_group,
        )
        self.cola_v_r_to_kv = ColumnParallelLinear(
            rank,
            kv_projection_size,
            config=config,
            init_method=init_method,
            bias=bias,
            gather_output=False,
            skip_bias_add=True,
            tp_group=tp_group,
        )

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply hidden -> rank -> QKV projections."""
        del weight, runtime_gather_output, kwargs
        qkv_rank, qkv_rank_bias = self.cola_h_to_3r(input_)
        qkv_rank = F.silu(_add_bias(qkv_rank, qkv_rank_bias))
        q_rank, k_rank, v_rank = torch.chunk(qkv_rank, 3, dim=-1)

        query, query_bias = self.cola_q_r_to_proj(q_rank)
        key, key_bias = self.cola_k_r_to_kv(k_rank)
        value, value_bias = self.cola_v_r_to_kv(v_rank)
        return (
            torch.cat(
                (
                    _add_bias(query, query_bias),
                    _add_bias(key, key_bias),
                    _add_bias(value, value_bias),
                ),
                dim=-1,
            ),
            None,
        )

    def backward_dw(self) -> None:
        """Compute delayed weight gradients for child linears if enabled."""
        self.cola_v_r_to_kv.backward_dw()
        self.cola_k_r_to_kv.backward_dw()
        self.cola_q_r_to_proj.backward_dw()
        self.cola_h_to_3r.backward_dw()


class CoLAAttentionOutputLinear(CoLARowThenColumnLinear):
    """CoLA row -> column(gather) projection for attention output."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool = False,
        tp_comm_buffer_name: str | None = None,
        tp_group: torch.distributed.ProcessGroup | None = None,
        stride: int = 1,
        name: str | None = None,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        del input_is_parallel, tp_comm_buffer_name, stride, name, kwargs
        if is_expert:
            raise NotImplementedError("Initial CoLA support does not cover MoE experts.")

        rank = _cola_rank(config.cola_attn_rank, config.hidden_size, "cola_attn_rank")
        self.cola_in_to_rank = RowParallelLinear(
            input_size,
            rank,
            config=config,
            init_method=init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=True,
            tp_group=tp_group,
        )
        _check_cola_config(config, self.cola_in_to_rank.tp_group)
        self.cola_rank_to_out = ColumnParallelLinear(
            rank,
            output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            gather_output=True,
            skip_bias_add=skip_bias_add,
            tp_group=tp_group,
        )
