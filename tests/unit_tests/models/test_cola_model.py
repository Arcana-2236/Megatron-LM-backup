# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Equation, head-layout, and integration checks for the isolated CoLA example."""

import os

import pytest
import torch
import torch.nn.functional as F

from examples.cola.cola_model import (
    CoLAFC1Linear,
    CoLAOutputLinear,
    CoLAQKVLinear,
    get_cola_layer_spec,
)
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture(scope="module", autouse=True)
def distributed():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    owns_group = not torch.distributed.is_initialized()
    if owns_group:
        torch.distributed.init_process_group("nccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    model_parallel_cuda_manual_seed(1234)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = tf32
    parallel_state.destroy_model_parallel()
    if owns_group:
        torch.distributed.destroy_process_group()


def config(groups=4):
    return TransformerConfig(
        num_layers=2,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        num_query_groups=groups,
        normalization="RMSNorm",
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        gradient_accumulation_fusion=False,
        bias_activation_fusion=False,
        attention_dropout=0,
        hidden_dropout=0,
    )


def make_projection(cls, cfg, input_size, output_size, rank=7):
    return cls(
        input_size,
        output_size,
        config=cfg,
        init_method=cfg.init_method,
        bias=False,
        rank=rank,
        tp_group=parallel_state.get_tensor_model_parallel_group(),
    )


def reference_branches(module, x):
    # Independent per-branch equations instead of the implementation's fused input operation.
    branches = []
    rank = module.output_projections[0].weight.shape[1]
    for index, projection in enumerate(module.output_projections):
        first = module.input_projection.weight[index * rank : (index + 1) * rank]
        branches.append(F.linear(F.silu(F.linear(x, first)), projection.weight))
    return branches


def randomize_factors(module):
    # Keep nonlinear outputs/gradients well above absolute comparison tolerances.
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.2)


def compare_values_and_gradients(module, x, actual, expected):
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)
    probe = torch.randn_like(actual)
    variables = (x, *module.parameters())
    actual_grads = torch.autograd.grad(actual, variables, probe, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, variables, probe)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_qkv_equations_and_megatron_unpacking(groups):
    cfg = config(groups)
    spec = get_cola_layer_spec(attn_rank=7, mlp_rank=11)
    attention = build_module(spec.submodules.self_attention, config=cfg, layer_number=1)
    module = attention.linear_qkv
    randomize_factors(module)
    x = torch.randn(3, 2, 64, device="cuda", requires_grad=True)
    query, key, value = reference_branches(module, x)
    q = query.reshape(3, 2, 4, 16)
    k = key.reshape(3, 2, groups, 16)
    v = value.reshape(3, 2, groups, 16)
    heads_per_group = 4 // groups
    expected = torch.cat(
        [
            part
            for group in range(groups)
            for part in (
                q[:, :, group * heads_per_group : (group + 1) * heads_per_group].flatten(-2),
                k[:, :, group],
                v[:, :, group],
            )
        ],
        dim=-1,
    )
    actual, bias = module(x)
    assert bias is None
    compare_values_and_gradients(module, x, actual, expected)
    unpacked = attention.get_query_key_value_tensors(x)
    for actual_head, expected_head in zip(unpacked[:3], (q, k, v)):
        torch.testing.assert_close(actual_head, expected_head, atol=1e-6, rtol=1e-4)


def test_fc1_and_outer_swiglu_equations():
    cfg = config()
    layer = build_module(get_cola_layer_spec(mlp_rank=11), config=cfg, layer_number=1)
    mlp = layer.mlp
    randomize_factors(mlp)
    x = torch.randn(3, 2, 64, device="cuda", requires_grad=True)
    gate, up = reference_branches(mlp.linear_fc1, x)
    intermediate = F.silu(gate) * up
    expected = F.linear(
        F.silu(F.linear(intermediate, mlp.linear_fc2.input_projection.weight)),
        mlp.linear_fc2.output_projection.weight,
    )
    actual, bias = mlp(x)
    assert bias is None
    compare_values_and_gradients(mlp, x, actual, expected)


@pytest.mark.parametrize("input_size", [64, 128])
def test_output_projection_equations(input_size):
    module = make_projection(CoLAOutputLinear, config(), input_size, 64)
    randomize_factors(module)
    x = torch.randn(3, 2, input_size, device="cuda", requires_grad=True)
    expected = F.linear(
        F.silu(F.linear(x, module.input_projection.weight)), module.output_projection.weight
    )
    actual, bias = module(x)
    assert bias is None
    compare_values_and_gradients(module, x, actual, expected)


def test_gpt_parameter_count_norms_and_backward():
    cfg = config()
    model = GPTModel(
        config=cfg,
        transformer_layer_spec=get_cola_layer_spec(attn_rank=8, mlp_rank=16),
        vocab_size=128,
        max_sequence_length=16,
        parallel_output=False,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
    ).cuda()
    for layer in model.decoder.layers:
        assert len([p for p in layer.parameters() if p.ndim == 2]) == 11
        assert len([p for p in layer.parameters() if p.ndim == 1]) == 2
    assert sum(p.numel() for p in model.parameters()) == 43328
    tokens = torch.randint(0, 128, (2, 16), device="cuda")
    positions = torch.arange(16, device="cuda").expand_as(tokens)
    logits = model(tokens, positions, None)
    assert logits.shape == (2, 16, 128)
    logits.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize(
    "cls,output_size", [(CoLAQKVLinear, 192), (CoLAFC1Linear, 256), (CoLAOutputLinear, 64)]
)
def test_rejects_invalid_rank_and_tensor_parallelism(cls, output_size):
    cfg = config()
    with pytest.raises(ValueError, match="positive"):
        make_projection(cls, cfg, 64, output_size, rank=0)
    cfg.tensor_model_parallel_size = 2
    with pytest.raises(ValueError, match="TP=PP=CP=1"):
        make_projection(cls, cfg, 64, output_size)
