# CoLA GPT example

A separate CoLA layer specification, adapted from ATC-Megatron's initial CoLA port. The existing `pretrain_gpt.py` supplies training, data, loss, logging, FSDP, and optimizer handling. No standard GPT/MLP/attention/configuration files are patched.

## Architecture

Weights use `[out, in]` storage. With hidden size `H`, FFN width `F`, attention rank `a`, MLP rank `m`, total query width `Q`, and total key/value width `K`:

| Projection | Weight shapes | Computation |
|---|---|---|
| QKV | `[3a,H]`, `[Q,a]`, `[K,a]`, `[K,a]` | Fused input projection → SiLU → split three bottlenecks → separate Q/K/V expansions |
| Attention output | `[a,Q]`, `[H,a]` | Input projection → SiLU → output projection |
| MLP gate/up | `[2m,H]`, `[F,m]`, `[F,m]` | Fused input projection → SiLU → split two bottlenecks → separate gate/up expansions |
| MLP down | `[m,F]`, `[H,m]` | After the existing outer SwiGLU: input projection → SiLU → output projection |

There are **11 matrix parameters and two RMSNorm vectors per layer**. Embeddings and the vocabulary projection remain standard GPT. All parameters are trainable; there is no frozen dense base or LoRA residual. This is nonlinear bottleneck training, not a dense-equivalent `W=BA` transformation.

QKV outputs are packed **per query group**, matching Megatron's unpacking: `[queries for group 0, K0, V0, queries for group 1, K1, V1, ...]`. This differs from the old wrapper's global `[all Q, all K, all V]` concatenation.

Like the old port, factor projections use Megatron's native `ColumnParallelLinear`/`RowParallelLinear`. Attention and standalone RMSNorm use Transformer Engine. Norms are explicit because the standard TE QKV/FC1 modules fuse norm+linear. Dense-versus-CoLA timing therefore changes architecture, parameter count and norm/linear fusion; it does not isolate granularity. Compare granularity policies within the same CoLA configuration.

## Scope and launch

Initial target: dense, bias-free RMSNorm/SwiGLU GPT training, TP=PP=CP=1, FP32/BF16, including FSDP v1 with optimizer offloading. MHA/GQA/MQA use the same grouped packing. FP8/FP4, MoE, MLA, QK normalization, output gating, delayed weight gradients and sequence parallelism are rejected. Checkpoint save/resume and other untested features are not validated by the smoke run.

From the repository root, replace the ordinary GPT entry with:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  examples/cola/pretrain_cola.py \
  --cola-attn-rank 768 --cola-mlp-rank 768 \
  ...existing bias-free RMSNorm/SwiGLU GPT training arguments...
```

Both ranks default to `hidden_size // 4`; positive overrides are supported. The thin launcher consumes only these two options, supplies the custom `--spec`, and runs the unchanged GPT entry. Alternatively, default ranks work directly with `pretrain_gpt.py --spec examples.cola.cola_model layer_spec`. Use the existing Eagle container and per-rank NUMA binding on Polaris; the abbreviated command above is not a standalone PBS script.

## Correctness checks

`tests/unit_tests/models/test_cola_model.py` compares values and input/parameter gradients with explicit per-branch equations; verifies the actual Megatron QKV unpacking for MHA/GQA/MQA; checks outer SwiGLU, both output projections, matrix/norm/parameter counts, and full GPT backward; rejects invalid ranks and TP settings.

GPU test invocation (four ranks on Polaris):

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 -m pytest \
  --confcutdir=tests/unit_tests/models tests/unit_tests/models/test_cola_model.py -q
```

The targeted invocation uses this file's distributed fixture and avoids unrelated repository-wide dataset downloads. Training smoke evidence and limitations are recorded separately in the project progress documents.
