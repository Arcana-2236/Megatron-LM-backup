# ATC DeepSpeed Wrapper Benchmark

Results root:
`/eagle/TensorCompress/zhengyangwang/atc_deepspeed_wrapper/7172605.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`

Parsed artifacts:
- `/eagle/TensorCompress/zhengyangwang/atc_deepspeed_wrapper/7172605.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/parsed/deepspeed_wrapper_table.csv`
- `/eagle/TensorCompress/zhengyangwang/atc_deepspeed_wrapper/7172605.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/parsed/deepspeed_wrapper_table.json`
- `/eagle/TensorCompress/zhengyangwang/atc_deepspeed_wrapper/7172605.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/parsed/deepspeed_wrapper_table.md`

The table below is for the current ATC DeepSpeed integration in model-wrapper mode:
`--deepspeed --deepspeed-wrapper-mode model`. In this mode DeepSpeed wraps the
model, while ATC/Megatron still owns backward and optimizer. Optimizer-offload
rows use ATC/Megatron distributed optimizer CPU offload, not DeepSpeed-owned ZeRO
optimizer offload.

| row | status | iter ms | fwd-bwd | grad sync | param all-gather | opt inner | opt copy | opt total | allocated | max allocated | reserved | log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FullRank DeepSpeed wrapper | failed: [rank2]: Cuda failure 2 'out of memory' | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | /eagle/TensorCompress/zhengyangwang/atc_deepspeed_wrapper/7172605.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260526T235854Z_3b_baseline_deepspeed_wrapper_model_offload0_cg0.log |
| CoLA DeepSpeed wrapper | ok | 254.47 | 188.39 | 52.36 | n/a | 32.13 | 9.46 | 57.91 | 23973.67 | 29485.75 | 30000.00 | /eagle/TensorCompress/zhengyangwang/atc_deepspeed_wrapper/7172605.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260526T235945Z_3b_cola_deepspeed_wrapper_model_offload0_cg0.log |
| FullRank DeepSpeed wrapper + optimizer offload | ok | 1603.75 | 212.35 | 76.29 | 24.41 | 1345.54 | 0.31 | 1378.91 | 18270.47 | 24276.99 | 24756.00 | /eagle/TensorCompress/zhengyangwang/atc_deepspeed_wrapper/7172605.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260527T000012Z_3b_baseline_deepspeed_wrapper_model_offload1_cg0.log |
| CoLA DeepSpeed wrapper + optimizer offload | ok | 848.43 | 175.28 | 33.40 | 10.85 | 648.16 | 0.60 | 664.34 | 7970.08 | 13463.05 | 14150.00 | /eagle/TensorCompress/zhengyangwang/atc_deepspeed_wrapper/7172605.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260527T000118Z_3b_cola_deepspeed_wrapper_model_offload1_cg0.log |
| FullRank DeepSpeed wrapper + CUDA Graph | failed: [rank0]: Cuda failure 2 'out of memory' | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | /eagle/TensorCompress/zhengyangwang/atc_deepspeed_wrapper/7172605.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260527T000203Z_3b_baseline_deepspeed_wrapper_model_offload0_cg1.log |
| CoLA DeepSpeed wrapper + CUDA Graph | teardown timeout | 208.01 | n/a | n/a | n/a | n/a | n/a | n/a | 23989.93 | 29494.34 | 30900.00 | /eagle/TensorCompress/zhengyangwang/atc_deepspeed_wrapper/7172605.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260527T000251Z_3b_cola_deepspeed_wrapper_model_offload0_cg1.log |
| FullRank DeepSpeed wrapper + optimizer offload + CUDA Graph | teardown timeout | 1430.21 | n/a | n/a | n/a | n/a | n/a | n/a | 18286.73 | 24276.99 | 26632.00 | /eagle/TensorCompress/zhengyangwang/atc_deepspeed_wrapper/7172605.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260527T001752Z_3b_baseline_deepspeed_wrapper_model_offload1_cg1.log |
| CoLA DeepSpeed wrapper + optimizer offload + CUDA Graph | teardown timeout | 807.51 | n/a | n/a | n/a | n/a | n/a | n/a | 7986.34 | 13490.75 | 14610.00 | /eagle/TensorCompress/zhengyangwang/atc_deepspeed_wrapper/7172605.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260527T003256Z_3b_cola_deepspeed_wrapper_model_offload1_cg1.log |

## Comparison Notes

- Previous Megatron-DeepSpeed 3B ZeRO-1 DP4 no-offload logs report warm
  iteration means of 317.13 ms for FullRank and 546.20 ms for CoLA in
  `/home/zhengyangwang/offloading/Megatron-DeepSpeed/.logging/0507/`.
  The current ATC model-wrapper CoLA no-offload row is 254.47 ms, so it is
  substantially faster than the previous Megatron-DeepSpeed CoLA path. The
  current FullRank no-offload row OOMs, matching the provided baseline behavior,
  so no FullRank speed comparison is available for that row.
- Against current Megatron distributed-optimizer offload rows supplied in the
  prompt, the model-wrapper offload result is mixed: FullRank is faster
  (1603.75 ms vs 1770.09 ms), while CoLA is slightly slower (848.43 ms vs
  828.89 ms). Because this wrapper mode still uses the ATC/Megatron optimizer,
  this should be treated as run-to-run/runtime-path comparison, not proof that a
  DeepSpeed-owned optimizer is faster.
- CUDA Graph capture works for the model-wrapper path when memory is sufficient:
  the successful graph rows contain `Capture CUDA graph for training!!!`,
  `CUDA graph capture done for training!!!`, and complete through iteration 20.
  The reported failure is a post-training NCCL process-group teardown timeout.
- FullRank no-offload with CUDA Graph fails from CUDA/NCCL OOM during the run.
  That is a memory-capacity blocker, not evidence that DeepSpeedEngine wrapping
  is graph-incompatible.
## Optimizer-Owned DeepSpeed Results

The optimizer-owned path was fixed after the model-wrapper benchmark above.
These rows use `--deepspeed --deepspeed-wrapper-mode optimizer` with ZeRO stage
2 optimizer CPU offload. DeepSpeed owns forward execution, backward via
`engine.backward(loss)`, and optimizer step via `engine.step()`.

3B results root:
`/eagle/TensorCompress/zhengyangwang/atc_deepspeed_wrapper/7172661.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`

| row | status | iter ms | fwd-bwd | grad sync | param all-gather | opt inner | opt copy | opt total | allocated | max allocated | reserved | log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FullRank DeepSpeed optimizer-owned + ZeRO optimizer offload | ok | 1787.31 | 173.91 | 0.02 | n/a | n/a | n/a | 1601.90 | 12421.88 | 18249.11 | 19370.00 | /eagle/TensorCompress/zhengyangwang/atc_deepspeed_wrapper/7172661.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260527T013246Z_3b_baseline_deepspeed_wrapper_optimizer_zero2_offload1_cg0.log |
| CoLA DeepSpeed optimizer-owned + ZeRO optimizer offload | ok | 1035.65 | 252.21 | 0.02 | n/a | n/a | n/a | 753.66 | 5609.60 | 11824.20 | 13148.00 | /eagle/TensorCompress/zhengyangwang/atc_deepspeed_wrapper/7172661.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260527T013405Z_3b_cola_deepspeed_wrapper_optimizer_zero2_offload1_cg0.log |

Tiny 3B-width/4-layer smoke results root:
`/eagle/TensorCompress/zhengyangwang/atc_deepspeed_wrapper/7172660.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`

Both FullRank and CoLA smoke rows completed before the full 3B run was launched.

## Optimizer-Owned DeepSpeed ZeRO-3 Results

The initial ZeRO-3 run with default ATC gradient accumulation fusion failed for
all four rows before iteration metrics. The root cause in each row was
`CUBLAS_STATUS_INVALID_VALUE` from
`fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16` during DeepSpeed-owned
backward.

Default-fusion failure root:
`/eagle/TensorCompress/zhengyangwang/atc_deepspeed_wrapper/7172884.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`

After adding `NO_GRADIENT_ACCUMULATION_FUSION=1`, passing
`--no-gradient-accumulation-fusion`, and avoiding incompatible ATC
`main_grad` copying for ZeRO-3 sharded gradients, all four ZeRO-3 rows completed.

3B ZeRO-3 results root:
`/eagle/TensorCompress/zhengyangwang/atc_deepspeed_wrapper/7172886.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`

| row | status | iter ms | fwd-bwd | grad sync | opt total | allocated | max allocated | reserved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FullRank DeepSpeed optimizer-owned ZeRO-3 | ok | 448.57 | 386.17 | 0.02 | 56.73 | 13163.36 | 20088.47 | 25276.00 |
| CoLA DeepSpeed optimizer-owned ZeRO-3 | ok | 908.51 | 865.47 | 0.02 | 35.15 | 6295.47 | 13752.54 | 15526.00 |
| FullRank DeepSpeed optimizer-owned ZeRO-3 + optimizer offload | ok | 2250.43 | 741.12 | 0.02 | 1502.52 | 2551.98 | 9720.56 | 11714.00 |
| CoLA DeepSpeed optimizer-owned ZeRO-3 + optimizer offload | ok | 1691.98 | 1034.51 | 0.02 | 643.18 | 1693.98 | 9152.72 | 10480.00 |

Notes:

- ZeRO-3 no-offload is faster for FullRank than CoLA in this wrapper path:
  CoLA is 2.03x slower, although it uses 31.5% less peak memory.
- ZeRO-3 optimizer offload flips the throughput relationship: CoLA is 24.8%
  faster than FullRank and uses 5.8% less peak memory, but both are much slower
  than no-offload.
- Compared with ZeRO-2 optimizer offload, ZeRO-3 optimizer offload uses less
  memory but is slower: FullRank is 25.9% slower with 46.7% lower peak memory;
  CoLA is 63.4% slower with 22.6% lower peak memory.
