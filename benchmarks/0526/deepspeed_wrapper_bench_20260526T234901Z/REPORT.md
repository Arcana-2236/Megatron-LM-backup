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
