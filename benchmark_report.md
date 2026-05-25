# CoLA Memory Benchmark Report

Status: completed on Polaris for the requested 1B and 3B DP=4 matrix. Failed rows are retained in the tables with the failure class observed in the row log.

## Environment

- Repo: `/home/zhengyangwang/offloading/ATC-Megatron`
- Initial HEAD: `4bd8bb33c`
- Reference CoLA repo: `/home/zhengyangwang/offloading/Megatron-DeepSpeed`
- Polaris module snapshot included `conda/2025-09-25`, `cudnn/9.13.0`, `gcc-native/14.2`, and `cray-mpich/9.0.1`.
- Reusable env found: `/home/zhengyangwang/.conda/envs/dspeed_env`
- `dspeed_env` check: PyTorch `2.5.1`, CUDA `12.4`, DeepSpeed `0.18.5+bb250a25` from `/home/zhengyangwang/offloading/DeepSpeed/deepspeed/__init__.py`.

## Code Changes

- Added `--model-impl baseline|cola`.
- Added `--mlp-rank` and `--attn-rank` transformer config fields.
- Added MCore CoLA transformer submodules in `megatron/core/transformer/cola.py`.
- Added CoLA GPT layer spec selection in `gpt_builders.py`.
- Routed local RMSNorm final layer norms through the torch norm wrapper when Apex is installed.
- Extended memory logging to include global max allocated/reserved memory across ranks.
- Added benchmark harness under `benchmarks/cola_memory/`.
- Added FSDP debug metadata and runtime guards to the benchmark harness, including `CUDA_LAUNCH_BLOCKING`, `TORCH_NCCL_ASYNC_ERROR_HANDLING`, FSDP tuning flags, and automatic unsetting of `CUDA_DEVICE_MAX_CONNECTIONS=1` for FSDP rows.
- Made Megatron-FSDP optimizer CPU offload DTensor-aware by copying local shards for CPU/offload operations, avoiding DTensor dispatch for pin-memory and host copies.
- Added an opt-in `MEGATRON_FSDP_USE_TORCH_OPTIMIZER=1` path so Megatron-FSDP can use `torch.optim.AdamW/Adam` instead of the Apex/TE fused optimizer in this environment.

## Methodology

- Runs use one Polaris node with four GPUs unless overridden.
- Data path uses `--mock-data` to isolate model/runtime memory and avoid dataset cache effects.
- Offloading mode in the scaffold means Megatron optimizer CPU offload (`--optimizer-cpu-offload`) with the precision-aware optimizer. In this harness, enabling optimizer CPU offload also enables Megatron distributed optimizer when it is not already active.
- FSDP scaffold uses Megatron-FSDP with `--data-parallel-sharding-strategy optim_grads_params`, the closest mode to ZeRO-3/full parameter, gradient, and optimizer sharding in this repo.
- Megatron distributed optimizer shards optimizer state and separate main parameters/grads across the data-parallel group when those tensors do not overlap with model state, making it ZeRO-2-like for this matrix rather than pure ZeRO-1.

## Commands

Smoke:

```bash
MODEL_SIZE=tiny MODEL_IMPL=baseline STRATEGY=baseline OFFLOAD=0 CUDA_GRAPH=0 TRAIN_ITERS=2 GPUS_PER_NODE=1 bash benchmarks/cola_memory/run_one.sh
MODEL_SIZE=tiny MODEL_IMPL=cola STRATEGY=baseline OFFLOAD=0 CUDA_GRAPH=0 TRAIN_ITERS=2 GPUS_PER_NODE=1 bash benchmarks/cola_memory/run_one.sh
```

Matrix runs on Polaris:

```bash
qsub -v MODEL_SIZES=1b,RUN_TIMEOUT_SECONDS=180 benchmarks/cola_memory/run_polaris.pbs
qsub -v MODEL_SIZES=1b,MODEL_IMPLS=cola,STRATEGIES=distopt,OFFLOADS=1,CUDA_GRAPHS=1,RUN_TIMEOUT_SECONDS=180 benchmarks/cola_memory/run_polaris.pbs
qsub -v MODEL_SIZES=1b,STRATEGIES=fsdp,FSDP_IMPL=torch,RUN_TIMEOUT_SECONDS=180 benchmarks/cola_memory/run_polaris.pbs
qsub -v MODEL_SIZES=1b,MODEL_IMPLS=cola,STRATEGIES=fsdp,RUN_TIMEOUT_SECONDS=120 benchmarks/cola_memory/run_polaris.pbs
qsub -v MODEL_SIZES=3b,STRATEGIES=baseline,RUN_TIMEOUT_SECONDS=180 benchmarks/cola_memory/run_polaris.pbs
qsub -v MODEL_SIZES=3b,STRATEGIES=distopt,RUN_TIMEOUT_SECONDS=180 benchmarks/cola_memory/run_polaris.pbs
qsub -v MODEL_SIZES=3b,STRATEGIES=fsdp,RUN_TIMEOUT_SECONDS=120 benchmarks/cola_memory/run_polaris.pbs
qsub -v MODEL_SIZES=1b,STRATEGIES=fsdp,RUN_TIMEOUT_SECONDS=120,CUDA_LAUNCH_BLOCKING=1,TORCH_NCCL_ASYNC_ERROR_HANDLING=1 benchmarks/cola_memory/run_polaris.pbs
qsub -v MODEL_SIZES=1b,STRATEGIES=fsdp,OFFLOADS=0,CUDA_GRAPHS=0,RUN_TIMEOUT_SECONDS=120,CUDA_LAUNCH_BLOCKING=1,TORCH_NCCL_ASYNC_ERROR_HANDLING=1,FSDP_USE_TORCH_OPTIMIZER=1 benchmarks/cola_memory/run_polaris.pbs
qsub -v MODEL_SIZES=1b,STRATEGIES=fsdp,OFFLOADS=0,CUDA_GRAPHS=1,RUN_TIMEOUT_SECONDS=120,CUDA_LAUNCH_BLOCKING=1,TORCH_NCCL_ASYNC_ERROR_HANDLING=1,FSDP_USE_TORCH_OPTIMIZER=1 benchmarks/cola_memory/run_polaris.pbs
```

Parse results:

```bash
python benchmarks/cola_memory/parse_results.py --results-root <RESULTS_ROOT>
```

## Smoke Verification

- `qsub benchmarks/cola_memory/smoke_polaris.pbs` job `7170030.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov` finished with `Exit_status = 0`.
- Baseline tiny row parsed as `ok`: 1955.65 ms/iter, 0.337 GiB global max allocated, 0.592 GiB global max reserved.
- CoLA tiny row parsed as `ok`: 645.90 ms/iter, 0.184 GiB global max allocated, 0.201 GiB global max reserved.
- Smoke logs and summaries are under `benchmarks/cola_memory/results/smoke_7170030.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/`.

## Caveats

- CoLA attention is currently implemented for tensor-parallel size 1 and standard multi-head attention, matching the requested DP=4 one-node matrix.
- Offloading mode means Megatron optimizer CPU offload (`--optimizer-cpu-offload`) with the precision-aware optimizer. In this harness, optimizer offload requires Megatron distributed optimizer, so rows labeled `DP=4 baseline` with optimizer CPU offload are no longer pure replicated-DP internally.
- Megatron distributed optimizer shards optimizer state and separate main parameters/grads across the DP group when those tensors do not overlap with model state. That is best described as ZeRO-2-like in this matrix; it is not pure ZeRO-1.
- Megatron-FSDP uses `--data-parallel-sharding-strategy optim_grads_params`, corresponding to ZeRO-3-style sharding of parameters, gradients, and optimizer state. After the focused 1B debug pass, non-CUDA-graph 1B Megatron-FSDP rows reach training when either optimizer CPU offload uses the DTensor local-shard copy fix or no-offload uses the `FSDP_USE_TORCH_OPTIMIZER=1` fallback.
- The default no-offload 1B Megatron-FSDP optimizer path still fails in Apex `FusedAdam` with CUDA illegal memory access. `--use-precision-aware-optimizer` was also tested as an alternative but this environment lacks TransformerEngine FusedAdam, so that flag fails before training.
- Megatron-FSDP full-iteration CUDA graph capture remains unsupported in the current setup. With the optimizer/offload fixes, rows log three iterations and then fail during capture with `dependency created on uncaptured work in another stream` in the FSDP pre-forward all-gather wait path.
- The 3B Megatron-FSDP non-CUDA-graph rows were rerun after the focused 1B fixes and now reach training for both Fullrank and CoLA, with and without optimizer CPU offload. The 3B CUDA-graph FSDP rows were not rerun after the fix; based on the focused 1B probes, full-iteration graph capture is still expected to fail in the FSDP pre-forward all-gather wait path.
- Torch FSDP2 was also probed for the 1B FSDP rows. It failed before training because this build reports missing FSDP2 support (`TorchFullyShardedDataParallel requires PyTorch >= 2.4.0 with FSDP 2 support`), so the reported FSDP tables use Megatron-FSDP.
- CUDA-graph rows for non-FSDP generally completed the configured 20 training iterations and logged usable metrics, then hung until `RUN_TIMEOUT_SECONDS` killed teardown. They are marked as `teardown timeout`; their timing and memory values are still the post-warmup training metrics.
- Fullrank 3B baseline without offload OOMed before logging iteration metrics during optimizer-state allocation. CoLA fit in the same no-offload baseline configuration.
- Full training logs and summaries were written under `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/<PBS_JOBID>/`; large core dumps were disabled with `ulimit -c 0`.

## 1B Results

| strategy | offloading mode | cuda graph | Fullrank iteration time | Fullrank peak memory | CoLA iteration time | CoLA peak memory |
|---|---|---:|---:|---:|---:|---:|
| DP=4 baseline | disabled | disabled | 203.3 ms | 26.33 / 26.46 GiB | 186.4 ms | 14.35 / 14.68 GiB |
| DP=4 baseline | disabled | enabled | 177.8 ms (teardown timeout) | 26.33 / 27.31 GiB | 112.5 ms (teardown timeout) | 14.36 / 14.84 GiB |
| DP=4 baseline | optimizer CPU offload | disabled | 780.1 ms | 11.19 / 11.99 GiB | 539.2 ms | 7.49 / 7.78 GiB |
| DP=4 baseline | optimizer CPU offload | enabled | 703.0 ms (teardown timeout) | 11.21 / 11.99 GiB | 439.3 ms (teardown timeout) | 7.52 / 8.03 GiB |
| DP=4 + distributed optimizer | disabled | disabled | 154.7 ms | 14.97 / 15.22 GiB | 163.0 ms | 9.21 / 9.42 GiB |
| DP=4 + distributed optimizer | disabled | enabled | 130.5 ms (teardown timeout) | 14.99 / 15.48 GiB | 90.0 ms (teardown timeout) | 9.23 / 9.57 GiB |
| DP=4 + distributed optimizer | optimizer CPU offload | disabled | 787.6 ms | 11.19 / 11.99 GiB | 551.2 ms | 7.49 / 7.90 GiB |
| DP=4 + distributed optimizer | optimizer CPU offload | enabled | 735.6 ms (teardown timeout) | 11.21 / 11.99 GiB | 443.5 ms (teardown timeout) | 7.52 / 8.09 GiB |
| DP=4 + Megatron-FSDP | disabled | disabled | 336.6 ms (torch optimizer fallback) | 11.77 / 13.35 GiB | 369.5 ms (torch optimizer fallback) | 8.22 / 9.63 GiB |
| DP=4 + Megatron-FSDP | disabled | enabled | failed: graph capture uncaptured work | 11.77 / 13.35 GiB before failure | failed: graph capture uncaptured work | 8.22 / 9.63 GiB before failure |
| DP=4 + Megatron-FSDP | optimizer CPU offload | disabled | 974.1 ms | 9.24 / 12.72 GiB | 734.2 ms | 7.09 / 9.11 GiB |
| DP=4 + Megatron-FSDP | optimizer CPU offload | enabled | failed: graph capture uncaptured work | 9.24 / 12.73 GiB before failure | failed: graph capture uncaptured work | 7.09 / 9.11 GiB before failure |

## 1B Iteration-Time Breakdown

This focused 1B DP=4 rerun used `--timing-log-level 1 --timing-log-option minmax` for non-CUDA-graph rows. Values are averaged over iterations 6-20 to skip the five warmup iterations. Timer columns report the max value across ranks from Megatron's `(min, max)` timer logs. Memory values are from the last reported memory line in each log. Logs are under `benchmarks/0525/manual_1b_dp4_20260525T184424Z/logs/`.

| Run | Avg iter ms | Fwd+bwd ms | Grad sync ms | Param AG ms | Opt inner ms | Opt copy ms | Optimizer ms | Alloc MB | Max alloc MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FullRank DP4 | 210.6 | 150.0 | 51.2 | - | 30.7 | 7.9 | 53.5 | 23263.5 | 26964.1 |
| FullRank DP4 + distopt | 160.9 | 129.0 | 32.3 | 10.5 | 7.8 | 2.0 | 24.7 | 11647.4 | 15331.5 |
| FullRank DP4 + distopt + CUDA graph | 130.4 | - | - | - | - | - | - | 11663.6 | 15342.2 |
| FullRank DP4 + distopt + offload | 774.3 | 133.3 | 32.4 | 10.6 | 617.9 | 0.0 | 633.2 | 7780.1 | 11463.2 |
| CoLA DP4 | 189.5 | 155.0 | 23.6 | - | 14.7 | 5.2 | 28.4 | 10580.1 | 14693.6 |
| CoLA DP4 + distopt | 167.2 | 147.3 | 15.1 | 5.0 | 3.8 | 1.4 | 13.5 | 5327.6 | 9425.5 |
| CoLA DP4 + distopt + CUDA graph | 90.0 | - | - | - | - | - | - | 5343.8 | 9447.4 |
| CoLA DP4 + distopt + offload | 588.4 | 168.1 | 15.1 | 11.3 | 390.8 | 0.8 | 408.4 | 3577.9 | 7673.0 |

CUDA graph rows do not include timer breakdowns because level-1 timing inserts synchronization/barrier calls that are incompatible with full-iteration CUDA graph capture. The breakdown shows that distributed optimizer reduces both optimizer time and memory substantially, while optimizer CPU offload shifts the bottleneck to the optimizer/data-movement path.

## 3B Results

| strategy | offloading mode | cuda graph | Fullrank iteration time | Fullrank peak memory | CoLA iteration time | CoLA peak memory |
|---|---|---:|---:|---:|---:|---:|
| DP=4 baseline | disabled | disabled | OOM before metrics | n/a | 249.1 ms | 28.79 / 29.30 GiB |
| DP=4 baseline | disabled | enabled | OOM before metrics | n/a | 207.9 ms (teardown timeout) | 28.80 / 30.18 GiB |
| DP=4 baseline | optimizer CPU offload | disabled | 1624.6 ms | 23.71 / 24.38 GiB | 816.3 ms | 13.15 / 13.82 GiB |
| DP=4 baseline | optimizer CPU offload | enabled | 1484.5 ms (teardown timeout) | 23.71 / 26.33 GiB | 823.3 ms (teardown timeout) | 13.17 / 14.27 GiB |
| DP=4 + distributed optimizer | disabled | disabled | 271.8 ms | 31.49 / 33.27 GiB | 194.6 ms | 17.03 / 17.72 GiB |
| DP=4 + distributed optimizer | disabled | enabled | 258.1 ms (teardown timeout) | 31.52 / 34.96 GiB | 158.4 ms (teardown timeout) | 17.04 / 18.14 GiB |
| DP=4 + distributed optimizer | optimizer CPU offload | disabled | 1607.4 ms | 23.71 / 24.38 GiB | 874.9 ms | 13.15 / 13.82 GiB |
| DP=4 + distributed optimizer | optimizer CPU offload | enabled | 1589.1 ms (teardown timeout) | 23.71 / 26.33 GiB | 774.5 ms (teardown timeout) | 13.17 / 14.27 GiB |
| DP=4 + Megatron-FSDP | disabled | disabled | 501.2 ms (torch optimizer fallback) | 23.51 / 28.30 GiB | 438.3 ms (torch optimizer fallback) | 13.47 / 16.04 GiB |
| DP=4 + Megatron-FSDP | disabled | enabled | not rerun after FSDP fix | n/a | not rerun after FSDP fix | n/a |
| DP=4 + Megatron-FSDP | optimizer CPU offload | disabled | 1740.2 ms | 17.55 / 19.72 GiB | 1119.6 ms | 10.87 / 15.26 GiB |
| DP=4 + Megatron-FSDP | optimizer CPU offload | enabled | not rerun after FSDP fix | n/a | not rerun after FSDP fix | n/a |
