# ATC-Megatron Throughput Optimization For CoLA

Status: analysis-ready for the measured study set. All submitted rows are done or explicitly failed with evidence; the next optimization is a recommended follow-up, not an unrecorded conclusion.

## Problem Statement And Scope

This study investigates why CoLA-style factorized Transformer training does not always convert memory and FLOP reductions into proportional throughput speedups in ATC-Megatron. FullRank is the control and CoLA is the compressed/factorized target. The focus is granularity-induced runtime overhead in memory-management systems: Megatron distributed optimizer, optimizer CPU offload, Megatron-FSDP, FSDP+offload, CUDA Graph/scoped graph capture, and related overlap/prefetch mechanisms.

Primary model target: 3B, DP=4, sequence length 1024, micro-batch 1, global batch 4. Expensive profiling may use a 3B-width 4-layer fallback and must be marked as such.

## Questions Or Clarifications

- No user clarification is currently required. If a later unsupported runtime path needs a policy decision, it will be recorded here.

## Run Matrix

| model | impl | runtime | optimization | status | job | log | profiler | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3B DP=4 seq1024 mb1 gb4 | FullRank | DistOpt | baseline | done | 7171699.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_fullrank_distopt_baseline.log |  |  |
| 3B DP=4 seq1024 mb1 gb4 | CoLA | DistOpt | baseline | done | 7171700.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_cola_distopt_baseline.log |  |  |
| 3B DP=4 seq1024 mb1 gb4 | FullRank | DistOpt+offload | baseline | done | 7171701.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_fullrank_distopt_offload_baseline.log |  |  |
| 3B DP=4 seq1024 mb1 gb4 | CoLA | DistOpt+offload | baseline | done | 7171705.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_cola_distopt_offload_baseline.log |  |  |
| 3B DP=4 seq1024 mb1 gb4 | FullRank | FSDP | baseline | failed | 7171708.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_fullrank_fsdp_baseline.log |  | CUDA illegal memory access during optimizer.step()/optimizer-inner-step; NCCL watchdog abort; PBS Exit_status=1 |
| 3B DP=4 seq1024 mb1 gb4 | CoLA | FSDP | baseline | failed | 7171709.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_cola_fsdp_baseline.log |  | CUDA illegal memory access during optimizer.step()/optimizer-inner-step; NCCL watchdog abort; PBS Exit_status=1 |
| 3B DP=4 seq1024 mb1 gb4 | FullRank | FSDP+offload | baseline | done | 7171711.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_fullrank_fsdp_offload_baseline.log |  |  |
| 3B DP=4 seq1024 mb1 gb4 | CoLA | FSDP+offload | baseline | done | 7171713.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_cola_fsdp_offload_baseline.log |  |  |
| 3B-width 4-layer DP=4 seq1024 mb1 gb4 | FullRank | DistOpt | Nsight lightweight | done | 7171720.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b4l_fullrank_distopt_nsys_light.log | benchmarks/0525/throughput_optimization_20260526T053247Z/profiles/throughput_3b4l_fullrank_distopt_nsys_light.sqlite |  |
| 3B-width 4-layer DP=4 seq1024 mb1 gb4 | CoLA | DistOpt | Nsight lightweight | failed | 7171723.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b4l_cola_distopt_nsys_light.log | benchmarks/0525/throughput_optimization_20260526T053247Z/profiles/throughput_3b4l_cola_distopt_nsys_light.sqlite | CUDA device busy or unavailable before training startup; PBS Exit_status=1 |
| 3B-width 4-layer DP=4 seq1024 mb1 gb4 | CoLA | DistOpt | Nsight lightweight retry1 | done | 7171726.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b4l_cola_distopt_nsys_light_retry1.log | benchmarks/0525/throughput_optimization_20260526T053247Z/profiles/throughput_3b4l_cola_distopt_nsys_light_retry1.sqlite |  |
| 3B-width 4-layer DP=4 seq1024 mb1 gb4 | CoLA | DistOpt | CUDA Graph baseline | done | 7171727.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b4l_cola_distopt_cudagraph_baseline.log |  |  |
| 3B DP=4 seq1024 mb1 gb4 | CoLA | DistOpt+offload | overlap_cpu_optimizer_d2h_h2d | done | 7171735.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_cola_distopt_offload_overlap_d2h_h2d.log |  |  |
| 3B DP=4 seq1024 mb1 gb4 | CoLA | DistOpt+offload | optimizer_offload_fraction_0.5 | done | 7171737.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_cola_distopt_offload_fraction_0p5.log |  |  |
| 3B DP=4 seq1024 mb1 gb4 | CoLA | FSDP+offload | fsdp_suggested_comm_unit_2e9 | done | 7171739.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_cola_fsdp_offload_communit_2e9.log |  |  |
| 3B DP=4 seq1024 mb1 gb4 | CoLA | FSDP+offload | fsdp_double_buffer | failed | 7171742.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_cola_fsdp_offload_double_buffer.log |  | CUDA device busy or unavailable before training startup; PBS Exit_status=1 |
| 3B DP=4 seq1024 mb1 gb4 | CoLA | FSDP+offload | fsdp_double_buffer_retry1 | done | 7171743.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_cola_fsdp_offload_double_buffer_retry1.log |  |  |
| 3B DP=4 seq1024 mb1 gb4 | CoLA | DistOpt+offload | prototype_overlap_group50m | done | 7171747.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_cola_distopt_offload_overlap_group50m.log |  |  |


## Baseline Timer Breakdown Tables

| run_id | impl | strategy | offload | cg | status | iter ms | fwd-bwd ms | grad sync ms | all-gather ms | opt inner ms | opt total ms | alloc MB | max alloc MB | reserved MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| throughput_3b4l_cola_distopt_cudagraph_baseline | cola | distopt | 0 | 1 | ok | 40.500 |  |  |  |  |  | 3535.930 | 4689.630 | 5410.000 |
| throughput_3b4l_cola_distopt_nsys_light | cola | distopt | 0 | 0 | failed |  |  |  |  |  |  |  |  |  |
| throughput_3b4l_cola_distopt_nsys_light_retry1 | cola | distopt | 0 | 0 | ok | 1503.257 | 50.860 | 10.183 | 3.397 | 2.440 | 8.814 | 3519.670 | 4671.500 | 4856.000 |
| throughput_3b4l_fullrank_distopt_nsys_light | baseline | distopt | 0 | 0 | ok | 1472.386 | 54.943 | 17.530 | 5.773 | 4.086 | 13.606 | 6095.260 | 7141.380 | 7562.000 |
| throughput_3b_cola_distopt_baseline | cola | distopt | 0 | 0 | ok | 202.547 | 167.591 | 33.229 | 10.795 | 8.157 | 26.448 | 11923.440 | 17427.840 | 18098.000 |
| throughput_3b_cola_distopt_offload_baseline | cola | distopt | 1 | 0 | ok | 854.573 | 175.232 | 33.328 | 10.848 | 654.067 | 670.315 | 7970.080 | 13464.800 | 14160.000 |
| throughput_3b_cola_distopt_offload_fraction_0p5 | cola | distopt | 1 | 0 | ok | 561.073 | 181.845 | 33.224 | 13.784 | 347.988 | 369.111 | 9935.290 | 15435.720 | 15756.000 |
| throughput_3b_cola_distopt_offload_overlap_d2h_h2d | cola | distopt | 1 | 0 | ok | 1047.520 | 227.529 | 33.283 | 16.467 | 780.706 | 806.380 | 7970.080 | 13464.800 | 14358.000 |
| throughput_3b_cola_distopt_offload_overlap_group50m | cola | distopt | 1 | 0 | ok | 1014.693 | 231.709 | 33.208 | 18.267 | 736.260 | 768.325 | 7970.080 | 13464.800 | 14358.000 |
| throughput_3b_cola_fsdp_baseline | cola | fsdp | 0 | 0 | failed |  |  |  |  |  |  |  |  |  |
| throughput_3b_cola_fsdp_offload_baseline | cola | fsdp | 1 | 0 | ok | 892.247 | 221.895 | 8.295 | 35.228 | 602.765 | 656.173 | 4807.320 | 11130.670 | 15822.000 |
| throughput_3b_cola_fsdp_offload_communit_2e9 | cola | fsdp | 1 | 0 | ok | 874.060 | 213.119 | 8.405 | 36.685 | 588.871 | 641.989 | 5747.320 | 12070.670 | 16754.000 |
| throughput_3b_cola_fsdp_offload_double_buffer | cola | fsdp | 1 | 0 | failed |  |  |  |  |  |  |  |  |  |
| throughput_3b_cola_fsdp_offload_double_buffer_retry1 | cola | fsdp | 1 | 0 | ok | 891.700 | 232.290 | 8.176 | 25.115 | 596.073 | 639.979 | 4156.540 | 10376.060 | 11344.000 |
| throughput_3b_fullrank_distopt_baseline | baseline | distopt | 0 | 0 | ok | 277.740 | 210.207 | 76.019 | 24.471 | 18.091 | 55.670 | 27384.370 | 32239.600 | 33650.000 |
| throughput_3b_fullrank_distopt_offload_baseline | baseline | distopt | 1 | 0 | ok | 1701.580 | 211.918 | 76.047 | 24.508 | 1443.735 | 1477.178 | 18270.470 | 24276.990 | 25002.000 |
| throughput_3b_fullrank_fsdp_baseline | baseline | fsdp | 0 | 0 | failed |  |  |  |  |  |  |  |  |  |
| throughput_3b_fullrank_fsdp_offload_baseline | baseline | fsdp | 1 | 0 | ok | 1585.520 | 229.503 | 4.705 | 13.270 | 1316.954 | 1345.717 | 9245.170 | 17972.440 | 27058.000 |


## Nsight / Profiler Summary Tables

| sqlite | kernels | avg kernel us | idle frac | CUDA API ms | launch APIs | sync APIs | memcpy count | memcpy bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| /home/zhengyangwang/offloading/ATC-Megatron/benchmarks/0525/throughput_optimization_20260526T053247Z/profiles/throughput_3b4l_cola_distopt_nsys_light_retry1.sqlite | 8829 | 66.442 | 0.141 | 462523.847 | 8832 | 3168 | 624 | 720867312 |
| /home/zhengyangwang/offloading/ATC-Megatron/benchmarks/0525/throughput_optimization_20260526T053247Z/profiles/throughput_3b4l_fullrank_distopt_nsys_light.sqlite | 7059 | 112.550 | 0.085 | 620655.910 | 7062 | 4813 | 624 | 720867312 |


## Bottleneck Models

### CPU Optimizer Offload

CoLA offload is dominated by optimizer/offload work rather than allocator fragmentation. CoLA DistOpt baseline is 202.55 ms/iter, while CoLA DistOpt+full offload is 854.57 ms/iter with optimizer total 670.31 ms. Reducing offload fraction to 0.5 improves to 561.07 ms/iter and optimizer total 369.11 ms, at the cost of max allocated GPU memory rising from 13464.80 MB to 15435.72 MB.

The supported overlap knob did not help CoLA: `--overlap-cpu-optimizer-d2h-h2d` worsened full offload to 1047.52 ms/iter and optimizer total 806.38 ms. The env-gated prototype `MEGATRON_CPU_OFFLOAD_GROUP_NUMEL=50000000` reduced the overlap penalty slightly to 1014.69 ms/iter and optimizer total 768.32 ms, but remained slower than the full-offload baseline. This rules out "too many CPU optimizer objects" as the only cause; transfer synchronization, CPU optimizer work, and H2D/D2H scheduling remain exposed.

### FSDP

Megatron-FSDP without offload failed for both FullRank and CoLA with CUDA illegal memory access during `optimizer.step()` / `optimizer-inner-step`, so this runtime is recorded as failed for the current 3B configuration.

CoLA FSDP+offload succeeds but remains optimizer/offload dominated: 892.25 ms/iter, optimizer total 656.17 ms, forward-backward 221.89 ms, params all-gather 35.23 ms. Code inspection shows Megatron-FSDP defaults FSDP units to `TransformerLayer`, so CoLA factors are not individually wrapped as FSDP units; the overhead is more consistent with within-layer factor granularity plus FSDP/offload orchestration.

### CoLA Kernel/Runtime Granularity

Nsight 4-layer profiles validate the granularity mechanism: CoLA has 8829 kernels vs 7059 for FullRank, average kernel duration 66.44 us vs 112.55 us, and idle-gap fraction 0.141 vs 0.085. Memcpy count/bytes are identical in these DistOpt profiles, so the difference is not explained by additional memory copies.

CUDA Graph full-iteration capture succeeded for CoLA 3B-width 4-layer DistOpt and produced 40.50 ms/iter after capture. A comparable non-graph CoLA 4-layer run had pre-profile iterations around 52-57 ms before Nsight capture/export overhead, so graph capture recovers a meaningful portion of launch/runtime overhead. Timer sub-breakdowns are unavailable because graph-safe runs disable timing barriers.

## Existing Runtime Knobs Discovered

| area | knob | evidence | result |
|---|---|---|---|
| Optimizer CPU offload | `--optimizer-offload-fraction` | `megatron/training/arguments.py` | 0.5 was best measured offload knob; faster with higher GPU memory |
| Optimizer CPU offload | `--overlap-cpu-optimizer-d2h-h2d` | `megatron/core/optimizer/cpu_offloading/README.md` | worsened CoLA full offload |
| Optimizer CPU offload | `MEGATRON_CPU_OFFLOAD_GROUP_NUMEL` prototype | `megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py` | modestly improved overlap path, still slower than baseline |
| Optimizer CPU offload | `--use-torch-optimizer-for-cpu-offload`, `--no-pin-cpu-grads`, `--no-pin-cpu-params` | `megatron/training/arguments.py` | discovered, not yet isolated |
| Distributed optimizer | `--overlap-grad-reduce`, `--overlap-param-gather`, `--ddp-num-buckets`, `--ddp-bucket-size` | `megatron/training/arguments.py` | discovered, not yet isolated |
| FSDP | `--suggested-communication-unit-size` | `megatron/training/arguments.py` | 2e9 gave small speedup, higher memory |
| FSDP | `--fsdp-double-buffer` | `megatron/training/arguments.py` | neutral iteration time, lower all-gather and lower memory in retry |
| CUDA Graph | `--cuda-graph-impl full_iteration` | `megatron/training/arguments.py` | successful for 4-layer CoLA DistOpt |

## Optimization Attempts And Commands

Commands are captured in `meta/*.cmd`. Key command artifacts:

- `meta/throughput_3b_cola_distopt_offload_overlap_d2h_h2d.cmd`
- `meta/throughput_3b_cola_distopt_offload_fraction_0p5.cmd`
- `meta/throughput_3b_cola_distopt_offload_overlap_group50m.cmd`
- `meta/throughput_3b_cola_fsdp_offload_communit_2e9.cmd`
- `meta/throughput_3b_cola_fsdp_offload_double_buffer_retry1.cmd`
- `meta/throughput_3b4l_cola_distopt_cudagraph_baseline.cmd`

## Before / After Result Tables

| optimization | baseline | optimized | iter delta | optimizer delta | memory impact | conclusion |
|---|---:|---:|---:|---:|---:|---|
| CoLA DistOpt+offload overlap | 854.57 ms | 1047.52 ms | +22.6% slower | +136.07 ms | max alloc unchanged; reserved +198 MB | reject |
| CoLA DistOpt+offload fraction 0.5 | 854.57 ms | 561.07 ms | 34.3% faster | -301.20 ms | max alloc +1970.92 MB | best existing offload knob |
| CoLA DistOpt+offload overlap + grouped CPU optimizers | 1047.52 ms overlap-only | 1014.69 ms | 3.1% faster than overlap-only | -38.06 ms | no material memory change | prototype insufficient |
| CoLA FSDP+offload comm unit 2e9 | 892.25 ms | 874.06 ms | 2.0% faster | -14.18 ms | max alloc +940.00 MB | small speedup |
| CoLA FSDP+offload double buffer | 892.25 ms | 891.70 ms | neutral | -16.19 ms | max alloc -754.61 MB; reserved -4478 MB | useful memory/cache behavior, not throughput |
| CoLA 4-layer DistOpt CUDA Graph | ~52-57 ms pre-profile non-graph | 40.50 ms | faster | unavailable | max alloc 4689.63 MB | recovers launch/runtime overhead |

## Failure / Skipped Configs

| run_id | status | failure | log |
| --- | --- | --- | --- |
| throughput_3b_fullrank_fsdp_baseline | failed | CUDA illegal memory access during optimizer.step()/optimizer-inner-step; NCCL watchdog abort | `logs/throughput_3b_fullrank_fsdp_baseline.log` |
| throughput_3b_cola_fsdp_baseline | failed | CUDA illegal memory access during optimizer.step()/optimizer-inner-step; NCCL watchdog abort | `logs/throughput_3b_cola_fsdp_baseline.log` |
| throughput_3b4l_cola_distopt_nsys_light | failed | CUDA device busy or unavailable before training startup; retried successfully | `logs/throughput_3b4l_cola_distopt_nsys_light.log` |
| throughput_3b_cola_fsdp_offload_double_buffer | failed | CUDA device busy or unavailable before training startup; retried successfully | `logs/throughput_3b_cola_fsdp_offload_double_buffer.log` |
| throughput_3b4l_cola_distopt_cudagraph_baseline | done with caveat | training completed and parsed; PBS job was later terminated after teardown hang | `logs/throughput_3b4l_cola_distopt_cudagraph_baseline.log` |

## Evidence-Backed Conclusions

- Dominant CoLA DistOpt bottleneck: kernel/runtime granularity. Evidence: more kernels, shorter kernels, higher idle-gap fraction for CoLA.
- Dominant CoLA DistOpt+offload bottleneck: optimizer/offload state movement and CPU/framework work. Evidence: offload fraction 0.5 cuts optimizer total by 44.9% while using more GPU memory.
- Does CoLA have better offload-overlap opportunity than FullRank? No measured support. The overlap knob made CoLA slower, and grouping CPU optimizers only partially reduced that penalty.
- Is offload bandwidth-limited? Not purely. If bandwidth alone dominated, overlap and grouping would be expected to help more; the measured results point to CPU optimizer/synchronization/framework overhead.
- FSDP-only status: failed/unsupported in this configuration for both FullRank and CoLA.
- Dominant CoLA FSDP+offload bottleneck: optimizer/offload plus FSDP orchestration. FSDP knobs move subcomponents but barely move total iteration time.
- Is FSDP unit granularity too fine for CoLA? At the FSDP wrapper level, no: units are `TransformerLayer`, not individual factors. At kernel/runtime level, yes: CoLA still creates smaller kernels and many factor parameters inside the layer unit.
- Strongest support for the paper thesis: CoLA reduces logical memory/FLOPs but creates more runtime events; that shows up as higher kernel count/idle gaps and poor offload-overlap conversion.

## Next Recommended Optimization

The next code optimization should coalesce actual D2H/H2D transfers and pinned staging buffers, not just group CPU optimizer objects. The 50M grouping prototype improved overlap slightly but did not recover baseline, so the remaining target is batched transfer/staging and reduced synchronization around each transfer.

For FSDP, do not split CoLA into finer FSDP units. Units are already layer-level. Prefer reducing within-layer hook/metadata scheduling for the many CoLA factor parameters or testing NCCL user-buffer registration as the next supported knob.

## Raw Artifacts

- Logs: `benchmarks/0525/throughput_optimization_20260526T053247Z/logs/*.log`
- Metadata and commands: `benchmarks/0525/throughput_optimization_20260526T053247Z/meta/*.json`, `meta/*.cmd`
- Nsight profiles: `benchmarks/0525/throughput_optimization_20260526T053247Z/profiles/*.nsys-rep`, `profiles/*.sqlite`
- Parsed metrics: `benchmarks/0525/throughput_optimization_20260526T053247Z/parsed/timer_breakdown.csv`, `parsed/nsys_summary.csv`, and matching JSON files
- Progress and matrix: `PROGRESS.md`, `RUN_MATRIX.csv`, `HYPOTHESES.md`

## External Sources

No external sources were used. All evidence came from ATC-Megatron code inspection and measured ATC-Megatron logs/profiles.
