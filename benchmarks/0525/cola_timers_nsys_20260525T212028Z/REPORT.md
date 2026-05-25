# ATC-Megatron CoLA Timer and Nsight Analysis

Status: completed for requested timer rows and Nsight profiling rows. Failed/skipped/unsupported rows are retained with concrete evidence.

## Scope

- Repository: `/home/zhengyangwang/offloading/ATC-Megatron`
- Framework: ATC-Megatron only
- Main timer shape: DP=4, sequence length 1024, micro batch 1, global batch 4
- Timer methodology: `--timing-log-level 1 --timing-log-option minmax`, average iterations 6-20, report max timer value across ranks.
- Memory methodology: last reported Megatron memory line in each log.
- CUDA graph timer breakdown: skipped/unsupported for level-1 timers because full-iteration graph capture is incompatible with timer sync/barriers in this harness.

## Run Matrix

| group | row | status | evidence |
|---|---|---|---|
| 1B timer | FullRank DP4 + Megatron-FSDP | done | Parsed fixed FSDP log from job `7170981.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`. |
| 1B timer | CoLA DP4 + Megatron-FSDP | done | Parsed fixed FSDP log from job `7170981.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`. |
| 1B timer | FullRank DP4 + Megatron-FSDP + optimizer CPU offload | done | Parsed fixed FSDP offload log from job `7170966.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`. |
| 1B timer | CoLA DP4 + Megatron-FSDP + optimizer CPU offload | done | Parsed fixed FSDP offload log from job `7170966.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`. |
| 3B timer | FullRank DP4 | failed | Job `7171107.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`, log `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171107.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212457Z_3b_baseline_baseline_offload0_cg0.log`; failed before metrics with traceback/OOM evidence. |
| 3B timer | CoLA DP4 | done | Job `7171107.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`, log `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171107.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212603Z_3b_cola_baseline_offload0_cg0.log`. |
| 3B timer | FullRank DP4 + distributed optimizer | done | Job `7171108.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`, log `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171108.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212644Z_3b_baseline_distopt_offload0_cg0.log`. |
| 3B timer | CoLA DP4 + distributed optimizer | done | Job `7171108.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`, log `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171108.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212818Z_3b_cola_distopt_offload0_cg0.log`. |
| 3B timer | FullRank DP4 + distributed optimizer + optimizer CPU offload | done | Job `7171108.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`, log `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171108.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212711Z_3b_baseline_distopt_offload1_cg0.log`. |
| 3B timer | CoLA DP4 + distributed optimizer + optimizer CPU offload | done | Job `7171108.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`, log `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171108.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212840Z_3b_cola_distopt_offload1_cg0.log`. |
| 3B timer | FullRank DP4 + Megatron-FSDP | done | Parsed fixed 3B FSDP log from job `7171062.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`. |
| 3B timer | CoLA DP4 + Megatron-FSDP | done | Parsed fixed 3B FSDP log from job `7171062.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`. |
| 3B timer | FullRank DP4 + Megatron-FSDP + optimizer CPU offload | done | Parsed fixed 3B FSDP log from job `7171062.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`. |
| 3B timer | CoLA DP4 + Megatron-FSDP + optimizer CPU offload | done | Parsed fixed 3B FSDP log from job `7171062.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`. |
| Nsight | 4-layer 3B-width FullRank + distributed optimizer | done | Sequential profile job `7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`, `.nsys-rep` and `.sqlite` produced. |
| Nsight | 4-layer 3B-width CoLA + distributed optimizer | done | Smoke/profile job `7171120.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`, `.nsys-rep` and `.sqlite` produced. |
| Nsight | 4-layer 3B-width CoLA + distributed optimizer + CUDA graph | failed/profile artifact empty | Sequential profile job `7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`; generated `.nsys-rep`/`.sqlite`, but rank 0 hit SIGSEGV after profiler capture start and SQLite contains no CUDA event tables. |
| Nsight | 4-layer 3B-width CoLA + distributed optimizer + optimizer CPU offload | done | Sequential profile job `7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`, `.nsys-rep` and `.sqlite` produced. |
| Nsight | 4-layer 3B-width FullRank + distributed optimizer + optimizer CPU offload | done | Optional row included in sequential profile job `7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`, `.nsys-rep` and `.sqlite` produced. |

## Artifact Directories

- Report directory: `benchmarks/0525/cola_timers_nsys_20260525T212028Z/`
- Derived CSV/JSON summaries: `benchmarks/0525/cola_timers_nsys_20260525T212028Z/derived/`
- Timer summary CSV: `benchmarks/0525/cola_timers_nsys_20260525T212028Z/derived/timer_breakdown_all.csv`
- Timer summary JSON: `benchmarks/0525/cola_timers_nsys_20260525T212028Z/derived/timer_breakdown_all.json`
- Nsight summary CSV: `benchmarks/0525/cola_timers_nsys_20260525T212028Z/derived/nsys_summary.csv`
- Nsight summary JSON: `benchmarks/0525/cola_timers_nsys_20260525T212028Z/derived/nsys_summary.json`
- Nsight profile roots:
  - `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171120.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/`
  - `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/`

## Commands

Commands used for new reruns and profiles:

```bash
qsub -v MODEL_SIZES=3b,STRATEGIES=baseline,OFFLOADS=0,CUDA_GRAPHS=0,RUN_TIMEOUT_SECONDS=300,TIMING_LOG_LEVEL=1,TIMING_LOG_OPTION=minmax benchmarks/cola_memory/run_polaris.pbs
qsub -v MODEL_SIZES=3b,STRATEGIES=distopt,CUDA_GRAPHS=0,RUN_TIMEOUT_SECONDS=300,TIMING_LOG_LEVEL=1,TIMING_LOG_OPTION=minmax benchmarks/cola_memory/run_polaris.pbs
qsub -v MODEL_SIZES=3b,MODEL_IMPLS=cola,STRATEGIES=distopt,OFFLOADS=0,CUDA_GRAPHS=0,NUM_LAYERS=4,TRAIN_ITERS=12,RUN_TIMEOUT_SECONDS=900,TIMING_LOG_LEVEL=0,PROFILE_NSYS=1,PROFILE_STEP_START=7,PROFILE_STEP_END=10,PROFILE_RANKS=0 benchmarks/cola_memory/run_polaris.pbs
qsub benchmarks/cola_memory/run_nsys_profiles_polaris.pbs
```

## Timer Breakdown Tables

Values are averaged over iterations 6-20 for timer runs. Timer columns are max-across-ranks values from Megatron `(min, max)` timer logs. Memory values are MB from the last memory line in each log.

### 1B Megatron-FSDP

| row | status | iter ms | fwd-bwd | grad sync | param all-gather | opt inner | opt copy | opt total | allocated | max allocated | reserved | log |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FullRank FSDP | ok | 336.56 | 278.69 | 4.92 | 11.24 | 28.49 | 4.93 | 50.32 | 7088.94 | 12050.76 | 13668.00 | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7170981.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T194656Z_1b_baseline_fsdp_offload0_cg0.log` |
| CoLA FSDP | ok | 369.53 | 315.64 | 8.47 | 18.15 | 17.63 | 5.58 | 45.51 | 3924.67 | 8420.22 | 9824.00 | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7170981.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T194743Z_1b_cola_fsdp_offload0_cg0.log` |
| FullRank FSDP + offload | ok | 974.11 | 296.58 | 4.78 | 16.25 | 638.75 | 6.43 | 667.91 | 4515.06 | 9461.76 | 13030.00 | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7170966.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T193837Z_1b_baseline_fsdp_offload1_cg0.log` |
| CoLA FSDP + offload | ok | 734.23 | 326.67 | 8.18 | 55.11 | 315.73 | 16.95 | 392.36 | 2720.75 | 7256.88 | 9324.00 | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7170966.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T194120Z_1b_cola_fsdp_offload1_cg0.log` |

### 3B Timer Rows

| row | status | iter ms | fwd-bwd | grad sync | param all-gather | opt inner | opt copy | opt total | allocated | max allocated | reserved | log |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FullRank DP4 | failed: OOM before metrics | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171107.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212457Z_3b_baseline_baseline_offload0_cg0.log` |
| CoLA DP4 | ok | 253.35 | 187.49 | 52.37 | n/a | 32.10 | 9.44 | 57.74 | 23973.67 | 29485.75 | 30000.00 | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171107.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212603Z_3b_cola_baseline_offload0_cg0.log` |
| FullRank distopt | ok | 278.49 | 210.97 | 76.19 | 24.39 | 18.09 | 4.62 | 55.61 | 27384.37 | 32239.60 | 33634.00 | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171108.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212644Z_3b_baseline_distopt_offload0_cg0.log` |
| CoLA distopt | ok | 203.89 | 168.97 | 33.23 | 10.79 | 8.16 | 2.87 | 26.47 | 11923.44 | 17427.97 | 18090.00 | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171108.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212818Z_3b_cola_distopt_offload0_cg0.log` |
| FullRank distopt + offload | ok | 1770.09 | 212.10 | 76.33 | 24.32 | 1512.26 | 0.32 | 1545.53 | 18270.47 | 24276.99 | 24756.00 | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171108.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212711Z_3b_baseline_distopt_offload1_cg0.log` |
| CoLA distopt + offload | ok | 828.89 | 177.50 | 33.29 | 10.81 | 626.26 | 0.62 | 642.50 | 7970.08 | 13463.05 | 14150.00 | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171108.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212840Z_3b_cola_distopt_offload1_cg0.log` |
| FullRank FSDP | ok | 501.23 | 403.83 | 4.81 | 11.52 | 58.47 | 7.23 | 87.50 | 15315.99 | 24078.10 | 28500.00 | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171062.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T204903Z_3b_baseline_fsdp_offload0_cg0.log` |
| CoLA FSDP | ok | 438.29 | 371.08 | 8.33 | 14.96 | 30.28 | 6.37 | 57.82 | 7448.97 | 13765.69 | 16420.00 | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171062.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T205059Z_3b_cola_fsdp_offload0_cg0.log` |
| FullRank FSDP + offload | ok | 1740.20 | 412.68 | 4.50 | 13.78 | 1283.26 | 7.89 | 1315.70 | 9245.17 | 17972.44 | 20178.00 | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171062.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T204953Z_3b_baseline_fsdp_offload1_cg0.log` |
| CoLA FSDP + offload | ok | 1119.62 | 385.67 | 7.79 | 29.84 | 667.76 | 16.87 | 721.70 | 4807.32 | 11130.67 | 15624.00 | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171062.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T205126Z_3b_cola_fsdp_offload1_cg0.log` |

## Nsight Summary

Nsight Systems profiles use a 4-layer 3B-width model, DP=4, sequence length 1024, micro batch 1, global batch 4. Profiling is requested for training steps 7-10 via `--profile --profile-step-start 7 --profile-step-end 10 --profile-ranks 0` and `nsys profile --capture-range=cudaProfilerApi --export=sqlite`.

### Profile Files

| row | status | `.nsys-rep` | `.sqlite` |
|---|---|---|---|
| FullRank distopt | ok | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/nsys_baseline_distopt.nsys-rep` | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/nsys_baseline_distopt.sqlite` |
| CoLA distopt | ok | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171120.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/20260525T213004Z_3b_cola_distopt_offload0_cg0.nsys-rep` | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171120.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/20260525T213004Z_3b_cola_distopt_offload0_cg0.sqlite` |
| CoLA distopt + CUDA graph | failed / empty SQLite | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/nsys_cola_distopt_cudagraph.nsys-rep` | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/nsys_cola_distopt_cudagraph.sqlite` |
| CoLA distopt + offload | ok | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/nsys_cola_distopt_offload.nsys-rep` | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/nsys_cola_distopt_offload.sqlite` |
| FullRank distopt + offload | ok, optional | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/nsys_baseline_distopt_offload.nsys-rep` | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/nsys_baseline_distopt_offload.sqlite` |

### SQLite-Derived Metrics

Idle gap is approximate: kernels are sorted globally by timestamp and positive gaps between adjacent kernel intervals are summed.

| row | kernels | avg kernel us | p50/p90/p99 kernel us | idle gap fraction | CUDA API calls | avg CUDA API us | total CUDA API ms | memcpy count | avg memcpy bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullRank distopt | 6795 | 96.44 | 12.06 / 211.08 / 976.39 | 0.13 | 17549 | 25.60 | 449.24 | 513 | 1405194 |
| CoLA distopt | 8565 | 55.78 | 13.66 / 84.71 / 773.66 | 0.27 | 20364 | 16.27 | 331.41 | 513 | 1405194 |
| CoLA distopt + CUDA graph | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| CoLA distopt + offload | 8499 | 143.07 | 13.66 / 83.94 / 804.97 | 0.48 | 20550 | 117.90 | 2422.83 | 861 | 11998574 |
| FullRank distopt + offload | 6675 | 194.48 | 12.03 / 176.54 / 978.88 | 0.63 | 17674 | 162.81 | 2877.43 | 693 | 25298037 |

## Failed, Skipped, Unsupported

- 3B FullRank DP4 without sharding/offload failed before metrics with CUDA OOM. Evidence: `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171107.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212457Z_3b_baseline_baseline_offload0_cg0.log`, ranks report `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 212.00 MiB`.
- FSDP full-iteration CUDA graph timer rows are unsupported in this setup. Prior focused FSDP probes failed in the FSDP pre-forward all-gather wait path with `dependency created on uncaptured work in another stream`; therefore this report does not try to collect level-1 FSDP CUDA-graph timer breakdowns.
- CoLA distributed optimizer + CUDA graph under Nsight is not profileable with the current `cudaProfilerApi` capture path. Evidence: `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/nsys_cola_distopt_cudagraph.log`, rank 0 receives SIGSEGV after profiler capture start. The `.nsys-rep` and `.sqlite` files were generated, but the SQLite has no CUDA event tables, so no kernel/API metrics are reported.
- CUDA graph rows intentionally omit level-1 timer breakdowns because timer sync/barrier calls are incompatible with full-iteration graph capture in this harness.

## Interpretation

- Megatron-FSDP adds substantial runtime overhead versus Megatron distributed optimizer in the 3B non-offload rows. FullRank FSDP is 501.23 ms versus 278.49 ms for distributed optimizer; CoLA FSDP is 438.29 ms versus 203.89 ms.
- FSDP does reduce memory. At 3B, FullRank global max allocated drops from 32246.66 MB with distributed optimizer to 24078.10 MB with FSDP. CoLA drops from 17442.14 MB to 13796.82 MB. The tradeoff is a large time increase, mostly in forward-backward plus extra FSDP orchestration.
- Optimizer CPU offload clearly shifts time into optimizer/data-movement paths. For 3B distributed optimizer, FullRank optimizer-inner time jumps from 18.09 ms to 1512.26 ms, and CoLA jumps from 8.16 ms to 626.26 ms. FSDP offload shows the same pattern: FullRank optimizer-inner is 1283.26 ms, CoLA is 667.76 ms.
- CoLA reduces memory and often total iteration time, but exposes more fine-grained runtime overhead. In the 4-layer Nsight profiles, CoLA distopt has more kernels than FullRank distopt (8565 vs 6795), shorter average kernel duration (55.78 us vs 96.44 us), and a higher approximate idle-gap fraction (0.27 vs 0.13).
- CUDA graph likely helps CoLA launch overhead in normal benchmark rows, but the Nsight `cudaProfilerApi` capture path for CoLA + CUDA graph is not usable here: it segfaults after capture start and yields an empty CUDA-event SQLite. Treat the profile as failed, not as evidence against CUDA graph performance.
- Offload and FSDP overheads are good candidates for granularity-aware batching/overlap optimization. The data points to two separate optimization targets: FSDP communication/orchestration around forward-backward and parameter all-gather, and offload optimizer/data movement where optimizer-inner/API overhead dominates.
