# ATC-Megatron CoLA Granularity Diagnosis

Status: complete for the currently available ATC-Megatron evidence. Every conclusion below is tied to timer logs, Nsight SQLite data, memory logs, or an explicitly recorded failed/skipped experiment.

## Scope

- Stack: ATC-Megatron only.
- Main timer shape: 1B and 3B width, DP=4, sequence length 1024, micro batch 1, global batch 4.
- Timer method: non-CUDA-graph rows use `--timing-log-level 1 --timing-log-option minmax`, averaged over iterations 6-20. Timer columns use the Megatron min/max timer max value across ranks.
- Memory method: last reported Megatron memory line in each log.
- Profile shape: 3B-width, 4 layers, DP=4, sequence length 1024, micro batch 1, global batch 4, profiling steps 7-10.
- Prior completed artifact used as input: `benchmarks/0525/cola_timers_nsys_20260525T212028Z/`.
- This diagnosis directory: `benchmarks/0525/granularity_diagnosis_20260525T230618Z/`.

## Progress Matrix

| item | status | evidence / conclusion |
|---|---|---|
| Pull existing timer and profiler artifacts | done | Read `benchmarks/0525/cola_timers_nsys_20260525T212028Z/REPORT.md` and derived CSV/JSON. |
| Load cluster/runtime guidance | done | Read `skills/run-on-slurm/SKILL.md`. |
| 1B timer breakdown: distopt, offload, FSDP, FSDP+offload | done | Parsed rows in `derived/timer_breakdown_all.csv`. |
| 3B timer breakdown: DP/no sharding, distopt, offload, FSDP, FSDP+offload | done/failed where applicable | 3B FullRank no-sharding OOM before metrics; all other requested 3B non-CUDA-graph rows parsed. |
| Nsight: FullRank + distributed optimizer | done | `.nsys-rep` and `.sqlite` from job `7171129`. |
| Nsight: CoLA + distributed optimizer | done | `.nsys-rep` and `.sqlite` from job `7171120`. |
| Nsight: CoLA + distributed optimizer + CUDA graph | failed/profile-unusable | Rank 0 SIGSEGV after CUDA profiler capture start; SQLite has no CUDA event tables. |
| Nsight: CoLA + distributed optimizer + CPU optimizer offload | done | `.nsys-rep` and `.sqlite` from job `7171129`. |
| Nsight: FullRank + distributed optimizer + CPU optimizer offload | done | Optional row completed in job `7171129`. |
| Nsight: CoLA + Megatron-FSDP | skipped, environment | Attempted a new profile submission, but current host cannot connect to PBS server: `qsub: cannot connect to server polaris (errno=15008)`. Timer evidence for FSDP is still included. |
| Detailed Nsight parser extension | done | Updated `benchmarks/cola_memory/summarize_nsys_sqlite.py`; wrote `derived/nsys_summary_detailed.csv` and `.json`. |
| Memory fragmentation table | done | Derived from last memory lines in timer logs. |
| Hook count | unavailable | Current logs do not emit per-hook counts. This is not used as evidence. |
| Optimizer state tensor/buffer size distribution | unavailable | Current logs do not emit per-state tensor distributions. Offload conclusions use optimizer timers plus Nsight memcpy/API data instead. |
| Bucket count/fill ratio | done where available | Distopt logs emit one gradient reduce-scatter bucket with fill ratio 1.000 for both FullRank and CoLA. |

## Hypothesis Checklist

| hypothesis | status | evidence | conclusion |
|---|---|---|---|
| H1: CoLA creates finer-grained kernels/runtime events | ruled in | 4-layer Nsight: CoLA distopt has 8565 kernels vs FullRank distopt 6795, avg kernel 55.78 us vs 96.44 us, idle-gap fraction 0.27 vs 0.13. | CoLA exposes kernel/runtime granularity overhead even when iteration time improves. |
| H1b: CUDA graph helps CoLA more than FullRank | partially supported, profile failed | 1B partial full-iteration graph logs show CoLA distopt 90.01 ms vs non-graph 167.19 ms, FullRank 130.41 ms vs 160.92 ms, but both graph runs end with SIGTERM after training and no level-1 timers. Nsight CoLA graph row segfaulted and produced no CUDA event metrics. | Treat graph benefit as likely but not fully measured by Nsight in this setup. |
| H2: Optimizer/runtime overhead is amplified by memory-management runtimes | ruled in for offload and FSDP, not for plain distopt | 3B distopt CoLA optimizer total is 26.47 ms vs FullRank 55.61 ms. With offload, CoLA optimizer total rises to 642.50 ms; FSDP CoLA rises to 57.82 ms, FSDP+offload to 721.70 ms. | Plain distopt handles CoLA well. Offload and FSDP add large optimizer/runtime overheads. |
| H2b: DDP bucket fragmentation explains CoLA distopt overhead | ruled out for measured distopt rows | Logs show one gradient reduce-scatter bucket and padded fill ratio 1.000 for FullRank and CoLA. | The observed CoLA Nsight granularity is not coming from DDP bucket underfill in these runs. |
| H3: CPU optimizer offload shifts bottleneck to transfer/API/optimizer paths | ruled in | 3B CoLA distopt optimizer-inner 8.16 ms -> 626.26 ms with offload. Nsight CoLA offload moves 10.33 GB, with 861 memcpy events and 2422.83 ms CUDA API time. | Offload saves memory but is dominated by optimizer, synchronization, and transfer overhead. |
| H4: FSDP reduces memory but adds communication/orchestration overhead | ruled in | 3B CoLA FSDP max allocated 13.77 GB vs CoLA distopt 17.43 GB, but iteration time 438.29 ms vs 203.89 ms. FSDP forward-backward is 371.08 ms vs distopt 168.97 ms. | FSDP buys additional memory reduction at a large runtime cost. |
| H5: Memory fragmentation/extra framework memory dominates CoLA overhead | partially ruled out | 3B distopt reserved-allocated gap is similar for FullRank and CoLA: 6249.63 MB vs 6166.56 MB. CoLA max-reserved minus max-allocated is lower: 662.03 MB vs 1394.40 MB. | Fragmentation/caching is not the main CoLA distopt bottleneck. It is more significant for FSDP/offload staging and framework buffers. |

## Commands Used

Timer and Nsight commands inherited from the completed measurement run:

```bash
qsub -v MODEL_SIZES=3b,STRATEGIES=baseline,OFFLOADS=0,CUDA_GRAPHS=0,RUN_TIMEOUT_SECONDS=300,TIMING_LOG_LEVEL=1,TIMING_LOG_OPTION=minmax benchmarks/cola_memory/run_polaris.pbs
qsub -v MODEL_SIZES=3b,STRATEGIES=distopt,CUDA_GRAPHS=0,RUN_TIMEOUT_SECONDS=300,TIMING_LOG_LEVEL=1,TIMING_LOG_OPTION=minmax benchmarks/cola_memory/run_polaris.pbs
qsub -v MODEL_SIZES=3b,MODEL_IMPLS=cola,STRATEGIES=distopt,OFFLOADS=0,CUDA_GRAPHS=0,NUM_LAYERS=4,TRAIN_ITERS=12,RUN_TIMEOUT_SECONDS=900,TIMING_LOG_LEVEL=0,PROFILE_NSYS=1,PROFILE_STEP_START=7,PROFILE_STEP_END=10,PROFILE_RANKS=0 benchmarks/cola_memory/run_polaris.pbs
qsub benchmarks/cola_memory/run_nsys_profiles_polaris.pbs
```

Detailed Nsight re-summary for this diagnosis:

```bash
/home/zhengyangwang/.conda/envs/dspeed_env/bin/python benchmarks/cola_memory/summarize_nsys_sqlite.py \
  /eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171120.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/20260525T213004Z_3b_cola_distopt_offload0_cg0.sqlite \
  /eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/nsys_baseline_distopt.sqlite \
  /eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/nsys_cola_distopt_cudagraph.sqlite \
  /eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/nsys_cola_distopt_offload.sqlite \
  /eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/nsys_baseline_distopt_offload.sqlite \
  --out-csv benchmarks/0525/granularity_diagnosis_20260525T230618Z/derived/nsys_summary_detailed.csv \
  --out-json benchmarks/0525/granularity_diagnosis_20260525T230618Z/derived/nsys_summary_detailed.json
```

Skipped CoLA+FSDP profile attempt:

```bash
qsub -v MODEL_SIZES=3b,MODEL_IMPLS=cola,STRATEGIES=fsdp,OFFLOADS=0,CUDA_GRAPHS=0,NUM_LAYERS=4,TRAIN_ITERS=12,RUN_TIMEOUT_SECONDS=900,TIMING_LOG_LEVEL=0,PROFILE_NSYS=1,PROFILE_STEP_START=7,PROFILE_STEP_END=10,PROFILE_RANKS=0,FSDP_USE_TORCH_OPTIMIZER=1 benchmarks/cola_memory/run_polaris.pbs
```

Result: `Unknown Host. qsub: cannot connect to server polaris (errno=15008)`.

## Derived Artifacts

- Timer summary CSV: `benchmarks/0525/granularity_diagnosis_20260525T230618Z/derived/timer_breakdown_all.csv`
- Timer summary JSON: `benchmarks/0525/granularity_diagnosis_20260525T230618Z/derived/timer_breakdown_all.json`
- Detailed Nsight summary CSV: `benchmarks/0525/granularity_diagnosis_20260525T230618Z/derived/nsys_summary_detailed.csv`
- Detailed Nsight summary JSON: `benchmarks/0525/granularity_diagnosis_20260525T230618Z/derived/nsys_summary_detailed.json`

## Timer And Memory Summary

Values are milliseconds and MB. `gap` is reserved minus allocated. `max gap` is max reserved minus max allocated.

| model | row | iter | fwd-bwd | grad sync | all-gather | opt inner | opt total | alloc | max alloc | reserved | gap | max gap | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1B | FullRank distopt | 160.92 | 128.97 | 32.34 | 10.53 | 7.76 | 24.69 | 11647.39 | 15331.48 | 15574.00 | 3926.61 | 242.52 | ok |
| 1B | CoLA distopt | 167.19 | 147.28 | 15.09 | 5.02 | 3.82 | 13.51 | 5327.55 | 9425.50 | 9518.00 | 4190.45 | 92.50 | ok |
| 1B | FullRank FSDP | 336.56 | 278.69 | 4.92 | 11.24 | 28.49 | 50.32 | 7088.94 | 12050.76 | 13668.00 | 6579.06 | 1617.24 | ok |
| 1B | CoLA FSDP | 369.53 | 315.64 | 8.47 | 18.15 | 17.63 | 45.51 | 3924.67 | 8420.22 | 9824.00 | 5899.33 | 1403.78 | ok |
| 1B | FullRank distopt + offload | 774.33 | 133.28 | 32.39 | 10.58 | 617.91 | 633.16 | 7780.11 | 11463.20 | 12280.00 | 4499.89 | 816.80 | ok |
| 1B | CoLA distopt + offload | 588.39 | 168.14 | 15.13 | 11.29 | 390.82 | 408.40 | 3577.91 | 7673.01 | 7970.00 | 4392.09 | 296.99 | ok |
| 1B | FullRank FSDP + offload | 974.11 | 296.58 | 4.78 | 16.25 | 638.75 | 667.91 | 4515.06 | 9461.76 | 13030.00 | 8514.94 | 3568.24 | ok |
| 1B | CoLA FSDP + offload | 734.23 | 326.67 | 8.18 | 55.11 | 315.73 | 392.36 | 2720.75 | 7256.88 | 9324.00 | 6603.25 | 2067.12 | ok |
| 3B | FullRank DP/no sharding | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | failed OOM |
| 3B | CoLA DP/no sharding | 253.35 | 187.49 | 52.37 | n/a | 32.10 | 57.74 | 23973.67 | 29485.75 | 30000.00 | 6026.33 | 514.25 | ok |
| 3B | FullRank distopt | 278.49 | 210.97 | 76.19 | 24.39 | 18.09 | 55.61 | 27384.37 | 32239.60 | 33634.00 | 6249.63 | 1394.40 | ok |
| 3B | CoLA distopt | 203.89 | 168.97 | 33.23 | 10.79 | 8.16 | 26.47 | 11923.44 | 17427.97 | 18090.00 | 6166.56 | 662.03 | ok |
| 3B | FullRank FSDP | 501.23 | 403.83 | 4.81 | 11.52 | 58.47 | 87.50 | 15315.99 | 24078.10 | 28500.00 | 13184.01 | 4421.90 | ok |
| 3B | CoLA FSDP | 438.29 | 371.08 | 8.33 | 14.96 | 30.28 | 57.82 | 7448.97 | 13765.69 | 16420.00 | 8971.03 | 2654.31 | ok |
| 3B | FullRank distopt + offload | 1770.09 | 212.10 | 76.33 | 24.32 | 1512.26 | 1545.53 | 18270.47 | 24276.99 | 24756.00 | 6485.53 | 479.01 | ok |
| 3B | CoLA distopt + offload | 828.89 | 177.50 | 33.29 | 10.81 | 626.26 | 642.50 | 7970.08 | 13463.05 | 14150.00 | 6179.92 | 686.95 | ok |
| 3B | FullRank FSDP + offload | 1740.20 | 412.68 | 4.50 | 13.78 | 1283.26 | 1315.70 | 9245.17 | 17972.44 | 20178.00 | 10932.83 | 2205.56 | ok |
| 3B | CoLA FSDP + offload | 1119.62 | 385.67 | 7.79 | 29.84 | 667.76 | 721.70 | 4807.32 | 11130.67 | 15624.00 | 10816.68 | 4493.33 | ok |

## Nsight Profile Artifacts

| row | status | `.nsys-rep` | `.sqlite` | raw log |
|---|---|---|---|---|
| FullRank distopt | ok | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/nsys_baseline_distopt.nsys-rep` | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/nsys_baseline_distopt.sqlite` | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/nsys_baseline_distopt.log` |
| CoLA distopt | ok | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171120.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/20260525T213004Z_3b_cola_distopt_offload0_cg0.nsys-rep` | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171120.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/20260525T213004Z_3b_cola_distopt_offload0_cg0.sqlite` | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171120.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T213004Z_3b_cola_distopt_offload0_cg0.log` |
| CoLA distopt + CUDA graph | failed/profile-unusable | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/nsys_cola_distopt_cudagraph.nsys-rep` | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/nsys_cola_distopt_cudagraph.sqlite` | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/nsys_cola_distopt_cudagraph.log` |
| CoLA distopt + offload | ok | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/nsys_cola_distopt_offload.nsys-rep` | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/nsys_cola_distopt_offload.sqlite` | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/nsys_cola_distopt_offload.log` |
| FullRank distopt + offload | ok | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/nsys_baseline_distopt_offload.nsys-rep` | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/profiles/nsys_baseline_distopt_offload.sqlite` | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/nsys_baseline_distopt_offload.log` |

## Nsight Summary

The profile was captured on rank 0 only. Idle fraction is approximate: positive gaps between globally sorted kernel intervals divided by the kernel span. Memcpy GB is decimal GB from `CUPTI_ACTIVITY_KIND_MEMCPY`.

| row | kernels | kernel total ms | avg kernel us | p50/p90/p99 kernel us | idle frac | CUDA API total ms | launch APIs / total ms | sync APIs / total ms | memcpy count | memcpy GB | avg memcpy MB | p50/p90/p99 memcpy MB | H2D GB | D2H GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CoLA distopt | 8565 | 477.74 | 55.78 | 13.66/84.71/773.66 | 0.27 | 331.41 | 8568 / 63.66 | 2390 / 249.26 | 513 | 0.72 | 1.41 | 0.00/6.55/6.55 | 0.01 | 0.00 |
| FullRank distopt | 6795 | 655.29 | 96.44 | 12.06/211.08/976.39 | 0.13 | 449.24 | 6798 / 48.24 | 4079 / 387.61 | 513 | 0.72 | 1.41 | 0.00/6.55/6.55 | 0.01 | 0.00 |
| CoLA distopt + CUDA graph | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| CoLA distopt + offload | 8499 | 1215.97 | 143.07 | 13.66/83.94/804.97 | 0.48 | 2422.83 | 8502 / 101.76 | 2094 / 1297.76 | 861 | 10.33 | 12.00 | 0.01/27.65/400.41 | 4.82 | 4.80 |
| FullRank distopt + offload | 6675 | 1298.18 | 194.48 | 12.03/176.54/978.88 | 0.63 | 2877.43 | 6678 / 59.60 | 3953 / 1519.85 | 693 | 17.53 | 25.30 | 0.01/110.59/409.60 | 8.42 | 8.41 |

Approximate memcpy bandwidth from the Nsight memcpy duration totals:

| row | overall GB/s | H2D GB/s | D2H GB/s |
|---|---:|---:|---:|
| CoLA distopt + offload | 10.33 | 6.41 | 19.44 |
| FullRank distopt + offload | 9.74 | 6.61 | 16.00 |

## Bucket Evidence

| row | bucket count | bucket elements | padded elements | fill ratio | evidence log |
|---|---:|---:|---:|---:|---|
| 3B FullRank distopt | 1 | 3178652800 | 3178652800 | 1.000 | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171108.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212644Z_3b_baseline_distopt_offload0_cg0.log` |
| 3B CoLA distopt | 1 | 1378460800 | 1378460800 | 1.000 | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171108.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212818Z_3b_cola_distopt_offload0_cg0.log` |
| 3B FullRank distopt + offload | 1 | 3178652800 | 3178652800 | 1.000 | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171108.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212711Z_3b_baseline_distopt_offload1_cg0.log` |
| 3B CoLA distopt + offload | 1 | 1378460800 | 1378460800 | 1.000 | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171108.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212840Z_3b_cola_distopt_offload1_cg0.log` |
| 4-layer FullRank distopt | 1 | 700444800 | 700444800 | 1.000 | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/nsys_baseline_distopt.log` |
| 4-layer CoLA distopt | 1 | 400412800 | 400412800 | 1.000 | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171120.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T213004Z_3b_cola_distopt_offload0_cg0.log` |

## Raw Timer Log Index

| row | raw log |
|---|---|
| 1B FullRank distopt | `/home/zhengyangwang/offloading/ATC-Megatron/benchmarks/0525/manual_1b_dp4_20260525T184424Z/logs/baseline_1b_dp4_distopt.log` |
| 1B CoLA distopt | `/home/zhengyangwang/offloading/ATC-Megatron/benchmarks/0525/manual_1b_dp4_20260525T184424Z/logs/cola_1b_dp4_distopt.log` |
| 1B FullRank distopt + CUDA graph partial | `/home/zhengyangwang/offloading/ATC-Megatron/benchmarks/0525/manual_1b_dp4_20260525T184424Z/logs/baseline_1b_dp4_distopt_cudagraph.log` |
| 1B CoLA distopt + CUDA graph partial | `/home/zhengyangwang/offloading/ATC-Megatron/benchmarks/0525/manual_1b_dp4_20260525T184424Z/logs/cola_1b_dp4_distopt_cudagraph.log` |
| 1B FullRank distopt + offload | `/home/zhengyangwang/offloading/ATC-Megatron/benchmarks/0525/manual_1b_dp4_20260525T184424Z/logs/baseline_1b_dp4_distopt_offload.log` |
| 1B CoLA distopt + offload | `/home/zhengyangwang/offloading/ATC-Megatron/benchmarks/0525/manual_1b_dp4_20260525T184424Z/logs/cola_1b_dp4_distopt_offload.log` |
| 1B FullRank FSDP | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7170981.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T194656Z_1b_baseline_fsdp_offload0_cg0.log` |
| 1B CoLA FSDP | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7170981.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T194743Z_1b_cola_fsdp_offload0_cg0.log` |
| 1B FullRank FSDP + offload | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7170966.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T193837Z_1b_baseline_fsdp_offload1_cg0.log` |
| 1B CoLA FSDP + offload | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7170966.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T194120Z_1b_cola_fsdp_offload1_cg0.log` |
| 3B FullRank DP/no sharding OOM | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171107.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212457Z_3b_baseline_baseline_offload0_cg0.log` |
| 3B CoLA DP/no sharding | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171107.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212603Z_3b_cola_baseline_offload0_cg0.log` |
| 3B FullRank distopt | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171108.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212644Z_3b_baseline_distopt_offload0_cg0.log` |
| 3B CoLA distopt | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171108.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212818Z_3b_cola_distopt_offload0_cg0.log` |
| 3B FullRank distopt + offload | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171108.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212711Z_3b_baseline_distopt_offload1_cg0.log` |
| 3B CoLA distopt + offload | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171108.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212840Z_3b_cola_distopt_offload1_cg0.log` |
| 3B FullRank FSDP | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171062.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T204903Z_3b_baseline_fsdp_offload0_cg0.log` |
| 3B CoLA FSDP | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171062.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T205059Z_3b_cola_fsdp_offload0_cg0.log` |
| 3B FullRank FSDP + offload | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171062.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T204953Z_3b_baseline_fsdp_offload1_cg0.log` |
| 3B CoLA FSDP + offload | `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171062.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T205126Z_3b_cola_fsdp_offload1_cg0.log` |

## Failed, Skipped, Unsupported

- 3B FullRank DP/no sharding: failed before metrics with CUDA OOM. Evidence log: `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171107.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/20260525T212457Z_3b_baseline_baseline_offload0_cg0.log`.
- CUDA graph level-1 timer breakdown: unsupported in this harness because timer synchronization/barrier calls are incompatible with full-iteration CUDA graph capture.
- 1B FullRank/CoLA distopt + CUDA graph: partial iteration-time evidence only. Both logs reach iteration 20, but torch distributed reports SIGTERM after training, so they are not treated as clean completed timer rows.
- CoLA distopt + CUDA graph under Nsight: failed/profile-unusable. Evidence log: `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171129.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/logs/nsys_cola_distopt_cudagraph.log`; rank 0 receives SIGSEGV after profiler capture start, and the exported SQLite lacks CUDA event tables.
- FSDP + CUDA graph: unsupported/skipped based on prior FSDP full-iteration graph capture failures in the pre-forward all-gather wait path.
- CoLA + Megatron-FSDP Nsight: skipped for this diagnosis because `qsub` is not reachable from the current host. This is an environment skip, not a runtime failure.
- Hook counts, per-hook overhead, and optimizer-state tensor size distributions: unavailable from current logs. They remain instrumentation work, not measured conclusions.

## Diagnosis

CoLA's overhead is a combination, but the dominant component depends on runtime mode.

For Megatron distributed optimizer without offload, CoLA converts memory reduction into speed and does not show a memory-fragmentation bottleneck. At 3B, CoLA distopt is faster than FullRank distopt, 203.89 ms vs 278.49 ms, while reducing max allocated memory from 32239.60 MB to 17427.97 MB. The reserved-allocated gap is nearly the same, 6166.56 MB for CoLA vs 6249.63 MB for FullRank, so allocator fragmentation/caching is not the main reason CoLA behaves differently in this mode.

The cost that remains in plain distopt is kernel/runtime granularity. In the 4-layer profile, CoLA has 26 percent more kernels than FullRank, 8565 vs 6795, and much smaller average kernels, 55.78 us vs 96.44 us. The approximate GPU idle-gap fraction doubles from 0.13 to 0.27. This supports the hypothesis that factorized/fine-grained computation exposes more launch and scheduling gaps, even when total iteration time is better.

CPU optimizer offload is a separate, much larger bottleneck. At 3B, CoLA distopt optimizer-inner time jumps from 8.16 ms to 626.26 ms with offload, and total optimizer time jumps from 26.47 ms to 642.50 ms. Nsight shows why: CoLA offload moves 10.33 GB across 861 memcpy events and spends 2422.83 ms in CUDA runtime APIs, including 1297.76 ms in synchronization-like APIs. FullRank offload moves more total bytes, 17.53 GB, but CoLA still pays high fixed overhead from many runtime and transfer events.

FSDP reduces memory but currently does not convert CoLA's smaller logical state into speed. At 3B, CoLA FSDP max allocated memory is 13765.69 MB, lower than CoLA distopt at 17427.97 MB, but iteration time is 438.29 ms vs 203.89 ms. The FSDP increase is mainly in forward-backward and orchestration: CoLA forward-backward rises from 168.97 ms in distopt to 371.08 ms in FSDP. Offloaded FSDP compounds this with optimizer/data movement: CoLA FSDP+offload total optimizer time is 721.70 ms.

Memory fragmentation and extra framework memory are present but are not the primary CoLA distopt bottleneck. The strongest fragmentation/staging signals are in FSDP and offload modes: 3B CoLA FSDP has an 8971.03 MB reserved-allocated gap, and CoLA FSDP+offload has a 10816.68 MB gap plus a 4493.33 MB max-reserved minus max-allocated gap. Those gaps matter for headroom, but the measured slowdowns line up more directly with forward-backward/FSDP overhead and optimizer/offload timers.

## Answers

1. CoLA's overhead is a combination. In plain Megatron distributed optimizer, the main residual overhead is kernel/runtime granularity, not fragmentation. With CPU offload, the dominant overhead is optimizer plus H2D/D2H transfer and CUDA synchronization overhead. With FSDP, the dominant overhead is forward-backward communication/orchestration plus additional optimizer overhead, especially when offload is enabled.

2. The best current memory-management mode is Megatron distributed optimizer without CPU offload. At 3B it gives both the best measured speed and a large CoLA memory reduction: 203.89 ms iteration time and 17427.97 MB max allocated for CoLA, compared with 278.49 ms and 32239.60 MB for FullRank.

3. The largest optimization opportunity is CPU offload, followed by FSDP. Offload adds hundreds to thousands of milliseconds to optimizer/runtime paths, and Nsight shows many memcpy/API/sync events. FSDP also has large headroom because it reduces memory but roughly doubles the CoLA distopt iteration time at 3B.

4. Runtime changes to try next:
   - Coalesce optimizer/offload state into larger slabs and batch H2D/D2H transfers.
   - Reuse pinned host staging buffers and avoid per-small-tensor transfer setup.
   - Overlap offload transfers with longer compute windows; avoid synchronization points around every small transfer.
   - Add instrumentation for per-optimizer-state tensor sizes, hook counts, and offload transfer grouping.
   - For FSDP, test coarser FSDP units, larger communication units, fewer metadata/hook operations, and torch-optimizer fallback on/off in isolation.
   - For CoLA granularity, use CUDA graph or scoped graph capture around stable forward/backward regions, but first fix the Nsight `cudaProfilerApi` capture failure for CoLA+graph.
   - Add reduced-size `torch.cuda.memory_snapshot()` probes for FSDP/offload to identify reserved-memory gaps and staging-buffer lifetimes.

5. Evidence supporting the conclusions:
   - Timer logs show CoLA distopt improves 3B iteration time and memory while reducing grad-sync, all-gather, and optimizer timers.
   - Nsight shows CoLA distopt uses more/smaller kernels and has higher idle-gap fraction.
   - Timer logs show offload shifts time into optimizer-inner and optimizer-total timers.
   - Nsight offload profiles show large H2D/D2H transfer volumes, many memcpy events, and high CUDA API/sync time.
   - FSDP timer logs show additional memory savings but much higher forward-backward and total iteration time.
   - Memory tables show plain distopt CoLA does not have a larger reserved-allocated gap than FullRank, ruling out fragmentation as the primary bottleneck in that mode.

