# CoLA Optimizer CPU Offload Overlap Optimization

Status: evidence complete for the offload-overlap optimization objective. The tested prototypes did not beat the no-overlap full-offload baseline; the report records why and identifies the deeper runtime change needed next.

## Scope And Problem Statement

This study optimizes CoLA DistOpt + optimizer CPU offload in ATC-Megatron, focusing on why the current `--overlap-cpu-optimizer-d2h-h2d` path hurts runtime and whether transfer coalescing, pinned staging reuse, and lower synchronization can recover throughput.

Primary target: 3B, DP=4, seq1024, micro-batch 1, global batch 4. Nsight profiling uses a 3B-width 4-layer shape when full 3B profiling is too expensive.

## Progress / Run Matrix

| model | impl | runtime | offload | overlap | variant | status | job | log | profile | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3B DP=4 seq1024 mb1 gb4 | CoLA | DistOpt | 0 | 0 | imported no offload baseline | done | 7171700.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_cola_distopt_baseline.log |  |  |
| 3B DP=4 seq1024 mb1 gb4 | CoLA | DistOpt+offload | 1.0 | 0 | imported full offload baseline | done | 7171705.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_cola_distopt_offload_baseline.log |  |  |
| 3B DP=4 seq1024 mb1 gb4 | CoLA | DistOpt+offload | 1.0 | 1 | imported current overlap d2h/h2d | done | 7171735.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_cola_distopt_offload_overlap_d2h_h2d.log |  |  |
| 3B DP=4 seq1024 mb1 gb4 | CoLA | DistOpt+offload | 0.5 | 0 | imported offload fraction 0.5 | done | 7171737.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_cola_distopt_offload_fraction_0p5.log |  |  |
| 3B DP=4 seq1024 mb1 gb4 | CoLA | DistOpt+offload | 1.0 | 1 | imported overlap plus CPU optimizer grouping 50M | done | 7171747.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_cola_distopt_offload_overlap_group50m.log |  |  |
| 3B DP=4 seq1024 mb1 gb4 | FullRank | DistOpt | 0 | 0 | imported control no offload baseline | done | 7171699.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_fullrank_distopt_baseline.log |  |  |
| 3B DP=4 seq1024 mb1 gb4 | FullRank | DistOpt+offload | 1.0 | 0 | imported control full offload baseline | done | 7171701.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_fullrank_distopt_offload_baseline.log |  |  |
| 3B-width 4-layer DP=4 seq1024 mb1 gb4 | CoLA | DistOpt+offload | 1.0 | 0 | Nsight offload no-overlap | done | 7171762.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0526/offload_overlap_optimization_20260526T072656Z/logs/offload_3b4l_cola_distopt_offload_nooverlap_nsys.log | benchmarks/0526/offload_overlap_optimization_20260526T072656Z/profiles/offload_3b4l_cola_distopt_offload_nooverlap_nsys.sqlite |  |
| 3B-width 4-layer DP=4 seq1024 mb1 gb4 | CoLA | DistOpt+offload | 1.0 | 1 | Nsight offload current overlap | done | 7171763.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0526/offload_overlap_optimization_20260526T072656Z/logs/offload_3b4l_cola_distopt_offload_overlap_nsys.log | benchmarks/0526/offload_overlap_optimization_20260526T072656Z/profiles/offload_3b4l_cola_distopt_offload_overlap_nsys.sqlite |  |
| 3B-width 4-layer DP=4 seq1024 mb1 gb4 | CoLA | DistOpt+offload | 1.0 | 1 | Nsight offload overlap plus grouping 50M | done | 7171764.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0526/offload_overlap_optimization_20260526T072656Z/logs/offload_3b4l_cola_distopt_offload_overlap_group50m_nsys.log | benchmarks/0526/offload_overlap_optimization_20260526T072656Z/profiles/offload_3b4l_cola_distopt_offload_overlap_group50m_nsys.sqlite |  |
| 3B-width 4-layer DP=4 seq1024 mb1 gb4 | CoLA | DistOpt+offload | 1.0 | 1 | foreach copy prototype smoke | done | 7171765.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0526/offload_overlap_optimization_20260526T072656Z/logs/offload_3b4l_cola_distopt_offload_overlap_group50m_foreach_smoke.log |  |  |
| 3B DP=4 seq1024 mb1 gb4 | CoLA | DistOpt+offload | 1.0 | 1 | foreach copy prototype full3B | done | 7171766.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0526/offload_overlap_optimization_20260526T072656Z/logs/offload_3b_cola_distopt_offload_overlap_group50m_foreach.log |  |  |
| 3B-width 4-layer DP=4 seq1024 mb1 gb4 | CoLA | DistOpt+offload | 1.0 | 1 | slab copy prototype smoke | done | 7171770.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0526/offload_overlap_optimization_20260526T072656Z/logs/offload_3b4l_cola_distopt_offload_overlap_group50m_slab_smoke.log |  |  |
| 3B-width 4-layer DP=4 seq1024 mb1 gb4 | CoLA | DistOpt+offload | 1.0 | 1 | slab copy prototype Nsight | done | 7171772.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0526/offload_overlap_optimization_20260526T072656Z/logs/offload_3b4l_cola_distopt_offload_overlap_group50m_slab_nsys.log | benchmarks/0526/offload_overlap_optimization_20260526T072656Z/profiles/offload_3b4l_cola_distopt_offload_overlap_group50m_slab_nsys.sqlite |  |
| 3B DP=4 seq1024 mb1 gb4 | CoLA | DistOpt+offload | 1.0 | 1 | slab copy prototype full3B | done | 7171773.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | benchmarks/0526/offload_overlap_optimization_20260526T072656Z/logs/offload_3b_cola_distopt_offload_overlap_group50m_slab.log |  |  |

## FSDP-Only Failure Reconciliation

Measured/code-inspection finding: the current failing 0525 FSDP-only rows used `FSDP_USE_TORCH_OPTIMIZER=0`, so `MEGATRON_FSDP_USE_TORCH_OPTIMIZER` was not exported. Both FullRank and CoLA then failed inside `optimizer.step()` / `optimizer-inner-step` with CUDA illegal memory access and NCCL watchdog aborts. Earlier successful 3B FSDP-only rows from PBS job `7171062` recorded `fsdp_use_torch_optimizer=1`, `CUDA_LAUNCH_BLOCKING=1`, and `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`.

Interpretation: the failure is a Megatron-FSDP-only optimizer-path mismatch, not evidence about CoLA DistOpt optimizer CPU offload. For future stable FSDP-only comparison, run with `FSDP_USE_TORCH_OPTIMIZER=1`; keep `CUDA_LAUNCH_BLOCKING=1` only for diagnosis because it can perturb timing.

Evidence paths:

- Failed current FullRank FSDP: `benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_fullrank_fsdp_baseline.log`, `benchmarks/0525/throughput_optimization_20260526T053247Z/meta/throughput_3b_fullrank_fsdp_baseline.json`
- Failed current CoLA FSDP: `benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_cola_fsdp_baseline.log`, `benchmarks/0525/throughput_optimization_20260526T053247Z/meta/throughput_3b_cola_fsdp_baseline.json`
- Successful prior FullRank/CoLA FSDP: `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171062.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/meta/*_fsdp_offload0_cg0.json`

## Baseline Overlap Timer Breakdown

| variant | iter ms | delta iter | fwd-bwd | grad sync | all-gather | opt total | delta opt | opt inner | max alloc MB | reserved MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no offload | 202.547 | -652.027 | 167.591 | 33.229 | 10.795 | 26.448 | -643.867 | 8.157 | 17427.840 | 18098.000 |
| full offload | 854.573 | 0.000 | 175.232 | 33.328 | 10.848 | 670.315 | 0.000 | 654.067 | 13464.800 | 14160.000 |
| overlap d2h/h2d | 1047.520 | 192.947 | 227.529 | 33.283 | 16.467 | 806.380 | 136.065 | 780.706 | 13464.800 | 14358.000 |
| overlap + optimizer grouping 50000000 | 1014.693 | 160.120 | 231.709 | 33.208 | 18.267 | 768.325 | 98.010 | 736.260 | 13464.800 | 14358.000 |
| offload fraction 0.5 | 561.073 | -293.500 | 181.845 | 33.224 | 13.784 | 369.111 | -301.204 | 347.988 | 15435.720 | 15756.000 |

Imported evidence: enabling current overlap increases CoLA full-offload iteration time from 854.57 ms to 1047.52 ms. The main increase is optimizer total time (+136.07 ms) and forward-backward time (+52.30 ms), with smaller all-gather increase (+5.62 ms). Memory stays effectively flat in max allocated and rises slightly in reserved memory. CPU optimizer grouping recovers about 32.83 ms of iteration time but remains slower than no-overlap full offload.

## Nsight Profile Summary

| run_id | status | kernels | avg kernel us | idle frac | CUDA API us | launch APIs | sync APIs | H2D count | H2D bytes | D2H count | D2H bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offload_3b4l_cola_distopt_offload_nooverlap_nsys | ok | 8763 | 137.119 | 0.492 | 2393716.788 | 8766 | 2925 | 426 | 4817881440 | 366 | 4804956576 |
| offload_3b4l_cola_distopt_offload_overlap_group50m_nsys | ok | 8763 | 234.140 | 0.411 | 3422345.048 | 8766 | 2958 | 426 | 4817881440 | 366 | 4804956576 |
| offload_3b4l_cola_distopt_offload_overlap_group50m_slab_nsys | ok | 8763 | 138.265 | 0.527 | 2053690.147 | 8766 | 2910 | 282 | 4817881440 | 222 | 4804956576 |
| offload_3b4l_cola_distopt_offload_overlap_nsys | ok | 8760 | 142.050 | 0.531 | 2420025.531 | 8763 | 3077 | 421 | 4816804192 | 366 | 4804956576 |

All planned local offload Nsight rows for no-overlap, current overlap, overlap+grouping, and slab transfer coalescing have been parsed.

## Timeline Figures

- `/home/zhengyangwang/offloading/ATC-Megatron/benchmarks/0526/offload_overlap_optimization_20260526T072656Z/figures/nsys_offload_profile_summary.png`
- `/home/zhengyangwang/offloading/ATC-Megatron/benchmarks/0526/offload_overlap_optimization_20260526T072656Z/figures/nsys_offload_profile_summary.pdf`
- `/home/zhengyangwang/offloading/ATC-Megatron/benchmarks/0526/offload_overlap_optimization_20260526T072656Z/figures/offload_overlap_timeline.png`
- `/home/zhengyangwang/offloading/ATC-Megatron/benchmarks/0526/offload_overlap_optimization_20260526T072656Z/figures/offload_overlap_timeline.pdf`

## Theoretical Model

| variant | H2D GB | D2H GB | transfer ms | opt ms | opt-transfer ms | sync API ms/iter | CUDA API ms/iter | idle frac | perfect overlap lb ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no-overlap | 1.204 | 1.201 | 251.383 | 256.671 | 5.288 | 287.631 | 598.429 | 0.492 | 251.383 |
| overlap+group50m | 1.204 | 1.201 | 326.175 | 258.279 | -67.897 | 542.015 | 855.586 | 0.411 | 326.175 |
| overlap+group50m+slab | 1.204 | 1.201 | 260.025 | 272.480 | 12.455 | 291.634 | 513.423 | 0.527 | 260.025 |
| overlap | 1.204 | 1.201 | 323.905 | 271.910 | -51.995 | 324.977 | 605.006 | 0.531 | 323.905 |

Model note: transfer and CUDA API values are normalized by captured iteration count. Negative `opt-transfer` means the rank-local memcpy durations overlap CPU/GPU work or the timer and Nsight windows are not measuring the same critical path exactly; use the table for bottleneck direction, not as a strict additive timing decomposition.

## CPU Offload Code Inspection

ATC-Megatron's `HybridDeviceOptimizer.step()` copies gradients from GPU to pinned CPU buffers on `_d2h_stream`, then synchronizes each CPU optimizer's D2H event immediately before calling that optimizer's `step()`. With overlap enabled, `build_cpu_optimizer_list()` defaults to one CPU optimizer per parameter; each optimizer also installs a post-step hook that copies updated CPU parameters back to GPU on `_h2d_stream` and then waits on the current stream. This creates many optimizer objects, D2H event waits, CPU optimizer calls, H2D copy-back hook invocations, and stream/event operations for fine-grained CoLA parameters.

The grouping prototype is env-gated by `MEGATRON_CPU_OFFLOAD_GROUP_NUMEL` and only groups CPU optimizer instances. The foreach prototype (`MEGATRON_CPU_OFFLOAD_FOREACH_COPY=1`) batches copy calls inside each group but still leaves the same underlying tensor granularity. The slab prototype (`MEGATRON_CPU_OFFLOAD_SLAB_COPY=1`) reuses flat CPU/GPU staging slabs and coalesces grouped D2H/H2D movement; it preserves optimizer math but adds GPU pack/scatter traffic.

## External Source Notes

- ZeRO-Offload paper: https://arxiv.org/abs/2101.06840. Relevance: frames CPU optimizer offload as useful only when GPU data movement is minimized and CPU compute time is reduced; this matches the current need to split transfer, CPU Adam, and framework overhead.
- ZeRO-Infinity paper: https://arxiv.org/abs/2104.07857. Relevance: motivates overlapping heterogeneous memory movement with compute and treating CPU/NVMe bandwidth as a first-class bottleneck.
- DeepSpeed ZeRO documentation: https://deepspeed.readthedocs.io/en/stable/zero3.html. Relevance: documents optimizer/gradient offload lineage, recommends optimized CPUAdam, and identifies pinned memory/effective bandwidth as relevant offload controls.
- DeepSpeed optimizer documentation: https://deepspeed.readthedocs.io/en/latest/optimizers.html. Relevance: contrasts CPUAdam and fused/multi-tensor GPU Adam; supports testing whether many small CPU optimizer calls are a CoLA granularity overhead.
- PyTorch FSDP notes: https://docs.pytorch.org/docs/2.8/notes/fsdp.html. Relevance: explains all-gather prefetch/overlap depends on CPU issue order and compute window length, which is directly relevant to fine-grained CoLA FSDP unit granularity.

## Optimization Attempts And Commands

- Existing env-gated prototype: `MEGATRON_CPU_OFFLOAD_GROUP_NUMEL` groups CPU optimizer instances in `megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py` when overlap is enabled. Command metadata is in `benchmarks/0525/throughput_optimization_20260526T053247Z/meta/throughput_3b_cola_distopt_offload_overlap_group50m.cmd`.
- New env-gated prototype: `MEGATRON_CPU_OFFLOAD_FOREACH_COPY=1` batches D2H grad-copy and H2D param-copy setup with `torch._foreach_copy_` inside each CPU optimizer group. This preserves optimizer math and is intended to reduce per-small-tensor Python/CUDA copy setup, not to coalesce transfers into slabs.
- New env-gated prototype: `MEGATRON_CPU_OFFLOAD_SLAB_COPY=1` packs grouped GPU gradients into a reusable GPU slab, copies that slab to a pinned CPU slab, exposes CPU optimizer gradient views from the slab, then stages updated CPU parameters back through a GPU slab. This is the first prototype that reduces actual host-copy count.

## Before / After Results

| variant | iter ms | delta vs full offload | fwd-bwd | opt total | opt inner | max alloc MB | reserved MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full offload | 854.573 | 0.000 | 175.232 | 670.315 | 654.067 | 13464.800 | 14160.000 |
| overlap d2h/h2d | 1047.520 | 192.947 | 227.529 | 806.380 | 780.706 | 13464.800 | 14358.000 |
| overlap + optimizer grouping 50000000 | 1014.693 | 160.120 | 231.709 | 768.325 | 736.260 | 13464.800 | 14358.000 |
| overlap + group50m + foreach copy | 1004.247 | 149.673 | 215.820 | 768.368 | 737.043 | 13464.800 | 14358.000 |
| overlap + group50m + slab copy | 1141.287 | 286.713 | 225.236 | 895.068 | 861.149 | 15306.240 | 16236.000 |
| offload fraction 0.5 | 561.073 | -293.500 | 181.845 | 369.111 | 347.988 | 15435.720 | 15756.000 |

Measured result: current best full-offload speed remains no-overlap full optimizer CPU offload at 854.57 ms. The CPU optimizer grouping prototype improves overlap-only from 1047.52 ms to 1014.69 ms but does not beat baseline full offload. The foreach-copy prototype validates correctly but gives 1004.25 ms and the same optimizer total as grouping alone, so it is not accepted as a meaningful optimization. The slab-copy prototype does reduce host-copy granularity in Nsight, but full 3B iteration time regresses to 1141.29 ms and global max allocation rises to 16069.49 MB because the staging slabs and GPU pack/scatter work become the new bottleneck.

## Failure / Skipped Rows

No local 0526 rows failed. The slab-copy full 3B row is marked done, not successful, because it produced valid finite-loss timing evidence but regressed throughput and memory.

## Final Diagnosis

Measured diagnosis: current overlap hurts because it does not reduce or coalesce real H2D/D2H movement, while it increases synchronization/API overhead and GPU idle gaps. In the 4-layer Nsight rows, no-overlap and overlap both move about 4.8 GB H2D and 4.8 GB D2H across the capture, with nearly identical copy counts. Overlap increases rank-local CUDA sync API time from 1.15 s to 1.30 s and idle-gap fraction from 0.492 to 0.531; grouping does not change copy counts/bytes and can increase sync/API time. The 3B timer evidence matches this: overlap is slower than full offload, grouping only partly recovers it, and foreach-copy batching is neutral.

The slab prototype confirms the right direction but not a usable implementation: H2D copies fall from 426 to 282 and D2H copies from 366 to 222 versus overlap+group50m in the 4-layer profile, while H2D/D2H byte volume stays constant. However, D2D bytes rise from 0.71 GB to 3.05 GB because the prototype packs and scatters through GPU staging buffers, and the full 3B row slows to 1141.29 ms. Coalescing host copies alone is insufficient when it is implemented by adding equivalent or larger device-side staging work.

Interpretation: CoLA has worse offload-overlap opportunity than FullRank because its compute window is shorter while offload traffic and CPU optimizer work remain large enough to dominate the step. The strongest supporting metrics are optimizer total time, Nsight H2D/D2H copy counts and bytes, CUDA sync API time, idle-gap fraction, and slab-induced D2D byte growth.

## Final Answers

- Why does current overlap hurt? Measured: it keeps the same H2D/D2H bytes and nearly the same copy counts, then adds stream/event waits, CUDA API time, and GPU idle gaps. The 3B row slows from 854.57 ms to 1047.52 ms.
- Which overhead increases? Measured 3B timers show optimizer total grows by 136.07 ms, forward-backward by 52.30 ms, and all-gather by 5.62 ms versus no-overlap full offload. Nsight shows sync/API and idle-gap increases on the reduced profile.
- What is the theoretical best speedup from perfect hiding? For the 3B full-offload row, fwd-bwd + grad sync + all-gather is about 219.41 ms. If optimizer/offload work were perfectly hidden, the coarse upper-bound speedup from 854.57 ms is about 3.89x. The 4-layer Nsight lower bound is less optimistic because rank-local H2D+D2H memcpy duration is about 251 ms/iter, already much larger than its 61 ms compute window.
- Does CoLA give better or worse offload-overlap opportunity than FullRank? Worse. CoLA reduces GPU compute time, leaving less compute window to hide CPU optimizer/offload traffic. The imported FullRank control is slower overall, but its longer compute window gives more room for overlap than CoLA.
- Is offload limited by bandwidth, CPU optimizer computation, synchronization, or framework overhead? Measured evidence points to transfer plus framework synchronization/API overhead, not pure bandwidth alone. No-overlap profile optimizer time is close to rank-local memcpy duration, while current overlap increases sync/API and idle gaps without reducing traffic. CPU Adam work is still present, but the decisive regression is orchestration and staging overhead.
- Is the current slab optimization successful? No. It proves actual host-copy coalescing is possible, but the naive pack/scatter implementation shifts cost to D2D staging and raises memory, so full 3B throughput regresses.
- Which existing knob improves performance? `optimizer_offload_fraction=0.5` is the strongest measured speed-memory Pareto point: 561.07 ms versus 854.57 ms full offload, with max allocated rising from 13464.80 MB to 15435.72 MB. `MEGATRON_CPU_OFFLOAD_GROUP_NUMEL=50000000` improves the overlap variant but remains slower than no-overlap full offload.
- Which new optimization is most promising? A deeper flattened-layout/slab design that avoids separate GPU pack/scatter copies: keep gradients/parameters in persistent contiguous layouts or use fused pack/scatter kernels so host transfers are coalesced without adding multi-GB D2D staging overhead.
- Which result most strongly supports the granularity-aware memory-management thesis? The strongest evidence is the combination of unchanged H2D/D2H traffic under current overlap, increased CUDA sync/API and idle-gap metrics, and the slab prototype reducing host copy count but regressing from added staging overhead. CoLA's finer-grained runtime events prevent memory reduction from becoming throughput speedup.

## Next Recommended Optimization

Prototype a persistent flattened offload layout for CoLA factorized parameters so CPU optimizer state, pinned host gradients, and GPU parameter/gradient views share bucketed storage from initialization. That should preserve the copy-count reduction demonstrated by `MEGATRON_CPU_OFFLOAD_SLAB_COPY=1` while removing the extra D2D pack/scatter step that made the naive slab prototype slower.

## Raw Artifacts

- Imported timer CSV/JSON: `parsed/offload_overlap_timers.csv`, `parsed/offload_overlap_timers.json`
- Baseline comparison CSV/JSON: `parsed/offload_overlap_comparison.csv`, `parsed/offload_overlap_comparison.json`
- Local logs: `logs/*.log`
- Local metadata and commands: `meta/*.json`, `meta/*.cmd`
- Local Nsight profiles: `profiles/*.nsys-rep`, `profiles/*.sqlite`
- Local parsed Nsight: `parsed/nsys_offload_overlap.csv`, `parsed/nsys_offload_overlap.json`
- Theoretical model: `parsed/theoretical_model.csv`, `parsed/theoretical_model.json`
- Figures: `figures/*.png`, `figures/*.pdf`

