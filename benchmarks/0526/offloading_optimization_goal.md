Optimize CoLA optimizer CPU offload in ATC-Megatron, focusing on overlap, transfer coalescing, staging-buffer reuse, and reducing synchronization overhead.

Primary objective:
Improve CoLA DistOpt + optimizer CPU offload iteration time by understanding why current overlap hurts, modeling the best possible overlap speedup, and prototyping offload optimizations that coalesce real D2H/H2D transfers and reduce synchronization.

Use ATC-Megatron as the primary stack.
Use CoLA as the optimization target and FullRank as the control when useful.

Main baseline:
3B, DP=4, seq1024, micro-batch 1, global batch 4.

Known measured reference points:
- CoLA DistOpt: 202.55 ms
- CoLA DistOpt + full optimizer CPU offload: 854.57 ms
- CoLA DistOpt + offload fraction 0.5: 561.07 ms
- CoLA DistOpt + full offload + current overlap d2h/h2d: 1047.52 ms
- CoLA DistOpt + full offload + current overlap + CPU optimizer grouping prototype: 1014.69 ms

Core questions:
1. Why does current optimizer CPU offload overlap hurt runtime?
2. What overhead does overlap introduce compared with non-overlap offload?
3. What is the theoretical lower bound / upper-bound speedup if data movement is fully hidden?
4. Can we improve CoLA offload by coalescing actual D2H/H2D transfers, reusing pinned staging buffers, batching offloaded state movement, and reducing synchronization?
5. Why do FSDP-only rows fail in the current run when prior runs succeeded? Identify the config difference and whether this matters for offload-focused conclusions.

Record progress:
Create a new results directory:
benchmarks/0526/offload_overlap_optimization_<timestamp>/

Maintain:
- PROGRESS.md
- RUN_MATRIX.csv
- HYPOTHESES.md
- REPORT.md

Update PROGRESS.md after every meaningful step:
- code inspection
- hypothesis added/revised
- run launched/completed/failed/skipped
- timer parsed
- Nsight profile collected
- timeline figure generated
- theoretical model updated
- optimization implemented
- validation result added
- blocker found

RUN_MATRIX.csv should track:
- model: 3B or reduced/profile shape
- impl: CoLA or FullRank
- runtime mode
- offload mode/fraction
- overlap enabled/disabled
- optimization variant
- status
- job id
- command path
- log path
- Nsight .nsys-rep path
- Nsight .sqlite path
- parsed metric path
- failure/skip reason
- notes

Tasks:

1. Reconcile FSDP-only failure with prior successful FSDP runs.
Inspect prior successful FSDP run commands/logs and current failed FSDP-only commands/logs.
Compare settings such as:
- optimizer type / torch optimizer fallback
- FSDP_USE_TORCH_OPTIMIZER
- precision-aware optimizer flags
- Apex/TE fused optimizer path
- CUDA_LAUNCH_BLOCKING
- TORCH_NCCL_ASYNC_ERROR_HANDLING
- CUDA_DEVICE_MAX_CONNECTIONS
- data-parallel sharding strategy
- any harness defaults that changed

Output:
- short explanation of why current FSDP-only rows failed
- whether the failure is relevant to optimizer offload optimization
- whether a stable FSDP-only command should be recorded for future comparison

2. Establish offload overlap baseline breakdown.
Run or parse existing timer breakdowns for CoLA 3B DP=4:
- DistOpt no offload
- DistOpt + full optimizer CPU offload
- DistOpt + full optimizer CPU offload + current overlap d2h/h2d
- DistOpt + optimizer CPU offload fraction 0.5
- DistOpt + full optimizer CPU offload + current overlap + existing CPU optimizer grouping prototype

Collect:
- iteration time
- forward-backward time
- grad sync time
- param all-gather time
- optimizer inner time
- optimizer copy time if available
- optimizer total time
- allocated / max allocated / reserved memory

Report the runtime breakdown clearly and answer:
- Which timer increases when overlap is enabled?
- Does overlap increase forward-backward time, optimizer time, all-gather time, synchronization, or multiple components?
- Does overlap change memory?

3. Detailed Nsight profiling of offload overlap.
Use a manageable profile shape:
- 3B-width 4-layer model
- DP=4
- seq1024
- micro-batch 1
- global batch 4
- profile steps 7-10

Profile:
- CoLA DistOpt + full optimizer CPU offload, overlap disabled
- CoLA DistOpt + full optimizer CPU offload, overlap enabled
- CoLA DistOpt + full optimizer CPU offload, overlap enabled + current CPU optimizer grouping prototype
- optionally FullRank DistOpt + full optimizer CPU offload, overlap disabled/enabled if queue time allows

Save:
- .nsys-rep
- exported .sqlite
- raw log

From Nsight SQLite, extract:
- kernel count
- kernel duration summary
- CUDA API total time
- CUDA launch API time/count
- CUDA synchronization API time/count
- H2D memcpy count/bytes/duration
- D2H memcpy count/bytes/duration
- average/p50/p90/p99 memcpy size
- approximate bandwidth by direction
- approximate GPU idle gaps
- CPU-side time in optimizer/offload-related calls if visible

Generate timeline figures:
- at least one timeline-style figure or extracted event plot comparing overlap disabled vs overlap enabled
- show compute kernels, memcpy H2D/D2H, and synchronization/API regions if possible
- store under figures/
- include figure paths in REPORT.md

4. Theoretical performance model.
Build a simple model for optimizer CPU offload time.

For each profiled row, estimate:
- total bytes transferred H2D
- total bytes transferred D2H
- peak or measured bandwidth for H2D/D2H
- lower-bound transfer time = bytes / peak bandwidth
- observed transfer time from Nsight memcpy durations
- CPU AdamW/update computation estimate if possible:
  - number of optimizer elements updated
  - approximate FLOPs per element for AdamW
  - CPU peak FLOPs or measured effective throughput if available
  - lower-bound CPU optimizer time
- synchronization/framework overhead = observed optimizer/offload time minus estimated transfer and CPU compute lower bounds

Also compute idealized upper bounds:
- no-overlap lower bound: compute time + transfer time + CPU optimizer time + unavoidable sync
- perfect-overlap lower bound: max(GPU compute time, transfer time + CPU optimizer time) plus unavoidable sync
- expected best possible speedup if offload is fully hidden

Report:
- whether current overlap is close to the model or far from it
- whether offload is bandwidth-limited, CPU-compute-limited, synchronization-limited, or framework-overhead-limited
- whether CoLA has better or worse overlap opportunity than FullRank

5. Search for related ideas before implementation.
Look at primary sources or official docs/papers/source code for ideas on:
- ZeRO-Offload
- ZeRO-Infinity
- PyTorch/FSDP CPU offload
- DeepSpeed offload overlap
- offload staging buffers
- pinned memory reuse
- transfer coalescing
- async H2D/D2H overlap
- optimizer state partitioning and bucketing

Prefer primary sources:
- official docs
- papers
- source code

Record sources in REPORT.md with links and one-line relevance notes.
Do not over-browse; focus on ideas that directly inform CoLA offload optimization.

6. Implement targeted offload optimizations.
Start from the current ATC-Megatron optimizer CPU offload path.

Prioritize:
- coalesce actual D2H/H2D transfers into larger slabs
- reuse pinned host staging buffers
- batch optimizer state movement
- reduce per-small-tensor transfer setup
- reduce synchronization around D2H/H2D transfers
- preserve correctness and optimizer semantics
- keep feature env-gated or flag-gated so baseline remains unchanged

Candidate variants:
- transfer coalescing only
- pinned staging-buffer reuse only
- transfer coalescing + staging reuse
- transfer coalescing + current overlap
- transfer coalescing + partial offload fraction 0.5

Avoid large unrelated refactors.

7. Validate optimizations.
Compare against:
- CoLA DistOpt no offload
- CoLA DistOpt + full offload baseline
- CoLA DistOpt + offload fraction 0.5
- CoLA DistOpt + current overlap
- CoLA DistOpt + current overlap + CPU optimizer grouping prototype
- new optimized variants

For each variant collect:
- iteration time
- optimizer total time
- optimizer inner time
- optimizer copy time if available
- forward-backward time
- memory: allocated / max allocated / reserved
- transfer count/bytes/size distribution if profiled
- correctness sanity: loss is finite and training reaches measured iterations

Success criteria:
- optimization must improve full-offload iteration time without exceeding offload_fraction_0.5 memory, or
- optimization must improve the speed-memory Pareto frontier, or
- optimization must produce clear evidence explaining why current architecture cannot be improved without deeper runtime changes.

8. Deliver final report.
Create:
benchmarks/0526/offload_overlap_optimization_<timestamp>/REPORT.md

REPORT.md must include:
- scope and problem statement
- progress/run matrix
- FSDP-only failure reconciliation
- baseline overlap vs no-overlap timer breakdown
- Nsight profile summary table
- timeline figure paths
- theoretical model table
- external source notes
- optimization attempts and commands
- before/after result table
- failure/skipped rows with reasons
- final diagnosis
- next recommended optimization

Final answers required:
- Why does current overlap hurt?
- Which overhead increases when overlap is enabled?
- What is the theoretical best possible speedup from perfect offload overlap?
- Is current offload bottleneck bandwidth, CPU AdamW compute, synchronization, or framework overhead?
- Did transfer coalescing/staging reuse improve performance?
- What is the best speed-memory tradeoff found?
- What should be tried next?

Do not mark complete until every conclusion is backed by logs, timers, Nsight profiles, code inspection, theoretical model calculations, or explicitly recorded failed/unsupported experiments.
