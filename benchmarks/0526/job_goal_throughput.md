Optimize ATC-Megatron throughput for CoLA-style factorized Transformer training, focusing on granularity-induced runtime overhead in memory-management systems.

Primary objective:
Identify and reduce throughput overheads that appear when CoLA interacts with memory-management/runtime systems such as optimizer CPU offload, FSDP, distributed optimizer, CUDA Graph, and related overlap/prefetch mechanisms.

Core research question:
CoLA reduces model state and FLOPs, but introduces finer-grained computation and runtime events. Which runtime overheads prevent CoLA from converting memory reduction into speedup, and which existing or new runtime optimizations can recover performance?

Use ATC-Megatron as the primary stack.
Use FullRank as the control and CoLA as the compressed/factorized architecture.

Record progress:
Create a new results directory:
benchmarks/0525/throughput_optimization_<timestamp>/

Maintain:
- PROGRESS.md
- RUN_MATRIX.csv
- HYPOTHESES.md
- REPORT.md

Update PROGRESS.md after every meaningful step:
- hypothesis added/revised
- run launched/completed/failed/skipped
- profile collected
- metric extracted
- optimization tried
- result validated
- conclusion updated
- blocker found

RUN_MATRIX.csv should track:
- model size/config
- model impl: FullRank or CoLA
- runtime mode
- optimization tried
- status
- job id
- command
- log path
- profiler path
- parsed metrics path
- failure/skip reason
- notes

Workflow:

1. Diagnose baseline bottlenecks.
For each target runtime mode, first run a baseline with detailed timers and lightweight profiling.

Start with:
- 3B model, DP=4, seq1024, micro-batch 1, global batch 4
- optionally use 3B-width 4-layer model for expensive profiles

Runtime modes:
- Megatron distributed optimizer
- distributed optimizer + optimizer CPU offload
- Megatron-FSDP
- Megatron-FSDP + optimizer CPU offload
- CUDA Graph / scoped graph capture where supported
- Megatron-DeepSpeed ZeRO-1 only as a secondary/motivation baseline if useful

Collect:
- iteration time
- forward-backward time
- optimizer time
- optimizer inner time
- optimizer copy time
- grad sync time
- param all-gather time
- allocated / max allocated / reserved memory
- kernel count and kernel duration if profiled
- CUDA API time
- memcpy count, total bytes, average size
- GPU idle gaps if available
- CPU-side overhead if available

2. Build bottleneck models.
For each bottleneck, build a simple explanatory model.

For CPU optimizer offload:
- identify what is offloaded: optimizer state only, gradients, weights, or combinations supported by ATC-Megatron
- estimate bytes transferred per iteration
- compute expected transfer time from measured or theoretical bandwidth
- compare expected transfer time to observed optimizer/offload time
- separate data movement, CPU optimizer computation, synchronization, and framework overhead where possible

For FSDP:
- identify all-gather/reduce-scatter points
- measure param all-gather, grad sync, forward-backward overhead
- estimate communication volume and compare with observed time
- inspect whether FSDP unit granularity is too fine for CoLA

For CoLA kernel/runtime granularity:
- compare FullRank vs CoLA kernel count, average kernel duration, GPU idle gaps, and CUDA launch overhead
- test whether CUDA Graph/scoped graph capture reduces the gap
- identify whether remaining overhead is kernel launch, CPU scheduling, synchronization, or memory-management bookkeeping

3. Try existing supported optimizations first.
Before implementing new code, search the ATC-Megatron codebase and configs for existing knobs.

For offload:
- check whether ATC-Megatron supports overlapping optimizer CPU offload with computation
- identify supported flags for offload overlap, offload bucket size, prefetching, pinned memory, staging buffer reuse, async copy, or optimizer state placement
- test relevant hyperparameters one at a time
- measure speedup and memory impact

For FSDP:
- check supported FSDP wrapping/unit granularity controls
- check prefetch, overlap, bucket size, communication overlap, limit-all-gathers, optimizer options, and torch optimizer fallback flags
- test coarser FSDP units and larger communication units if supported
- measure speedup and memory impact

For CUDA Graph:
- check whether full-iteration or scoped graph capture is supported for the target runtime
- test CoLA vs FullRank benefit
- if graph capture fails, record failure reason and identify the operation/path that breaks capture

4. Then prototype targeted optimizations.
If existing knobs are insufficient, propose and implement minimal targeted changes.

Candidate offload optimizations:
- coalesce optimizer/offload state into larger slabs
- batch H2D/D2H transfers
- reuse pinned host staging buffers
- avoid per-small-tensor transfer setup
- overlap offload transfers with longer compute windows
- reduce synchronization around each transfer
- tune offload bucket size for CoLA

Candidate FSDP optimizations:
- coarser FSDP units for factorized layers
- larger communication buckets
- better prefetch timing
- fewer metadata/hook operations
- isolate and reduce optimizer fallback overhead
- avoid communication points around very small factorized submodules

Candidate CoLA runtime optimizations:
- CUDA Graph or scoped graph capture around stable backward regions
- group small kernels where possible
- reduce Python/runtime bookkeeping
- reduce per-small-tensor hook overhead
- improve parameter/gradient layout for factorized modules

5. Validate each optimization.
For every optimization, compare against the matching baseline:
- FullRank baseline
- CoLA baseline
- CoLA optimized

Report:
- speedup
- memory impact
- timer breakdown change
- profiler evidence
- whether improvement comes from reduced data movement, reduced CPU optimizer time, reduced CUDA API overhead, better overlap, fewer idle gaps, or lower communication overhead

Do not accept an optimization as successful based only on iteration time. Explain why it improved using timers/profilers.

6. Use external systems work only as supporting idea generation.
If needed, search official docs and papers for related systems ideas on:
- ZeRO-Offload
- FSDP prefetch/overlap
- activation/offload overlap
- CUDA Graphs in training
- tensor/gradient coalescing
- communication bucketing
- parameter flattening
- offload staging buffers

Prefer primary sources: papers, official docs, or source code.
Record any external source used in REPORT.md with links and a short note on how it informs the optimization.

Deliverables:
Create:
benchmarks/0525/throughput_optimization_<timestamp>/REPORT.md

REPORT.md must include:
1. Problem statement and scope.
2. Baseline timer breakdown tables.
3. Nsight/profiler summary tables where available.
4. Bottleneck model for offload, FSDP, and CoLA granularity.
5. Existing runtime knobs discovered.
6. Optimization attempts and commands.
7. Before/after result tables.
8. Failure/skipped configs with concrete reasons.
9. Evidence-backed conclusions.
10. Next recommended optimization.

Final answers to produce:
- What is the dominant throughput bottleneck for CoLA under each runtime mode?
- Does CoLA give better or worse opportunity for offload overlap compared with FullRank?
- Is offload limited by bandwidth, CPU optimizer computation, synchronization, or framework overhead?
- Is FSDP limited by communication volume, unit granularity, prefetch/overlap, optimizer fallback, or orchestration overhead?
- Which existing ATC-Megatron knobs improve performance?
- Which new optimization is most promising?
- Which result most strongly supports the paper’s granularity-aware memory-management thesis?

Important constraints:
- Keep FullRank as the control.
- Keep CoLA as the target architecture.
- Separate memory overhead from throughput overhead.
- Clearly distinguish measured evidence from hypothesis.
- Do not mark complete until every conclusion is backed by logs, timers, profiler data, code inspection, or explicitly recorded failed/unsupported experiments.
