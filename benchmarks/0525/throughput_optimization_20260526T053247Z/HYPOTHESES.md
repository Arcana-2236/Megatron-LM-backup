# Hypotheses

## H1: CoLA DistOpt Overhead Is Runtime Granularity

CoLA under Megatron distributed optimizer should reduce active memory and iteration time relative to FullRank, but leave a measurable overhead signature from finer-grained kernels and runtime events.

Evidence already available:
- Prior granularity diagnosis: 4-layer CoLA DistOpt had more kernels than FullRank DistOpt and a higher idle-gap fraction.
- Memory-liveness study: CoLA DistOpt did not materially increase inactive-split bytes, so allocator fragmentation is not the primary plain-DistOpt bottleneck.
- This study: 4-layer CoLA DistOpt profile had 8829 kernels vs 7059 for FullRank, average kernel duration 66.44 us vs 112.55 us, and idle-gap fraction 0.141 vs 0.085. This validates the runtime-granularity hypothesis for CoLA.

Validation required in this study:
- Re-run or reuse 3B timer rows with matching logs.
- Collect/reuse Nsight summaries for FullRank vs CoLA DistOpt.
- Test CUDA Graph/scoped graph where supported to see whether reducing launch/runtime overhead narrows the gap.

## H2: CPU Optimizer Offload Is Fixed-Cost/Transfer/Synchronization Bound

CoLA reduces optimizer state size, but optimizer CPU offload may still spend large time in CPU optimizer execution, D2H/H2D transfers, transfer setup, and synchronization. CoLA may have less data movement than FullRank but worse opportunity for overlap because its compute windows are shorter.

Evidence already available:
- Prior 3B timers show CoLA DistOpt optimizer time rises sharply with optimizer CPU offload.
- Prior Nsight offload summaries show many memcpy events and high CUDA API/synchronization time.
- This study: CoLA full optimizer offload took 854.57 ms/iter with 670.31 ms optimizer time; `--overlap-cpu-optimizer-d2h-h2d` worsened this to 1047.52 ms/iter and 806.38 ms optimizer time; `--optimizer-offload-fraction 0.5` improved to 561.07 ms/iter and 369.11 ms optimizer time at higher GPU memory. This supports optimizer/offload work and state placement as the dominant offload bottleneck.
- Prototype: grouping offloaded CPU parameters into 50M-element CPU optimizer buckets improved the overlap path only modestly (1014.69 ms/iter vs 1047.52 ms/iter) and remained slower than full offload without overlap, so the remaining problem is not just the number of CPU optimizer objects.

Validation required in this study:
- Identify exactly which offload flags are supported.
- Estimate transferred bytes and expected copy time.
- Compare observed optimizer/offload time to copy-time estimates and CPU optimizer timers.
- Test existing overlap/bucket/staging knobs one at a time.

## H3: FSDP CoLA Overhead Comes From Unit Granularity And Orchestration

FSDP reduces memory but can add all-gather/reduce-scatter, hooks, metadata work, and optimizer fallback overhead. CoLA's factorized layer structure may make FSDP units too fine or less compute-rich per runtime event.

Evidence already available:
- Prior timer diagnosis shows CoLA FSDP has much higher forward-backward time than CoLA DistOpt.
- Latest memory-liveness FSDP-only rows failed at first `optimizer.step()` for both FullRank and CoLA, full and 4-layer fallback, so current FSDP-only throughput conclusions need fresh revalidation or explicit failure recording.
- This study: 3B FSDP-only failed for both FullRank and CoLA with CUDA illegal memory access in `optimizer.step()`. FSDP+offload completed; CoLA FSDP+offload had higher forward-backward and all-gather than CoLA DistOpt+offload. Code inspection shows Megatron-FSDP units default to `TransformerLayer`, so CoLA factors are not individually wrapped as FSDP units; the overhead is more consistent with within-layer factor granularity plus FSDP/offload orchestration.

Validation required in this study:
- Discover FSDP wrapping/unit and communication-bucket knobs.
- Re-run baseline FSDP and FSDP+offload rows.
- Test coarser units or supported FSDP knobs if available.

## H4: CUDA Graph Can Recover CoLA Granularity Overhead But Is Fragile

CUDA Graph or scoped graph capture should help CoLA more than FullRank when kernel launch/runtime overhead is significant, but full-iteration graph capture may be incompatible with timer barriers, profiling, FSDP paths, or unsupported operations.

Evidence already available:
- Prior partial CUDA Graph logs suggest large CoLA DistOpt iteration-time improvement, but the runs were not clean timer rows and Nsight graph capture failed.
- This study: CoLA 3B-width 4-layer full-iteration CUDA Graph captured successfully and ran at 40.50 ms/iter after capture. A comparable non-graph CoLA 4-layer profile had pre-profile iterations around 52-57 ms, so graph capture recovers a meaningful portion of launch/runtime overhead. Timer sub-breakdowns are unavailable because graph capture requires disabling timer barriers.

Validation required in this study:
- Run graph rows with timing disabled or graph-safe metrics.
- Record exact failure path for unsupported graph configurations.
- Compare FullRank baseline, CoLA baseline, and CoLA graph using iteration time plus profiler evidence where available.

## H5: FSDP+Offload Reserved Headroom Is Cache/Staging, Not Fragmentation

FSDP+offload can increase reserved-inactive GPU memory and reserved-allocated gap while inactive-split bytes remain small. That points to allocator cache/headroom or staging/framework buffers rather than allocator fragmentation.

Evidence already available:
- Memory-liveness study: CoLA FSDP+offload steady state had lower inactive split than CoLA DistOpt but higher reserved-inactive/gap.

Validation required in this study:
- Correlate reserved headroom with throughput timers and profiler events.
- Keep memory overhead and throughput overhead separate in the report.
