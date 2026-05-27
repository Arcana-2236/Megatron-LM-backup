# Hypotheses

## H1: Current Offload Overlap Hurts Because It Adds Per-Parameter Runtime Work

The current overlap path in `HybridDeviceOptimizer` creates many CPU optimizer objects and stream/event interactions. CoLA has many factorized parameters and shorter compute windows, so overlap overhead may be exposed rather than hidden.

Evidence required:
- timer comparison of offload no-overlap vs overlap
- Nsight CUDA API/sync/memcpy comparison
- code inspection of CPU offload implementation

Status: supported. 3B timers show overlap is slower than full offload. 4-layer Nsight rows show H2D/D2H copy counts and bytes are effectively unchanged while sync/API overhead and idle gaps rise.

## H2: Reducing Offload Fraction Improves Speed By Moving Work Back To GPU

The existing 0.5 offload-fraction result likely improves throughput by reducing CPU optimizer/state movement work at the cost of higher GPU memory. This is a speed-memory Pareto point, not a pure overlap fix.

Evidence required:
- timer table including memory columns
- model comparing state moved/offloaded work against observed optimizer time

Status: supported. Offload fraction 0.5 improves CoLA iteration time from 854.57 ms to 561.07 ms at higher max allocated memory, so it is a speed-memory Pareto tradeoff rather than an overlap fix.

## H3: Transfer Coalescing And Staging Reuse Are Needed Beyond CPU Optimizer Grouping

The prior CPU optimizer grouping prototype only modestly improved current overlap, so actual D2H/H2D transfer coalescing, pinned staging reuse, and synchronization reduction may be needed to improve full-offload throughput.

Evidence required:
- implementation inspection
- prototype result against full-offload and overlap baselines
- Nsight transfer count/size distributions

Status: supported and refined. CPU optimizer grouping improves overlap-only but does not reduce H2D/D2H copy counts or byte volume; foreach-copy batching validates but is neutral on 3B throughput. The slab-copy prototype reduces H2D copies from 426 to 282 and D2H copies from 366 to 222 in the 4-layer Nsight profile, but adds D2D pack/scatter traffic and regresses full 3B throughput to 1141.29 ms. The next required step is not naive staging, but persistent flattened layouts or fused pack/scatter that coalesce host transfers without adding multi-GB D2D overhead.

## H4: FSDP-Only Failure Is Orthogonal Unless It Shares Optimizer-Offload Root Cause

FSDP-only failures must be reconciled against prior successful rows, but offload optimization conclusions should remain valid if the failure is tied to FSDP-only optimizer semantics rather than CPU offload.

Evidence required:
- command/log comparison
- config difference table

Status: supported. FSDP-only failure reproduces with `FSDP_USE_TORCH_OPTIMIZER=0`; prior successful rows used `fsdp_use_torch_optimizer=1` plus diagnostic blocking/NCCL envs. This is orthogonal to DistOpt CPU offload conclusions.
