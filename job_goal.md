/goal

Objective:
Build a reproducible Megatron-LM benchmarking workflow in /home/zhengyangwang/offloading/ATC-Megatron to compare Fullrank GPT models vs CoLA models on Polaris. Measure iteration time and peak GPU memory for 3B and 1B model sizes across memory-efficiency strategies.

Context:
- Primary repo: /home/zhengyangwang/offloading/ATC-Megatron
- Existing CoLA reference implementation: /home/zhengyangwang/offloading/Megatron-DeepSpeed
- Polaris compute nodes have limited /home quota, so avoid putting large env/cache/build artifacts in /home if possible. 
- Use Eagle/scratch/tmp locations for large caches, package caches, build dirs, logs, and benchmark outputs when practical.
- I am using Polaris, usually one node with 4 GPUs.
- Use a new environment if needed, but first inspect existing envs such as dspeed_env and current ATC-Megatron env. Reuse when safer.
- Make surgical code changes only. Do not rewrite unrelated Megatron code.

Hard requirements:
1. Maintain progress.md in /home/zhengyangwang/offloading/ATC-Megatron.
   - After every major step, append:
     - timestamp
     - what changed
     - commands run
     - result/status
     - next step
   - This is mandatory so work can be resumed after node/session loss.

2. Create a benchmark report:
   - Write final report to benchmark_report.md in the ATC-Megatron repo.
   - Include environment details, code changes, commands, caveats, and final tables.
   - Include any OOM/crash/failure rows with the observed failure reason.

3. Final benchmark tables:
   Produce one table for 3B and one table for 1B.
   Each row is a strategy combination.
   Required result columns:
   - strategy
   - offloading mode
   - cuda graph
   - Fullrank iteration time
   - Fullrank peak memory
   - CoLA iteration time
   - CoLA peak memory

   Use consistent units:
   - iteration time in ms/iter or sec/iter
   - peak memory in GiB per GPU, preferably max across ranks

Benchmark matrix:
For each model size: 3B and 1B
For each model type: Fullrank and CoLA
Run the following base parallelism strategies on DP=4:
1. DP=4 baseline, no distributed optimizer, no FSDP
2. DP=4 + distributed optimizer
3. DP=4 + FSDP

For each base strategy, test:
- offloading disabled / offloading enabled
- CUDA graph disabled / CUDA graph enabled

This gives 3 * 2 * 2 = 12 rows per model size. If a combination is unsupported, crashes, or OOMs, mark it clearly instead of hiding it. And try to solve the error and have a number.

Clarifications to determine and document:
- Confirm whether Megatron distributed optimizer is equivalent or closest to ZeRO-1 in this setup.
- Confirm whether each FSDP mode behaves like ZeRO-3/full parameter+grad+optimizer sharding.
- Megatron may have both torch FSDP and Megatron custom FSDP. Try both if supported. If both are viable, either:
  - add them as separate strategy rows, or
  - choose the more stable one and document why.
- For offloading, explicitly document what is offloaded: optimizer states, parameters, gradients, activations, or CPU/NVMe. Prefer optimizer-state offload first if multiple forms exist.
- CUDA graph support may be incompatible with FSDP or optimizer/offload combinations. Test and document rather than assuming. If incompatible, try to look for solutions to solve it.

Suggested workflow:
1. Inspect the repo and current scripts.
   - Find existing Megatron benchmark/pretrain scripts.
   - Find current env files, requirements, pyproject, uv usage, conda envs, and module load assumptions.
   - Find how current jobs are launched on Polaris.
   - Record findings in progress.md.

2. Environment setup.
   - Avoid exhausting /home quota.
   - Prefer setting cache/build locations away from /home if available:
     UV_CACHE_DIR, PIP_CACHE_DIR, CONDA_PKGS_DIRS, TORCH_EXTENSIONS_DIR, TMPDIR.
   - Try to use uv only if it works cleanly on Polaris and does not explode quota.
   - Otherwise reuse or clone a conda env such as dspeed_env/current working env.
   - Verify imports: torch, megatron, apex/transformer-engine if needed.
   - Record exact env/module/package versions in benchmark_report.md.

3. Implement CoLA support in ATC-Megatron.
   - Study CoLA implementation in /home/zhengyangwang/offloading/Megatron-DeepSpeed.
   - Port only the minimal required model/layer changes into ATC-Megatron.
   - Add flags/config needed to switch between Fullrank and CoLA.
   - Keep Fullrank behavior unchanged.
   - Add a small smoke test to verify both model variants build and run at tiny size.

4. Build benchmark harness.
   - Create scripts under a new benchmark folder, e.g. benchmarks/cola_memory/
   - Include launch scripts for Polaris/PBS and local command wrappers.
   - Capture stdout/stderr logs per run.
   - Extract iteration time from Megatron logs after warmup.
   - Capture peak memory per rank using torch.cuda.max_memory_allocated/reserved if possible, or parse existing Megatron memory logs if already available.
   - Save raw results as CSV/JSON plus human-readable markdown tables.

5. Run 1B first.
   - Run a tiny smoke test first.
   - Then run the full 1B matrix.
   - If a run fails, record OOM/crash/unsupported and continue.

6. Run 3B second.
   - Use the same harness and matrix.
   - Mark OOM if it cannot fit.

7. Write final benchmark_report.md.
   - Include tables for 1B and 3B.
   - Include methodology, environment, exact commit/status, commands, and caveats.
   - Include notes on unsupported combinations, especially FSDP + CUDA graph if it crashes.

Execution safety:
- Do not make destructive git changes.
- Do not delete user files.
- Before editing, inspect files and make minimal patches.
- Use git diff frequently to track changes.
- Keep all generated benchmark logs/results under a clearly named folder.
- If a compute allocation is about to expire, write current status and next command to progress.md before continuing.
- Prefer running experiments directly on the current active compute node. Only submit new qsub jobs if the current allocation is unavailable, insufficient, or near expiration.
- Use the existing Python environment when possible. If the repo/dependencies require Python 3.12, try current env first, if it will have too many failures, create or upgrade an environment accordingly, but first check /home quota and route caches/build/temp artifacts outside /home using PIP_CACHE_DIR, UV_CACHE_DIR, CONDA_PKGS_DIRS, TORCH_EXTENSIONS_DIR, and TMPDIR.


Initial command hints:
- Start from:
  cd /home/zhengyangwang/offloading/ATC-Megatron
  pwd
  git status
  df -h .
  quota -s || true
  module list
  conda env list
  python -c "import torch; print(torch.__version__, torch.version.cuda)"