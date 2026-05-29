# Repository Guidelines

## Skills

The `skills/` directory contains structured guides for common tasks (running
tests, building containers, managing dependencies, submitting SLURM jobs, etc.).
**Always read the relevant `SKILL.md` before starting any task it covers —
skills are mandatory context, not optional background reading.**

**Workflow — mandatory order for every task:**
1. **Pull information first.** Read the commit, PR, error log, file, or
   whatever artifact the task is about. Do not reason about it yet.
2. **Select and invoke the skill.** Based on what you just read, identify
   the relevant skill and invoke it before forming any answer or plan.
3. **Answer or implement.** Only after the skill is loaded, use its context
   to reason, diagnose, or write code.

Never skip or reorder these steps. Do not wait for the user to name the right
skill keyword — infer it from the artifact you read.

## Progress And Logging

Use the `progress/` directory for day-by-day work records and generated run
artifacts.

- For each day of meaningful work, create or reuse a dated subdirectory under
  `progress/`, such as `progress/YYYY-MM-DD/`.
- For each distinct goal, create or reuse one goal-specific subdirectory under
  that date, such as `progress/YYYY-MM-DD/Goal1_Smoketest/`.
- Record that goal's work in its goal directory, including the task,
  assumptions, commands run, results/status, blockers, and next steps.
- Put related logs, parsed results, benchmark outputs, and other generated
  artifacts under the same goal directory when practical. Use subdirectories
  such as `logs/`, `results/`, or `artifacts/` if that keeps the record clearer.
- Keep progress records concise but sufficient for another agent or engineer to
  resume the work without reconstructing context from shell history.

## Polaris ATC Environment

When working on this Polaris ATC-Megatron checkout, enable the existing Python
environment and keep caches/build artifacts on Eagle rather than under `/home`.

```bash
module use /soft/modulefiles
module load conda

export ATC_WORK=/eagle/TensorCompress/zhengyangwang/atc-megatron
export UV_CACHE_DIR=$ATC_WORK/uv-cache
export PIP_CACHE_DIR=$ATC_WORK/pip-cache
export CONDA_PKGS_DIRS=$ATC_WORK/conda-pkgs
export TMPDIR=$ATC_WORK/tmp
export TORCH_EXTENSIONS_DIR=$ATC_WORK/torch-extensions
export XDG_CACHE_HOME=$ATC_WORK/cache
export HF_HOME=$ATC_WORK/hf-cache
export PATH="$HOME/.local/bin:$PATH"

mkdir -p \
  $UV_CACHE_DIR \
  $PIP_CACHE_DIR \
  $CONDA_PKGS_DIRS \
  $TMPDIR \
  $TORCH_EXTENSIONS_DIR \
  $XDG_CACHE_HOME \
  $HF_HOME

conda activate $ATC_WORK/envs/atc-py312

cd /home/zhengyangwang/offloading/ATC-Megatron
```

Verify the environment before running or debugging training code:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -c "import megatron; print(megatron)"
```

For single-node tests or smoke runs, prefer using the current interactive
compute node when one is available instead of submitting a separate debug job.
Submit a scheduler job when the run requires more than one node, a longer wall
time, or resources that are not available in the current allocation.

## Coding Behavior Guidelines

These guidelines reduce common LLM coding mistakes. They bias toward caution
over speed; for trivial tasks, use judgment.

### Think Before Coding

Do not assume. Do not hide confusion. Surface tradeoffs.

Before implementing:

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them instead of picking silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name what is confusing, and ask.

### Simplicity First

Use the minimum code that solves the problem. Add nothing speculative.

- Do not add features beyond what was asked.
- Do not add abstractions for single-use code.
- Do not add flexibility or configurability that was not requested.
- Do not add error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.
- Ask whether a senior engineer would call the solution overcomplicated. If yes, simplify.

### Surgical Changes

Touch only what is necessary. Clean up only your own mess.

When editing existing code:

- Do not improve adjacent code, comments, or formatting.
- Do not refactor things that are not broken.
- Match existing style, even if you would choose differently.
- If you notice unrelated dead code, mention it instead of deleting it.

When your changes create orphans:

- Remove imports, variables, and functions that your changes made unused.
- Do not remove pre-existing dead code unless asked.
- Every changed line should trace directly to the user's request.

### Goal-Driven Execution

Define success criteria and loop until verified.

Transform tasks into verifiable goals:

- "Add validation" means write tests for invalid inputs, then make them pass.
- "Fix the bug" means write a test that reproduces it, then make it pass.
- "Refactor X" means ensure tests pass before and after.

For multi-step tasks, state a brief plan:

1. Step: describe the work. Verify: describe the check.
2. Step: describe the work. Verify: describe the check.
3. Step: describe the work. Verify: describe the check.

These guidelines are working if diffs have fewer unnecessary changes, fewer
rewrites caused by overcomplication, and clarifying questions happen before
implementation mistakes.

## Contributing

### Pull Requests

- All PRs must be created as **drafts**. Use `gh pr create --draft` or the GitHub UI draft option.
- Never push branches directly to `https://github.com/NVIDIA/Megatron-LM`. You must push your branch to a personal fork (e.g. `https://github.com/<your-username>/Megatron-LM`), then open a PR from the fork's branch against `NVIDIA/Megatron-LM`.
- Read @docs/developer/contribute.md for the full contribution policy, including code style, commit message conventions, and issue guidelines.

### Code Quality

- After editing imports in any Python files, always run `uv run isort` on those files to fix import order before committing.
