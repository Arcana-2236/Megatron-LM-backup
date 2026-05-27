#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path


STUDY_DIR = Path(__file__).resolve().parent
ROOT_DIR = STUDY_DIR.parents[2]
sys.path.insert(0, str(ROOT_DIR / "benchmarks" / "cola_memory"))

from parse_timer_breakdown import parse_log  # noqa: E402
from summarize_nsys_sqlite import summarize as summarize_nsys  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")


def metadata_rows() -> list[dict]:
    rows = []
    for meta_path in sorted((STUDY_DIR / "meta").glob("*.json")):
        meta = json.loads(meta_path.read_text())
        warmup = int(meta.get("warmup_iters", 5))
        log_path = Path(meta["log_file"])
        parsed = parse_log(log_path, warmup)
        if (
            parsed.get("status") == "failed"
            and parsed.get("avg_iteration_ms") is not None
            and log_path.exists()
            and "[after training is done]" in log_path.read_text(errors="replace")
        ):
            parsed["status"] = "ok"
            parsed["post_training_caveat"] = "log reached training completion before later teardown signal"
            parsed["failure_reason"] = ""
        if parsed.get("status") == "failed":
            parsed["failure_reason"] = classify_failure(log_path, parsed.get("failure_reason", ""))
        rows.append(
            {
                **meta,
                "meta_file": str(meta_path),
                "cmd_file": str(STUDY_DIR / "meta" / f"{meta['run_id']}.cmd"),
                **parsed,
            }
        )
    return rows


def classify_failure(log_path: Path, fallback: str) -> str:
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    if "CUDA error: an illegal memory access was encountered" in text:
        if "optimizer.step()" in text or "optimizer-inner-step" in text:
            return "CUDA illegal memory access during optimizer.step()/optimizer-inner-step; NCCL watchdog abort"
        return "CUDA illegal memory access; NCCL watchdog abort"
    if "CUDA-capable device(s) is/are busy or unavailable" in text:
        return "CUDA device busy or unavailable before training startup"
    if "CUDA out of memory" in text or "torch.OutOfMemoryError" in text:
        return "CUDA out of memory"
    if "ChildFailedError" in text:
        return "torch.distributed.elastic ChildFailedError"
    if "RUN_TIMEOUT" in text:
        return "run timeout"
    return fallback


def nsys_rows() -> list[dict]:
    rows = []
    for sqlite_path in sorted((STUDY_DIR / "profiles").glob("*.sqlite")):
        try:
            row = summarize_nsys(sqlite_path)
        except Exception as exc:  # keep failed exports visible in artifacts
            row = {"sqlite_file": str(sqlite_path), "status": "failed", "failure_reason": repr(exc)}
        rows.append(row)
    return rows


def markdown_table(rows: list[dict], columns: list[tuple[str, str]], limit: int | None = None) -> str:
    if not rows:
        return "No rows yet.\n"
    shown = rows[:limit] if limit else rows
    header = "| " + " | ".join(title for title, _key in columns) + " |"
    sep = "| " + " | ".join("---" for _title, _key in columns) + " |"
    body = []
    for row in shown:
        values = []
        for _title, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append(str(value) if value is not None else "")
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def load_run_matrix() -> list[dict]:
    path = STUDY_DIR / "RUN_MATRIX.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def generate_report(timer_rows: list[dict], profile_rows: list[dict]) -> None:
    report_path = STUDY_DIR / "REPORT.md"
    matrix_rows = load_run_matrix()
    done_rows = [row for row in timer_rows if row.get("status") == "ok"]
    failed_rows = [row for row in timer_rows if row.get("status") == "failed"]

    lines = [
        "# ATC-Megatron Throughput Optimization For CoLA",
        "",
        "Status: in progress. Conclusions are separated into measured evidence, prior evidence, hypothesis, failed, or pending.",
        "",
        "## Problem Statement And Scope",
        "",
        "This study investigates why CoLA-style factorized Transformer training does not always convert memory and FLOP reductions into proportional throughput speedups in ATC-Megatron. FullRank is the control and CoLA is the compressed/factorized target. The focus is granularity-induced runtime overhead in memory-management systems: Megatron distributed optimizer, optimizer CPU offload, Megatron-FSDP, FSDP+offload, CUDA Graph/scoped graph capture, and related overlap/prefetch mechanisms.",
        "",
        "Primary model target: 3B, DP=4, sequence length 1024, micro-batch 1, global batch 4. Expensive profiling may use a 3B-width 4-layer fallback and must be marked as such.",
        "",
        "## Questions Or Clarifications",
        "",
        "- No user clarification is currently required. If a later unsupported runtime path needs a policy decision, it will be recorded here.",
        "",
        "## Run Matrix",
        "",
        markdown_table(
            matrix_rows,
            [
                ("model", "model_size_config"),
                ("impl", "model_impl"),
                ("runtime", "runtime_mode"),
                ("optimization", "optimization_tried"),
                ("status", "status"),
                ("job", "job_id"),
                ("log", "log_path"),
                ("profiler", "profiler_path"),
                ("reason", "failure_skip_reason"),
            ],
        ),
        "",
        "## Baseline Timer Breakdown Tables",
        "",
        markdown_table(
            timer_rows,
            [
                ("run_id", "run_id"),
                ("impl", "model_impl"),
                ("strategy", "strategy"),
                ("offload", "offload"),
                ("cg", "cuda_graph"),
                ("status", "status"),
                ("iter ms", "avg_iteration_ms"),
                ("fwd-bwd ms", "forward_backward_ms"),
                ("grad sync ms", "grad_sync_ms"),
                ("all-gather ms", "param_all_gather_ms"),
                ("opt inner ms", "optimizer_inner_ms"),
                ("opt total ms", "optimizer_total_ms"),
                ("alloc MB", "allocated_mb"),
                ("max alloc MB", "max_allocated_mb"),
                ("reserved MB", "reserved_mb"),
            ],
        ),
        "",
        "## Nsight / Profiler Summary Tables",
        "",
        markdown_table(
            profile_rows,
            [
                ("sqlite", "sqlite_file"),
                ("kernels", "kernel_count"),
                ("avg kernel us", "avg_kernel_us"),
                ("idle frac", "idle_gap_fraction"),
                ("CUDA API ms", "total_cuda_api_us"),
                ("launch APIs", "cuda_launch_api_count"),
                ("sync APIs", "cuda_sync_api_count"),
                ("memcpy count", "memcpy_count"),
                ("memcpy bytes", "total_memcpy_bytes"),
            ],
        ),
        "",
        "## Bottleneck Models",
        "",
        "### CPU Optimizer Offload",
        "",
        "Hypothesis: offload is not purely bandwidth-bound. CoLA transfers fewer bytes than FullRank, but shorter compute windows and many runtime events can leave optimizer CPU work, transfer setup, CUDA API synchronization, and framework overhead exposed.",
        "",
        "Measured rows in this study: " + str(len(done_rows)) + " timer rows, " + str(len(profile_rows)) + " profiler rows.",
        "",
        "### FSDP",
        "",
        "Hypothesis: CoLA FSDP is limited by communication/orchestration and unit granularity rather than raw communication volume alone. FSDP-only rows must be treated carefully because latest memory-liveness FSDP-only rows failed at first optimizer step.",
        "",
        "### CoLA Kernel/Runtime Granularity",
        "",
        "Hypothesis: CoLA creates more, shorter kernels and more CPU/runtime events; CUDA Graph or scoped capture can recover part of this overhead if capture succeeds.",
        "",
        "## Existing Runtime Knobs Discovered",
        "",
        "| area | knob | evidence |",
        "|---|---|---|",
        "| Optimizer CPU offload | `--optimizer-offload-fraction` | `megatron/training/arguments.py` |",
        "| Optimizer CPU offload | `--overlap-cpu-optimizer-d2h-h2d` | `megatron/core/optimizer/cpu_offloading/README.md` recommends it |",
        "| Optimizer CPU offload | `--use-torch-optimizer-for-cpu-offload` | `megatron/training/arguments.py` |",
        "| Optimizer CPU offload | `--no-pin-cpu-grads`, `--no-pin-cpu-params` | `megatron/training/arguments.py` |",
        "| Distributed optimizer | `--overlap-grad-reduce`, `--overlap-param-gather`, `--overlap-param-gather-with-optimizer-step` | `megatron/training/arguments.py` |",
        "| Distributed optimizer | `--ddp-num-buckets`, `--ddp-bucket-size`, `--ddp-pad-buckets-for-high-nccl-busbw` | `megatron/training/arguments.py` |",
        "| FSDP | `--fsdp-double-buffer` | `megatron/training/arguments.py` says it reuses temporary communication memory |",
        "| FSDP | `--suggested-communication-unit-size` | `megatron/training/arguments.py` says it affects communication buffer size and all-gather prefetch |",
        "| FSDP | `--use-nccl-ub`, `--fsdp-manual-registration` | `megatron/training/arguments.py` |",
        "| CUDA Graph | `--cuda-graph-impl`, `--cuda-graph-modules`, mempool/backward retain flags | `megatron/training/arguments.py` |",
        "",
        "## Optimization Attempts And Commands",
        "",
        "Commands are captured in `meta/*.cmd` for launched rows.",
        "",
        "## Before / After Result Tables",
        "",
        "Pending controlled optimization rows.",
        "",
        "## Failure / Skipped Configs",
        "",
        markdown_table(
            failed_rows,
            [("run_id", "run_id"), ("status", "status"), ("failure", "failure_reason"), ("log", "log_file")],
        ),
        "",
        "## Evidence-Backed Conclusions",
        "",
        "- Prior evidence: CoLA under plain DistOpt reduces memory and improves iteration time, while exposing more/finer kernels and higher idle gaps.",
        "- Prior evidence: DistOpt+offload is dominated by optimizer/offload runtime, memcpy activity, and synchronization/API overhead.",
        "- Prior evidence: FSDP reduces memory but increases forward-backward/orchestration time for CoLA.",
        "- Prior evidence: FSDP+offload increases reserved headroom/cache and optimizer time, while inactive-split metrics do not support allocator fragmentation as the main mechanism.",
        "- Measured evidence from this study will be added after baseline and optimization rows complete.",
        "",
        "## Next Recommended Optimization",
        "",
        "Test `--overlap-cpu-optimizer-d2h-h2d` first for CoLA DistOpt+offload because the source README explicitly recommends it and prior evidence points to offload synchronization/transfer overhead.",
        "",
        "## Raw Artifacts",
        "",
        "- Logs: `logs/*.log`",
        "- Metadata and commands: `meta/*.json`, `meta/*.cmd`",
        "- Nsight profiles: `profiles/*.nsys-rep`, `profiles/*.sqlite`",
        "- Parsed metrics: `parsed/*.csv`, `parsed/*.json`",
        "",
        "## External Sources",
        "",
        "No external sources used yet.",
        "",
    ]
    report_path.write_text("\n".join(lines))


def main() -> int:
    parsed_dir = STUDY_DIR / "parsed"
    timer_rows = metadata_rows()
    profile_rows = nsys_rows()
    write_csv(parsed_dir / "timer_breakdown.csv", timer_rows)
    write_json(parsed_dir / "timer_breakdown.json", timer_rows)
    write_csv(parsed_dir / "nsys_summary.csv", profile_rows)
    write_json(parsed_dir / "nsys_summary.json", profile_rows)
    generate_report(timer_rows, profile_rows)
    print(f"Wrote parsed artifacts under {parsed_dir}")
    print(f"Wrote report {STUDY_DIR / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
