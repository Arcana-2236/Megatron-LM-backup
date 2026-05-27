#!/usr/bin/env python3
import csv
import json
import os
import sqlite3
import sys
from pathlib import Path


STUDY_DIR = Path(__file__).resolve().parent
ROOT_DIR = STUDY_DIR.parents[2]
PREVIOUS_STUDY = ROOT_DIR / "benchmarks" / "0525" / "throughput_optimization_20260526T053247Z"

sys.path.insert(0, str(ROOT_DIR / "benchmarks" / "cola_memory"))
from parse_timer_breakdown import parse_log  # noqa: E402
from summarize_nsys_sqlite import summarize as summarize_nsys  # noqa: E402


BASELINE_RUN_IDS = [
    "throughput_3b_cola_distopt_baseline",
    "throughput_3b_cola_distopt_offload_baseline",
    "throughput_3b_cola_distopt_offload_overlap_d2h_h2d",
    "throughput_3b_cola_distopt_offload_fraction_0p5",
    "throughput_3b_cola_distopt_offload_overlap_group50m",
    "throughput_3b_fullrank_distopt_baseline",
    "throughput_3b_fullrank_distopt_offload_baseline",
]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


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
    if "RUN_TIMEOUT" in text:
        return "run timeout"
    return fallback


def imported_baseline_rows() -> list[dict]:
    rows = []
    previous_rows = read_csv(PREVIOUS_STUDY / "parsed" / "timer_breakdown.csv")
    by_id = {row.get("run_id"): row for row in previous_rows}
    for run_id in BASELINE_RUN_IDS:
        row = dict(by_id.get(run_id, {}))
        if not row:
            continue
        row["source"] = "imported_0525_throughput_study"
        rows.append(row)
    return rows


def local_timer_rows() -> list[dict]:
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
                "source": "local_0526_offload_overlap_study",
                "meta_file": str(meta_path),
                "cmd_file": str(STUDY_DIR / "meta" / f"{meta['run_id']}.cmd"),
                **parsed,
            }
        )
    return rows


def nsys_rows() -> list[dict]:
    rows = []
    for sqlite_path in sorted((STUDY_DIR / "profiles").glob("*.sqlite")):
        try:
            row = summarize_nsys(sqlite_path)
            row["status"] = "ok"
        except Exception as exc:
            row = {"sqlite_file": str(sqlite_path), "status": "failed", "failure_reason": repr(exc)}
        row["run_id"] = sqlite_path.stem
        rows.append(row)
    return rows


def build_theoretical_rows(timer_rows: list[dict], profile_rows: list[dict]) -> list[dict]:
    timers_by_id = {row.get("run_id"): row for row in timer_rows}
    rows = []
    for profile in profile_rows:
        run_id = profile.get("run_id", "")
        timer = timers_by_id.get(run_id, {})
        profile_steps = 1
        start_step = to_float(timer, "profile_step_start")
        end_step = to_float(timer, "profile_step_end")
        if start_step is not None and end_step is not None and end_step >= start_step:
            profile_steps = int(end_step - start_step + 1)
        h2d_count = to_float(profile, "h2d_memcpy_count") or 0
        d2h_count = to_float(profile, "d2h_memcpy_count") or 0
        h2d_bytes = (to_float(profile, "h2d_memcpy_bytes") or 0) / profile_steps
        d2h_bytes = (to_float(profile, "d2h_memcpy_bytes") or 0) / profile_steps
        h2d_time_ms = (to_float(profile, "h2d_avg_memcpy_us") or 0) * h2d_count / 1000 / profile_steps
        d2h_time_ms = (to_float(profile, "d2h_avg_memcpy_us") or 0) * d2h_count / 1000 / profile_steps
        transfer_time_ms = h2d_time_ms + d2h_time_ms
        optimizer_ms = to_float(timer, "optimizer_total_ms")
        fwd_bwd_ms = to_float(timer, "forward_backward_ms")
        lower_bound_no_overlap = (fwd_bwd_ms or 0) + transfer_time_ms
        perfect_overlap_bound = max((fwd_bwd_ms or 0), transfer_time_ms)
        rows.append(
            {
                "run_id": run_id,
                "variant": local_variant_name(timers_by_id.get(run_id, {})),
                "h2d_gb": h2d_bytes / 1e9,
                "d2h_gb": d2h_bytes / 1e9,
                "h2d_memcpy_time_ms": h2d_time_ms,
                "d2h_memcpy_time_ms": d2h_time_ms,
                "transfer_time_ms": transfer_time_ms,
                "optimizer_total_ms": optimizer_ms,
                "forward_backward_ms": fwd_bwd_ms,
                "profile_steps": profile_steps,
                "sync_api_ms_per_iter": (to_float(profile, "total_cuda_sync_api_us") or 0)
                / 1000
                / profile_steps,
                "cuda_api_ms_per_iter": (to_float(profile, "total_cuda_api_us") or 0)
                / 1000
                / profile_steps,
                "idle_gap_fraction": to_float(profile, "idle_gap_fraction"),
                "observed_minus_transfer_ms": (optimizer_ms - transfer_time_ms)
                if optimizer_ms is not None
                else None,
                "ideal_no_overlap_lower_bound_ms": lower_bound_no_overlap,
                "ideal_perfect_overlap_lower_bound_ms": perfect_overlap_bound,
            }
        )
    return rows


def local_variant_name(row: dict) -> str:
    if not row:
        return ""
    if row.get("overlap_cpu_optimizer_d2h_h2d") != "1":
        return "no-overlap"
    if row.get("cpu_offload_slab_copy") == "1":
        return "overlap+group50m+slab"
    if row.get("cpu_offload_foreach_copy") == "1":
        return "overlap+group50m+foreach"
    if row.get("cpu_offload_overlap_group_numel"):
        return "overlap+group50m"
    return "overlap"


def query_timeline_events(sqlite_path: Path, max_seconds: float = 0.35) -> tuple[list[dict], float]:
    conn = sqlite3.connect(sqlite_path)
    try:
        start_values = []
        for table in ["CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY"]:
            value = conn.execute(f"SELECT MIN(start) FROM {table}").fetchone()[0]
            if value is not None:
                start_values.append(value)
        if not start_values:
            return [], 0.0
        base = min(start_values)
        window_end = base + int(max_seconds * 1e9)
        events = []
        for start, end in conn.execute(
            """
            SELECT start, end
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE end >= start AND start <= ?
            ORDER BY start
            """,
            (window_end,),
        ):
            events.append({"kind": "kernel", "start_s": (start - base) / 1e9, "duration_s": (end - start) / 1e9})
        for start, end, kind in conn.execute(
            """
            SELECT m.start, m.end, COALESCE(e.label, CAST(m.copyKind AS TEXT))
            FROM CUPTI_ACTIVITY_KIND_MEMCPY m
            LEFT JOIN ENUM_CUDA_MEMCPY_OPER e ON m.copyKind = e.id
            WHERE m.end >= m.start AND m.start <= ?
            ORDER BY m.start
            """,
            (window_end,),
        ):
            label = "memcpy"
            if kind == "Host-to-Device":
                label = "h2d"
            elif kind == "Device-to-Host":
                label = "d2h"
            events.append({"kind": label, "start_s": (start - base) / 1e9, "duration_s": (end - start) / 1e9})
        return events, max_seconds
    finally:
        conn.close()


def generate_figures(profile_rows: list[dict]) -> list[str]:
    if not profile_rows:
        return []
    os.environ.setdefault("MPLCONFIGDIR", str(STUDY_DIR / ".mplconfig"))
    (STUDY_DIR / ".mplconfig").mkdir(exist_ok=True)
    import matplotlib.pyplot as plt  # noqa: E402

    figure_paths = []
    figure_dir = STUDY_DIR / "figures"
    figure_dir.mkdir(exist_ok=True)

    labels = [row["run_id"].replace("offload_3b4l_cola_distopt_offload_", "") for row in profile_rows]
    sync_ms = [(to_float(row, "total_cuda_sync_api_us") or 0) / 1000 for row in profile_rows]
    launch_ms = [(to_float(row, "total_cuda_launch_api_us") or 0) / 1000 for row in profile_rows]
    idle_frac = [to_float(row, "idle_gap_fraction") or 0 for row in profile_rows]
    h2d_count = [to_float(row, "h2d_memcpy_count") or 0 for row in profile_rows]
    d2h_count = [to_float(row, "d2h_memcpy_count") or 0 for row in profile_rows]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    x = range(len(profile_rows))
    axes[0].bar(x, sync_ms, label="sync API ms", color="#b54a4a")
    axes[0].bar(x, launch_ms, bottom=sync_ms, label="launch API ms", color="#4d7fb8")
    axes[0].set_ylabel("CUDA API time (ms)")
    axes[0].set_xticks(list(x), labels, rotation=20, ha="right")
    axes[0].legend()
    axes[1].bar(x, h2d_count, label="H2D memcpy count", color="#4a9b73")
    axes[1].bar(x, d2h_count, bottom=h2d_count, label="D2H memcpy count", color="#d39b38")
    axes[1].plot(list(x), [item * 1000 for item in idle_frac], marker="o", color="#333333", label="idle gap fraction x1000")
    axes[1].set_ylabel("Count / scaled fraction")
    axes[1].set_xticks(list(x), labels, rotation=20, ha="right")
    axes[1].legend()
    out_png = figure_dir / "nsys_offload_profile_summary.png"
    out_pdf = figure_dir / "nsys_offload_profile_summary.pdf"
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_pdf)
    plt.close(fig)
    figure_paths.extend([str(out_png), str(out_pdf)])

    timeline_rows = profile_rows[:3]
    if timeline_rows:
        lane_names = []
        fig, ax = plt.subplots(figsize=(12, 0.9 + 1.15 * len(timeline_rows)), constrained_layout=True)
        colors = {"kernel": "#4d7fb8", "h2d": "#4a9b73", "d2h": "#d39b38", "memcpy": "#8a8a8a"}
        for run_idx, row in enumerate(timeline_rows):
            events, _window = query_timeline_events(Path(row["sqlite_file"]))
            base_y = run_idx * 4
            lane_names.extend([row["run_id"].replace("offload_3b4l_cola_distopt_offload_", "")])
            for offset, kind in enumerate(["kernel", "h2d", "d2h"]):
                segments = [
                    (event["start_s"], max(event["duration_s"], 1e-6))
                    for event in events
                    if event["kind"] == kind
                ]
                if segments:
                    ax.broken_barh(segments, (base_y + offset, 0.75), facecolors=colors[kind], alpha=0.75)
            ax.text(-0.006, base_y + 1.1, row["run_id"].replace("offload_3b4l_cola_distopt_offload_", ""), ha="right", va="center", fontsize=8)
        ax.set_xlabel("Seconds from first captured GPU event")
        ax.set_yticks([])
        ax.set_xlim(left=0)
        ax.set_title("Nsight event timeline sample: kernels, H2D, D2H")
        out_png = figure_dir / "offload_overlap_timeline.png"
        out_pdf = figure_dir / "offload_overlap_timeline.pdf"
        fig.savefig(out_png, dpi=180)
        fig.savefig(out_pdf)
        plt.close(fig)
        figure_paths.extend([str(out_png), str(out_pdf)])
    return figure_paths


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "No rows yet.\n"
    header = "| " + " | ".join(title for title, _key in columns) + " |"
    sep = "| " + " | ".join("---" for _title, _key in columns) + " |"
    body = []
    for row in rows:
        values = []
        for _title, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append(str(value) if value is not None else "")
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def to_float(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def baseline_comparison_rows(rows: list[dict]) -> list[dict]:
    by_id = {row.get("run_id"): row for row in rows}
    baseline = by_id.get("throughput_3b_cola_distopt_offload_baseline", {})
    base_iter = to_float(baseline, "avg_iteration_ms")
    base_opt = to_float(baseline, "optimizer_total_ms")
    out = []
    for run_id in [
        "throughput_3b_cola_distopt_baseline",
        "throughput_3b_cola_distopt_offload_baseline",
        "throughput_3b_cola_distopt_offload_overlap_d2h_h2d",
        "throughput_3b_cola_distopt_offload_overlap_group50m",
        "throughput_3b_cola_distopt_offload_fraction_0p5",
    ]:
        row = by_id.get(run_id, {})
        if not row:
            continue
        iter_ms = to_float(row, "avg_iteration_ms")
        opt_ms = to_float(row, "optimizer_total_ms")
        out.append(
            {
                "run_id": run_id,
                "variant": variant_name(row),
                "iter_ms": iter_ms,
                "iter_delta_vs_full_offload_ms": iter_ms - base_iter
                if iter_ms is not None and base_iter is not None
                else None,
                "fwd_bwd_ms": to_float(row, "forward_backward_ms"),
                "grad_sync_ms": to_float(row, "grad_sync_ms"),
                "all_gather_ms": to_float(row, "param_all_gather_ms"),
                "optimizer_total_ms": opt_ms,
                "optimizer_delta_vs_full_offload_ms": opt_ms - base_opt
                if opt_ms is not None and base_opt is not None
                else None,
                "optimizer_inner_ms": to_float(row, "optimizer_inner_ms"),
                "max_allocated_mb": to_float(row, "max_allocated_mb"),
                "reserved_mb": to_float(row, "reserved_mb"),
            }
        )
    return out


def optimization_result_rows(rows: list[dict]) -> list[dict]:
    interesting = [
        "throughput_3b_cola_distopt_offload_baseline",
        "throughput_3b_cola_distopt_offload_overlap_d2h_h2d",
        "throughput_3b_cola_distopt_offload_overlap_group50m",
        "offload_3b_cola_distopt_offload_overlap_group50m_foreach",
        "offload_3b_cola_distopt_offload_overlap_group50m_slab",
        "throughput_3b_cola_distopt_offload_fraction_0p5",
    ]
    by_id = {row.get("run_id"): row for row in rows}
    baseline_iter = to_float(by_id.get("throughput_3b_cola_distopt_offload_baseline", {}), "avg_iteration_ms")
    output = []
    for run_id in interesting:
        row = by_id.get(run_id)
        if not row:
            continue
        iter_ms = to_float(row, "avg_iteration_ms")
        label = variant_name(row)
        if run_id.endswith("_foreach"):
            label = "overlap + group50m + foreach copy"
        if run_id.endswith("_slab"):
            label = "overlap + group50m + slab copy"
        output.append(
            {
                "variant": label,
                "run_id": run_id,
                "iter_ms": iter_ms,
                "delta_vs_full_offload_ms": iter_ms - baseline_iter
                if iter_ms is not None and baseline_iter is not None
                else None,
                "forward_backward_ms": to_float(row, "forward_backward_ms"),
                "optimizer_total_ms": to_float(row, "optimizer_total_ms"),
                "optimizer_inner_ms": to_float(row, "optimizer_inner_ms"),
                "max_allocated_mb": to_float(row, "max_allocated_mb"),
                "reserved_mb": to_float(row, "reserved_mb"),
            }
        )
    return output


def variant_name(row: dict) -> str:
    if row.get("offload") != "1":
        return "no offload"
    if row.get("optimizer_offload_fraction") == "0.5":
        return "offload fraction 0.5"
    if row.get("cpu_offload_slab_copy") == "1":
        return "overlap + optimizer grouping 50000000 + slab copy"
    if row.get("cpu_offload_foreach_copy") == "1":
        return "overlap + optimizer grouping 50000000 + foreach copy"
    if row.get("overlap_cpu_optimizer_d2h_h2d") == "1" and row.get("cpu_offload_overlap_group_numel"):
        return f"overlap + optimizer grouping {row['cpu_offload_overlap_group_numel']}"
    if row.get("overlap_cpu_optimizer_d2h_h2d") == "1":
        return "overlap d2h/h2d"
    return "full offload"


def generate_report(
    timer_rows: list[dict],
    profile_rows: list[dict],
    comparison_rows: list[dict],
    optimization_rows: list[dict],
    theoretical_rows: list[dict],
    figure_paths: list[str],
) -> None:
    matrix_rows = read_csv(STUDY_DIR / "RUN_MATRIX.csv")
    report = [
        "# CoLA Optimizer CPU Offload Overlap Optimization",
        "",
        "Status: evidence complete for the offload-overlap optimization objective. The tested prototypes did not beat the no-overlap full-offload baseline; the report records why and identifies the deeper runtime change needed next.",
        "",
        "## Scope And Problem Statement",
        "",
        "This study optimizes CoLA DistOpt + optimizer CPU offload in ATC-Megatron, focusing on why the current `--overlap-cpu-optimizer-d2h-h2d` path hurts runtime and whether transfer coalescing, pinned staging reuse, and lower synchronization can recover throughput.",
        "",
        "Primary target: 3B, DP=4, seq1024, micro-batch 1, global batch 4. Nsight profiling uses a 3B-width 4-layer shape when full 3B profiling is too expensive.",
        "",
        "## Progress / Run Matrix",
        "",
        markdown_table(
            matrix_rows,
            [
                ("model", "model"),
                ("impl", "impl"),
                ("runtime", "runtime_mode"),
                ("offload", "offload_mode_fraction"),
                ("overlap", "overlap_enabled"),
                ("variant", "optimization_variant"),
                ("status", "status"),
                ("job", "job_id"),
                ("log", "log_path"),
                ("profile", "nsys_sqlite_path"),
                ("reason", "failure_skip_reason"),
            ],
        ),
        "## FSDP-Only Failure Reconciliation",
        "",
        "Measured/code-inspection finding: the current failing 0525 FSDP-only rows used `FSDP_USE_TORCH_OPTIMIZER=0`, so `MEGATRON_FSDP_USE_TORCH_OPTIMIZER` was not exported. Both FullRank and CoLA then failed inside `optimizer.step()` / `optimizer-inner-step` with CUDA illegal memory access and NCCL watchdog aborts. Earlier successful 3B FSDP-only rows from PBS job `7171062` recorded `fsdp_use_torch_optimizer=1`, `CUDA_LAUNCH_BLOCKING=1`, and `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`.",
        "",
        "Interpretation: the failure is a Megatron-FSDP-only optimizer-path mismatch, not evidence about CoLA DistOpt optimizer CPU offload. For future stable FSDP-only comparison, run with `FSDP_USE_TORCH_OPTIMIZER=1`; keep `CUDA_LAUNCH_BLOCKING=1` only for diagnosis because it can perturb timing.",
        "",
        "Evidence paths:",
        "",
        "- Failed current FullRank FSDP: `benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_fullrank_fsdp_baseline.log`, `benchmarks/0525/throughput_optimization_20260526T053247Z/meta/throughput_3b_fullrank_fsdp_baseline.json`",
        "- Failed current CoLA FSDP: `benchmarks/0525/throughput_optimization_20260526T053247Z/logs/throughput_3b_cola_fsdp_baseline.log`, `benchmarks/0525/throughput_optimization_20260526T053247Z/meta/throughput_3b_cola_fsdp_baseline.json`",
        "- Successful prior FullRank/CoLA FSDP: `/eagle/TensorCompress/zhengyangwang/atc_megatron_cola/7171062.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/meta/*_fsdp_offload0_cg0.json`",
        "",
        "## Baseline Overlap Timer Breakdown",
        "",
        markdown_table(
            comparison_rows,
            [
                ("variant", "variant"),
                ("iter ms", "iter_ms"),
                ("delta iter", "iter_delta_vs_full_offload_ms"),
                ("fwd-bwd", "fwd_bwd_ms"),
                ("grad sync", "grad_sync_ms"),
                ("all-gather", "all_gather_ms"),
                ("opt total", "optimizer_total_ms"),
                ("delta opt", "optimizer_delta_vs_full_offload_ms"),
                ("opt inner", "optimizer_inner_ms"),
                ("max alloc MB", "max_allocated_mb"),
                ("reserved MB", "reserved_mb"),
            ],
        ),
        "Imported evidence: enabling current overlap increases CoLA full-offload iteration time from 854.57 ms to 1047.52 ms. The main increase is optimizer total time (+136.07 ms) and forward-backward time (+52.30 ms), with smaller all-gather increase (+5.62 ms). Memory stays effectively flat in max allocated and rises slightly in reserved memory. CPU optimizer grouping recovers about 32.83 ms of iteration time but remains slower than no-overlap full offload.",
        "",
        "## Nsight Profile Summary",
        "",
        markdown_table(
            profile_rows,
            [
                ("run_id", "run_id"),
                ("status", "status"),
                ("kernels", "kernel_count"),
                ("avg kernel us", "avg_kernel_us"),
                ("idle frac", "idle_gap_fraction"),
                ("CUDA API us", "total_cuda_api_us"),
                ("launch APIs", "cuda_launch_api_count"),
                ("sync APIs", "cuda_sync_api_count"),
                ("H2D count", "h2d_memcpy_count"),
                ("H2D bytes", "h2d_memcpy_bytes"),
                ("D2H count", "d2h_memcpy_count"),
                ("D2H bytes", "d2h_memcpy_bytes"),
            ],
        ),
        "All planned local offload Nsight rows for no-overlap, current overlap, overlap+grouping, and slab transfer coalescing have been parsed.",
        "",
        "## Timeline Figures",
        "",
        "\n".join(f"- `{path}`" for path in figure_paths) if figure_paths else "Pending.",
        "",
        "## Theoretical Model",
        "",
        markdown_table(
            theoretical_rows,
            [
                ("variant", "variant"),
                ("H2D GB", "h2d_gb"),
                ("D2H GB", "d2h_gb"),
                ("transfer ms", "transfer_time_ms"),
                ("opt ms", "optimizer_total_ms"),
                ("opt-transfer ms", "observed_minus_transfer_ms"),
                ("sync API ms/iter", "sync_api_ms_per_iter"),
                ("CUDA API ms/iter", "cuda_api_ms_per_iter"),
                ("idle frac", "idle_gap_fraction"),
                ("perfect overlap lb ms", "ideal_perfect_overlap_lower_bound_ms"),
            ],
        ),
        "Model note: transfer and CUDA API values are normalized by captured iteration count. Negative `opt-transfer` means the rank-local memcpy durations overlap CPU/GPU work or the timer and Nsight windows are not measuring the same critical path exactly; use the table for bottleneck direction, not as a strict additive timing decomposition.",
        "",
        "## CPU Offload Code Inspection",
        "",
        "ATC-Megatron's `HybridDeviceOptimizer.step()` copies gradients from GPU to pinned CPU buffers on `_d2h_stream`, then synchronizes each CPU optimizer's D2H event immediately before calling that optimizer's `step()`. With overlap enabled, `build_cpu_optimizer_list()` defaults to one CPU optimizer per parameter; each optimizer also installs a post-step hook that copies updated CPU parameters back to GPU on `_h2d_stream` and then waits on the current stream. This creates many optimizer objects, D2H event waits, CPU optimizer calls, H2D copy-back hook invocations, and stream/event operations for fine-grained CoLA parameters.",
        "",
        "The grouping prototype is env-gated by `MEGATRON_CPU_OFFLOAD_GROUP_NUMEL` and only groups CPU optimizer instances. The foreach prototype (`MEGATRON_CPU_OFFLOAD_FOREACH_COPY=1`) batches copy calls inside each group but still leaves the same underlying tensor granularity. The slab prototype (`MEGATRON_CPU_OFFLOAD_SLAB_COPY=1`) reuses flat CPU/GPU staging slabs and coalesces grouped D2H/H2D movement; it preserves optimizer math but adds GPU pack/scatter traffic.",
        "",
        "## External Source Notes",
        "",
        "- ZeRO-Offload paper: https://arxiv.org/abs/2101.06840. Relevance: frames CPU optimizer offload as useful only when GPU data movement is minimized and CPU compute time is reduced; this matches the current need to split transfer, CPU Adam, and framework overhead.",
        "- ZeRO-Infinity paper: https://arxiv.org/abs/2104.07857. Relevance: motivates overlapping heterogeneous memory movement with compute and treating CPU/NVMe bandwidth as a first-class bottleneck.",
        "- DeepSpeed ZeRO documentation: https://deepspeed.readthedocs.io/en/stable/zero3.html. Relevance: documents optimizer/gradient offload lineage, recommends optimized CPUAdam, and identifies pinned memory/effective bandwidth as relevant offload controls.",
        "- DeepSpeed optimizer documentation: https://deepspeed.readthedocs.io/en/latest/optimizers.html. Relevance: contrasts CPUAdam and fused/multi-tensor GPU Adam; supports testing whether many small CPU optimizer calls are a CoLA granularity overhead.",
        "- PyTorch FSDP notes: https://docs.pytorch.org/docs/2.8/notes/fsdp.html. Relevance: explains all-gather prefetch/overlap depends on CPU issue order and compute window length, which is directly relevant to fine-grained CoLA FSDP unit granularity.",
        "",
        "## Optimization Attempts And Commands",
        "",
        "- Existing env-gated prototype: `MEGATRON_CPU_OFFLOAD_GROUP_NUMEL` groups CPU optimizer instances in `megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py` when overlap is enabled. Command metadata is in `benchmarks/0525/throughput_optimization_20260526T053247Z/meta/throughput_3b_cola_distopt_offload_overlap_group50m.cmd`.",
        "- New env-gated prototype: `MEGATRON_CPU_OFFLOAD_FOREACH_COPY=1` batches D2H grad-copy and H2D param-copy setup with `torch._foreach_copy_` inside each CPU optimizer group. This preserves optimizer math and is intended to reduce per-small-tensor Python/CUDA copy setup, not to coalesce transfers into slabs.",
        "- New env-gated prototype: `MEGATRON_CPU_OFFLOAD_SLAB_COPY=1` packs grouped GPU gradients into a reusable GPU slab, copies that slab to a pinned CPU slab, exposes CPU optimizer gradient views from the slab, then stages updated CPU parameters back through a GPU slab. This is the first prototype that reduces actual host-copy count.",
        "",
        "## Before / After Results",
        "",
        markdown_table(
            optimization_rows,
            [
                ("variant", "variant"),
                ("iter ms", "iter_ms"),
                ("delta vs full offload", "delta_vs_full_offload_ms"),
                ("fwd-bwd", "forward_backward_ms"),
                ("opt total", "optimizer_total_ms"),
                ("opt inner", "optimizer_inner_ms"),
                ("max alloc MB", "max_allocated_mb"),
                ("reserved MB", "reserved_mb"),
            ],
        ),
        "Measured result: current best full-offload speed remains no-overlap full optimizer CPU offload at 854.57 ms. The CPU optimizer grouping prototype improves overlap-only from 1047.52 ms to 1014.69 ms but does not beat baseline full offload. The foreach-copy prototype validates correctly but gives 1004.25 ms and the same optimizer total as grouping alone, so it is not accepted as a meaningful optimization. The slab-copy prototype does reduce host-copy granularity in Nsight, but full 3B iteration time regresses to 1141.29 ms and global max allocation rises to 16069.49 MB because the staging slabs and GPU pack/scatter work become the new bottleneck.",
        "",
        "## Failure / Skipped Rows",
        "",
        "No local 0526 rows failed. The slab-copy full 3B row is marked done, not successful, because it produced valid finite-loss timing evidence but regressed throughput and memory.",
        "",
        "## Final Diagnosis",
        "",
        "Measured diagnosis: current overlap hurts because it does not reduce or coalesce real H2D/D2H movement, while it increases synchronization/API overhead and GPU idle gaps. In the 4-layer Nsight rows, no-overlap and overlap both move about 4.8 GB H2D and 4.8 GB D2H across the capture, with nearly identical copy counts. Overlap increases rank-local CUDA sync API time from 1.15 s to 1.30 s and idle-gap fraction from 0.492 to 0.531; grouping does not change copy counts/bytes and can increase sync/API time. The 3B timer evidence matches this: overlap is slower than full offload, grouping only partly recovers it, and foreach-copy batching is neutral.",
        "",
        "The slab prototype confirms the right direction but not a usable implementation: H2D copies fall from 426 to 282 and D2H copies from 366 to 222 versus overlap+group50m in the 4-layer profile, while H2D/D2H byte volume stays constant. However, D2D bytes rise from 0.71 GB to 3.05 GB because the prototype packs and scatters through GPU staging buffers, and the full 3B row slows to 1141.29 ms. Coalescing host copies alone is insufficient when it is implemented by adding equivalent or larger device-side staging work.",
        "",
        "Interpretation: CoLA has worse offload-overlap opportunity than FullRank because its compute window is shorter while offload traffic and CPU optimizer work remain large enough to dominate the step. The strongest supporting metrics are optimizer total time, Nsight H2D/D2H copy counts and bytes, CUDA sync API time, idle-gap fraction, and slab-induced D2D byte growth.",
        "",
        "## Final Answers",
        "",
        "- Why does current overlap hurt? Measured: it keeps the same H2D/D2H bytes and nearly the same copy counts, then adds stream/event waits, CUDA API time, and GPU idle gaps. The 3B row slows from 854.57 ms to 1047.52 ms.",
        "- Which overhead increases? Measured 3B timers show optimizer total grows by 136.07 ms, forward-backward by 52.30 ms, and all-gather by 5.62 ms versus no-overlap full offload. Nsight shows sync/API and idle-gap increases on the reduced profile.",
        "- What is the theoretical best speedup from perfect hiding? For the 3B full-offload row, fwd-bwd + grad sync + all-gather is about 219.41 ms. If optimizer/offload work were perfectly hidden, the coarse upper-bound speedup from 854.57 ms is about 3.89x. The 4-layer Nsight lower bound is less optimistic because rank-local H2D+D2H memcpy duration is about 251 ms/iter, already much larger than its 61 ms compute window.",
        "- Does CoLA give better or worse offload-overlap opportunity than FullRank? Worse. CoLA reduces GPU compute time, leaving less compute window to hide CPU optimizer/offload traffic. The imported FullRank control is slower overall, but its longer compute window gives more room for overlap than CoLA.",
        "- Is offload limited by bandwidth, CPU optimizer computation, synchronization, or framework overhead? Measured evidence points to transfer plus framework synchronization/API overhead, not pure bandwidth alone. No-overlap profile optimizer time is close to rank-local memcpy duration, while current overlap increases sync/API and idle gaps without reducing traffic. CPU Adam work is still present, but the decisive regression is orchestration and staging overhead.",
        "- Is the current slab optimization successful? No. It proves actual host-copy coalescing is possible, but the naive pack/scatter implementation shifts cost to D2D staging and raises memory, so full 3B throughput regresses.",
        "- Which existing knob improves performance? `optimizer_offload_fraction=0.5` is the strongest measured speed-memory Pareto point: 561.07 ms versus 854.57 ms full offload, with max allocated rising from 13464.80 MB to 15435.72 MB. `MEGATRON_CPU_OFFLOAD_GROUP_NUMEL=50000000` improves the overlap variant but remains slower than no-overlap full offload.",
        "- Which new optimization is most promising? A deeper flattened-layout/slab design that avoids separate GPU pack/scatter copies: keep gradients/parameters in persistent contiguous layouts or use fused pack/scatter kernels so host transfers are coalesced without adding multi-GB D2D staging overhead.",
        "- Which result most strongly supports the granularity-aware memory-management thesis? The strongest evidence is the combination of unchanged H2D/D2H traffic under current overlap, increased CUDA sync/API and idle-gap metrics, and the slab prototype reducing host copy count but regressing from added staging overhead. CoLA's finer-grained runtime events prevent memory reduction from becoming throughput speedup.",
        "",
        "## Next Recommended Optimization",
        "",
        "Prototype a persistent flattened offload layout for CoLA factorized parameters so CPU optimizer state, pinned host gradients, and GPU parameter/gradient views share bucketed storage from initialization. That should preserve the copy-count reduction demonstrated by `MEGATRON_CPU_OFFLOAD_SLAB_COPY=1` while removing the extra D2D pack/scatter step that made the naive slab prototype slower.",
        "",
        "## Raw Artifacts",
        "",
        "- Imported timer CSV/JSON: `parsed/offload_overlap_timers.csv`, `parsed/offload_overlap_timers.json`",
        "- Baseline comparison CSV/JSON: `parsed/offload_overlap_comparison.csv`, `parsed/offload_overlap_comparison.json`",
        "- Local logs: `logs/*.log`",
        "- Local metadata and commands: `meta/*.json`, `meta/*.cmd`",
        "- Local Nsight profiles: `profiles/*.nsys-rep`, `profiles/*.sqlite`",
        "- Local parsed Nsight: `parsed/nsys_offload_overlap.csv`, `parsed/nsys_offload_overlap.json`",
        "- Theoretical model: `parsed/theoretical_model.csv`, `parsed/theoretical_model.json`",
        "- Figures: `figures/*.png`, `figures/*.pdf`",
        "",
    ]
    (STUDY_DIR / "REPORT.md").write_text("\n".join(report) + "\n")


def main() -> int:
    timer_rows = imported_baseline_rows() + local_timer_rows()
    profile_rows = nsys_rows()
    comparison_rows = baseline_comparison_rows(timer_rows)
    optimization_rows = optimization_result_rows(timer_rows)
    theoretical_rows = build_theoretical_rows(timer_rows, profile_rows)
    figure_paths = generate_figures(profile_rows)

    write_csv(STUDY_DIR / "parsed" / "offload_overlap_timers.csv", timer_rows)
    write_json(STUDY_DIR / "parsed" / "offload_overlap_timers.json", timer_rows)
    write_csv(STUDY_DIR / "parsed" / "nsys_offload_overlap.csv", profile_rows)
    write_json(STUDY_DIR / "parsed" / "nsys_offload_overlap.json", profile_rows)
    write_csv(STUDY_DIR / "parsed" / "offload_overlap_comparison.csv", comparison_rows)
    write_json(STUDY_DIR / "parsed" / "offload_overlap_comparison.json", comparison_rows)
    write_csv(STUDY_DIR / "parsed" / "theoretical_model.csv", theoretical_rows)
    write_json(STUDY_DIR / "parsed" / "theoretical_model.json", theoretical_rows)
    generate_report(
        timer_rows, profile_rows, comparison_rows, optimization_rows, theoretical_rows, figure_paths
    )
    print(f"Wrote parsed artifacts under {STUDY_DIR / 'parsed'}")
    print(f"Updated {STUDY_DIR / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
