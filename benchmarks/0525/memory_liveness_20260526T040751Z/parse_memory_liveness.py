#!/usr/bin/env python3
"""Parse CUDA memory snapshots for the 2026-05-25 CoLA liveness study."""

import argparse
import csv
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


BYTES_PER_GB = 1024**3
PHASE_ORDER = {
    "post_init": 0,
    "post_backward": 1,
    "post_optimizer_step": 2,
    "steady_state_after_iter_10": 3,
}
MODEL_LABELS = {"baseline": "FullRank", "cola": "CoLA"}
RUNTIME_LABELS = {
    ("distopt", "0"): "DistOpt",
    ("distopt", "1"): "DistOpt+offload",
    ("fsdp", "0"): "FSDP",
    ("fsdp", "1"): "FSDP+offload",
}


def gb(value: Optional[Union[int, float]]) -> Optional[float]:
    return None if value is None else value / BYTES_PER_GB


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def stats_value(stats: Dict[str, Any], key: str) -> Optional[int]:
    value = stats.get("memory_stats", {}).get(key)
    if value is None:
        return None
    return int(value)


def block_state(block: Dict[str, Any]) -> str:
    state = str(block.get("state", ""))
    if state:
        return state
    if block.get("allocated"):
        return "active_allocated"
    return "inactive"


def block_size(block: Dict[str, Any]) -> int:
    for key in ("requested_size", "size"):
        value = block.get(key)
        if value is not None:
            return int(value)
    return 0


def block_storage_size(block: Dict[str, Any]) -> int:
    return int(block.get("size") or block_size(block))


def iter_blocks(snapshot: Any):
    if isinstance(snapshot, dict):
        segments = snapshot.get("segments", [])
    else:
        segments = snapshot
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        for block in segment.get("blocks", []) or []:
            if isinstance(block, dict):
                yield segment, block


def stack_key(block: Dict[str, Any]) -> str:
    frames = block.get("frames") or block.get("history") or []
    if isinstance(frames, dict):
        frames = frames.get("frames", [])
    parts = []
    for frame in frames[:8]:
        if not isinstance(frame, dict):
            continue
        filename = frame.get("filename") or frame.get("file") or "?"
        line = frame.get("line") or frame.get("lineno") or "?"
        name = frame.get("name") or frame.get("func") or "?"
        parts.append(f"{filename}:{line}:{name}")
    return " | ".join(parts) if parts else "unavailable"


def hist_bin(size: int) -> str:
    if size <= 0:
        return "0"
    exp = int(math.floor(math.log2(size)))
    lo = 2**exp
    hi = 2 ** (exp + 1)
    return f"[{lo},{hi})"


def parse_snapshot(
    stats_path: Path, meta_by_run: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    stats = read_json(stats_path)
    snapshot_path = Path(stats["snapshot_path"])
    run_id = stats_path.parent.name
    meta = meta_by_run.get(run_id, {})
    model_impl = MODEL_LABELS.get(meta.get("model_impl"), meta.get("model_impl", "unknown"))
    runtime_mode = RUNTIME_LABELS.get((meta.get("strategy"), str(meta.get("offload"))), "unknown")

    active_snapshot = 0
    inactive_split_snapshot = 0
    reserved_inactive_snapshot = 0
    largest_free_block = None
    active_blocks = 0
    inactive_split_blocks = 0
    active_hist = Counter()
    inactive_split_hist = Counter()
    active_stacks = defaultdict(int)
    inactive_split_stacks = defaultdict(int)

    snapshot_status = "missing"
    if snapshot_path.exists() and stats.get("status") == "ok":
        snapshot_status = "ok"
        with snapshot_path.open("rb") as f:
            snapshot = pickle.load(f)
        for _segment, block in iter_blocks(snapshot):
            state = block_state(block)
            size = block_size(block)
            storage_size = block_storage_size(block)
            if state == "active_allocated":
                active_snapshot += size
                active_blocks += 1
                active_hist[hist_bin(size)] += 1
                active_stacks[stack_key(block)] += size
            else:
                reserved_inactive_snapshot += storage_size
                largest_free_block = max(largest_free_block or 0, storage_size)
                if state == "inactive_split":
                    inactive_split_snapshot += storage_size
                    inactive_split_blocks += 1
                    inactive_split_hist[hist_bin(storage_size)] += 1
                    inactive_split_stacks[stack_key(block)] += storage_size
    else:
        snapshot_status = stats.get("status", "missing")

    allocated = int(stats.get("memory_allocated") or stats_value(stats, "allocated_bytes.all.current") or active_snapshot)
    reserved = int(stats.get("memory_reserved") or stats_value(stats, "reserved_bytes.all.current") or 0)
    max_allocated = int(stats.get("max_memory_allocated") or stats_value(stats, "allocated_bytes.all.peak") or 0)
    max_reserved = int(stats.get("max_memory_reserved") or stats_value(stats, "reserved_bytes.all.peak") or 0)
    active_allocated = int(stats_value(stats, "active_bytes.all.current") or active_snapshot or allocated)
    inactive_split = int(stats_value(stats, "inactive_split_bytes.all.current") or inactive_split_snapshot)
    reserved_inactive = reserved_inactive_snapshot if reserved_inactive_snapshot else max(reserved - allocated, 0)

    row = {
        "run_id": run_id,
        "model_impl": model_impl,
        "runtime_mode": runtime_mode,
        "snapshot_phase": stats.get("phase"),
        "iteration": stats.get("iteration"),
        "rank": stats.get("rank"),
        "snapshot_status": snapshot_status,
        "active_allocated_bytes": active_allocated,
        "inactive_split_bytes": inactive_split,
        "reserved_inactive_bytes": reserved_inactive,
        "reserved_bytes": reserved,
        "allocated_bytes": allocated,
        "reserved_allocated_gap_bytes": max(reserved - allocated, 0),
        "max_reserved_max_allocated_gap_bytes": max(max_reserved - max_allocated, 0) if max_reserved and max_allocated else None,
        "largest_free_block_bytes": largest_free_block,
        "active_block_count": active_blocks,
        "inactive_split_block_count": inactive_split_blocks,
        "active_allocated_gb": gb(active_allocated),
        "inactive_split_gb": gb(inactive_split),
        "reserved_inactive_gb": gb(reserved_inactive),
        "reserved_gb": gb(reserved),
        "allocated_gb": gb(allocated),
        "reserved_allocated_gap_gb": gb(max(reserved - allocated, 0)),
        "max_reserved_max_allocated_gap_gb": gb(max(max_reserved - max_allocated, 0)) if max_reserved and max_allocated else None,
        "largest_free_block_gb": gb(largest_free_block),
        "stats_path": str(stats_path),
        "snapshot_path": str(snapshot_path),
    }

    active_hist_rows = [
        {**row_key(row), "size_bin_bytes": size_bin, "block_count": count, "block_state": "active_allocated"}
        for size_bin, count in sorted(active_hist.items())
    ]
    inactive_hist_rows = [
        {**row_key(row), "size_bin_bytes": size_bin, "block_count": count, "block_state": "inactive_split"}
        for size_bin, count in sorted(inactive_split_hist.items())
    ]
    stack_rows = []
    for state, stacks in (("active_allocated", active_stacks), ("inactive_split", inactive_split_stacks)):
        for stack, size in sorted(stacks.items(), key=lambda item: item[1], reverse=True)[:20]:
            stack_rows.append({**row_key(row), "block_state": state, "bytes": size, "gb": gb(size), "stack": stack})
    return row, active_hist_rows, inactive_hist_rows, stack_rows


def row_key(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "model_impl": row["model_impl"],
        "runtime_mode": row["runtime_mode"],
        "snapshot_phase": row["snapshot_phase"],
        "rank": row["rank"],
    }


def aggregate_rank0(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rank0 = [row for row in rows if row.get("rank") == 0]
    return rank0 if rank0 else rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


def build_comparison(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key = {(row["runtime_mode"], row["snapshot_phase"], row["model_impl"]): row for row in aggregate_rank0(rows)}
    out = []
    for runtime in ("DistOpt", "DistOpt+offload", "FSDP", "FSDP+offload"):
        for phase in sorted(PHASE_ORDER, key=PHASE_ORDER.get):
            full = by_key.get((runtime, phase, "FullRank"))
            cola = by_key.get((runtime, phase, "CoLA"))
            if not full or not cola:
                continue
            inactive_delta = cola["inactive_split_gb"] - full["inactive_split_gb"]
            conclusion = "unclear"
            if abs(inactive_delta) < 0.25:
                conclusion = "no"
            elif inactive_delta > 0.25:
                conclusion = "yes"
            else:
                conclusion = "no"
            out.append(
                {
                    "runtime_mode": runtime,
                    "snapshot_phase": phase,
                    "active_allocated_reduction_gb": full["active_allocated_gb"] - cola["active_allocated_gb"],
                    "inactive_split_change_gb": inactive_delta,
                    "reserved_inactive_change_gb": cola["reserved_inactive_gb"] - full["reserved_inactive_gb"],
                    "reserved_allocated_gap_change_gb": cola["reserved_allocated_gap_gb"] - full["reserved_allocated_gap_gb"],
                    "active_block_count_change": cola["active_block_count"] - full["active_block_count"],
                    "conclusion_cola_increases_fragmentation": conclusion,
                }
            )
    return out


def likely_runtime_cause(base: Dict[str, Any], row: Dict[str, Any]) -> str:
    inactive_delta = row["inactive_split_gb"] - base["inactive_split_gb"]
    reserved_delta = row["reserved_inactive_gb"] - base["reserved_inactive_gb"]
    active_delta = row["active_allocated_gb"] - base["active_allocated_gb"]
    if inactive_delta > 0.25:
        return "fragmentation"
    if active_delta > 0.5:
        return "staging buffers or framework overhead"
    if reserved_delta > 0.5:
        return "allocator cache/headroom"
    return "unclear"


def build_runtime_overhead(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = [row for row in aggregate_rank0(rows) if row["model_impl"] == "CoLA"]
    by_key = {(row["runtime_mode"], row["snapshot_phase"]): row for row in selected}
    out = []
    for phase in sorted(PHASE_ORDER, key=PHASE_ORDER.get):
        base = by_key.get(("DistOpt", phase))
        if not base:
            continue
        for runtime in ("DistOpt", "DistOpt+offload", "FSDP", "FSDP+offload"):
            row = by_key.get((runtime, phase))
            if not row:
                continue
            out.append(
                {
                    "runtime_mode": runtime,
                    "snapshot_phase": phase,
                    "active_allocated_delta_vs_cola_distopt_gb": row["active_allocated_gb"] - base["active_allocated_gb"],
                    "inactive_split_delta_gb": row["inactive_split_gb"] - base["inactive_split_gb"],
                    "reserved_inactive_delta_gb": row["reserved_inactive_gb"] - base["reserved_inactive_gb"],
                    "reserved_allocated_gap_delta_gb": row["reserved_allocated_gap_gb"] - base["reserved_allocated_gap_gb"],
                    "likely_cause": "baseline" if runtime == "DistOpt" else likely_runtime_cause(base, row),
                }
            )
    return out


def generate_figures(
    result_dir: Path,
    rows: List[Dict[str, Any]],
    active_hist: List[Dict[str, Any]],
    inactive_hist: List[Dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    figures = result_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_rows = [row for row in aggregate_rank0(rows) if row["snapshot_phase"] == "steady_state_after_iter_10"]
    labels = [f"{row['model_impl']}\n{row['runtime_mode']}" for row in plot_rows]
    x = list(range(len(plot_rows)))

    plt.figure(figsize=(12, 5))
    active = [row["active_allocated_gb"] for row in plot_rows]
    inactive = [row["inactive_split_gb"] for row in plot_rows]
    reserved_inactive = [row["reserved_inactive_gb"] for row in plot_rows]
    plt.bar(x, active, label="active allocated")
    plt.bar(x, inactive, bottom=active, label="inactive split")
    plt.bar(x, reserved_inactive, bottom=[a + b for a, b in zip(active, inactive)], label="reserved inactive")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("GB")
    plt.title("Steady-state CUDA memory components")
    plt.legend()
    plt.tight_layout()
    for suffix in ("png", "pdf"):
        plt.savefig(figures / f"steady_state_stacked_memory.{suffix}")
    plt.close()

    phase_rows = aggregate_rank0(rows)
    plt.figure(figsize=(12, 5))
    for model in ("FullRank", "CoLA"):
        for runtime in ("DistOpt", "DistOpt+offload", "FSDP", "FSDP+offload"):
            series = [row for row in phase_rows if row["model_impl"] == model and row["runtime_mode"] == runtime]
            series.sort(key=lambda row: PHASE_ORDER.get(row["snapshot_phase"], 99))
            if series:
                plt.plot(
                    [row["snapshot_phase"] for row in series],
                    [row["active_allocated_gb"] for row in series],
                    marker="o",
                    label=f"{model} {runtime}",
                )
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Active allocated GB")
    plt.title("Memory liveness across snapshot phases")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    for suffix in ("png", "pdf"):
        plt.savefig(figures / f"phase_active_liveness.{suffix}")
    plt.close()

    for rows_for_hist, state_name, filename in (
        (active_hist, "active_allocated", "distopt_active_allocation_histogram"),
        (inactive_hist, "inactive_split", "distopt_inactive_split_histogram"),
    ):
        hist_rows = [
            row
            for row in rows_for_hist
            if row["runtime_mode"] == "DistOpt"
            and row["snapshot_phase"] == "steady_state_after_iter_10"
            and row["rank"] == 0
        ]
        if not hist_rows:
            plt.figure(figsize=(10, 4))
            plt.axis("off")
            plt.text(
                0.02,
                0.55,
                f"No {state_name} block-size records were present in the parsed snapshot.\n"
                "The aggregate bytes metric remains available from torch.cuda.memory_stats(), "
                "but a size histogram is unavailable for this row.",
                fontsize=11,
                va="center",
            )
            plt.title(f"DistOpt {state_name} allocation size histogram")
            plt.tight_layout()
            for suffix in ("png", "pdf"):
                plt.savefig(figures / f"{filename}.{suffix}")
            plt.close()
            continue
        bins = sorted({row["size_bin_bytes"] for row in hist_rows})
        plt.figure(figsize=(12, 5))
        for model in ("FullRank", "CoLA"):
            counts = [
                sum(row["block_count"] for row in hist_rows if row["model_impl"] == model and row["size_bin_bytes"] == bin_name)
                for bin_name in bins
            ]
            plt.plot(bins, counts, marker="o", label=model)
        plt.xticks(rotation=60, ha="right")
        plt.ylabel("Block count")
        plt.title(f"DistOpt {state_name} allocation size histogram")
        plt.legend()
        plt.tight_layout()
        for suffix in ("png", "pdf"):
            plt.savefig(figures / f"{filename}.{suffix}")
        plt.close()

    cola_rows = [row for row in plot_rows if row["model_impl"] == "CoLA"]
    if cola_rows:
        plt.figure(figsize=(9, 5))
        x = list(range(len(cola_rows)))
        plt.bar(x, [row["active_allocated_gb"] for row in cola_rows], label="active allocated")
        plt.bar(
            x,
            [row["inactive_split_gb"] for row in cola_rows],
            bottom=[row["active_allocated_gb"] for row in cola_rows],
            label="inactive split",
        )
        plt.xticks(x, [row["runtime_mode"] for row in cola_rows], rotation=25, ha="right")
        plt.ylabel("GB")
        plt.title("CoLA runtime memory comparison")
        plt.legend()
        plt.tight_layout()
        for suffix in ("png", "pdf"):
            plt.savefig(figures / f"cola_runtime_comparison.{suffix}")
        plt.close()


def write_report(
    result_dir: Path,
    rows: List[Dict[str, Any]],
    comparison: List[Dict[str, Any]],
    runtime: List[Dict[str, Any]],
) -> None:
    report = result_dir / "REPORT.md"
    matrix = result_dir / "RUN_MATRIX.csv"
    lines = [
        "# CoLA Memory Liveness Report",
        "",
        "## Run Matrix",
        "",
        f"See `{matrix}` for status, commands, logs, snapshots, and parsed metrics.",
        "",
        "## Memory Liveness Table",
        "",
        "| model impl | runtime mode | snapshot phase | active allocated GB | inactive split GB | reserved inactive GB | reserved GB | allocated GB | reserved-allocated gap GB | max reserved-max allocated gap GB | largest free block GB | active block count | inactive split block count |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rank0(sorted(rows, key=lambda r: (r["runtime_mode"], r["model_impl"], PHASE_ORDER.get(r["snapshot_phase"], 99)))):
        max_gap = row["max_reserved_max_allocated_gap_gb"]
        largest_free = row["largest_free_block_gb"]
        lines.append(
            f"| {row['model_impl']} | {row['runtime_mode']} | {row['snapshot_phase']} | "
            f"{row['active_allocated_gb']:.3f} | {row['inactive_split_gb']:.3f} | {row['reserved_inactive_gb']:.3f} | "
            f"{row['reserved_gb']:.3f} | {row['allocated_gb']:.3f} | {row['reserved_allocated_gap_gb']:.3f} | "
            f"{'' if max_gap is None else f'{max_gap:.3f}'} | "
            f"{'' if largest_free is None else f'{largest_free:.3f}'} | "
            f"{row['active_block_count']} | {row['inactive_split_block_count']} |"
        )
    lines += [
        "",
        "## FullRank Vs CoLA",
        "",
        "| runtime mode | snapshot phase | active allocated reduction GB | inactive split change GB | reserved inactive change GB | reserved-allocated gap change GB | active block count change | CoLA increases fragmentation? |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['runtime_mode']} | {row['snapshot_phase']} | {row['active_allocated_reduction_gb']:.3f} | "
            f"{row['inactive_split_change_gb']:.3f} | {row['reserved_inactive_change_gb']:.3f} | "
            f"{row['reserved_allocated_gap_change_gb']:.3f} | {row['active_block_count_change']} | "
            f"{row['conclusion_cola_increases_fragmentation']} |"
        )
    lines += [
        "",
        "## CoLA Runtime-System Overhead",
        "",
        "| runtime mode | snapshot phase | active allocated delta vs CoLA DistOpt GB | inactive split delta GB | reserved inactive delta GB | reserved-allocated gap delta GB | likely cause |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in runtime:
        lines.append(
            f"| {row['runtime_mode']} | {row['snapshot_phase']} | {row['active_allocated_delta_vs_cola_distopt_gb']:.3f} | "
            f"{row['inactive_split_delta_gb']:.3f} | {row['reserved_inactive_delta_gb']:.3f} | "
            f"{row['reserved_allocated_gap_delta_gb']:.3f} | {row['likely_cause']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `inactive_split_bytes` is the primary allocator-fragmentation proxy.",
        "- `reserved - allocated` and `reserved inactive` are treated as allocator cache/headroom unless inactive split blocks or stack traces indicate fragmentation.",
        "- FSDP/offload conclusions should be tied to active live bytes and stack traces when available, because those systems can create real staging/framework buffers.",
        "",
        "Final conclusions are updated after every requested row is done, failed, skipped, or unsupported with evidence.",
        "",
        "## Raw Artifacts",
        "",
        "- Raw snapshots: `raw/<run_id>/*.pickle`",
        "- Per-phase stats: `raw/<run_id>/*.stats.json`",
        "- Parsed tables: `parsed/*.csv` and `parsed/*.json`",
        "- Figures: `figures/*.png` and `figures/*.pdf`",
    ]
    report.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default=Path(__file__).resolve().parent)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    meta_by_run = {path.stem: read_json(path) for path in (result_dir / "meta").glob("*.json")}
    rows = []
    active_hist = []
    inactive_hist = []
    stacks = []
    for stats_path in sorted((result_dir / "raw").glob("*/*.stats.json")):
        row, active_rows, inactive_rows, stack_rows = parse_snapshot(stats_path, meta_by_run)
        rows.append(row)
        active_hist.extend(active_rows)
        inactive_hist.extend(inactive_rows)
        stacks.extend(stack_rows)

    parsed = result_dir / "parsed"
    write_csv(parsed / "memory_liveness.csv", rows)
    write_json(parsed / "memory_liveness.json", rows)
    write_csv(parsed / "active_allocation_histogram.csv", active_hist)
    write_csv(parsed / "inactive_split_histogram.csv", inactive_hist)
    write_csv(parsed / "top_allocation_stacks.csv", stacks)
    comparison = build_comparison(rows)
    runtime = build_runtime_overhead(rows)
    write_csv(parsed / "fullrank_vs_cola.csv", comparison)
    write_csv(parsed / "cola_runtime_overhead.csv", runtime)
    if rows and not args.no_figures:
        generate_figures(result_dir, rows, active_hist, inactive_hist)
    write_report(result_dir, rows, comparison, runtime)
    print(f"Wrote parsed artifacts under {parsed}")
    print(f"Wrote report {result_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
