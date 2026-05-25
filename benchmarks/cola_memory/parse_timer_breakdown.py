#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\):\s*([0-9.]+)"
)
MEM_RE = re.compile(
    r"allocated:\s*([0-9.]+).*?max allocated:\s*([0-9.]+).*?"
    r"reserved:\s*([0-9.]+).*?max reserved:\s*([0-9.]+).*?"
    r"global max allocated:\s*([0-9.]+).*?global max reserved:\s*([0-9.]+)"
)
TIMER_RE = re.compile(r"^\s*([A-Za-z0-9_-]+)\s+\.+:\s+\(([0-9.]+),\s*([0-9.]+)\)")
FAIL_RE = re.compile(
    r"(RUN_TIMEOUT|torch\.OutOfMemoryError|CUDA out of memory|OutOfMemoryError|"
    r"Traceback \(most recent call last\)|AssertionError|ValueError|RuntimeError|"
    r"NotImplementedError|FAILED|error:)",
    re.IGNORECASE,
)

TIMER_COLUMNS = {
    "forward-backward": "forward_backward_ms",
    "all-grads-sync": "grad_sync_ms",
    "params-all-gather": "param_all_gather_ms",
    "optimizer-copy-to-main-grad": "optimizer_copy_to_main_grad_ms",
    "optimizer-inner-step": "optimizer_inner_ms",
    "optimizer-copy-main-to-model-params": "optimizer_copy_main_to_model_ms",
    "optimizer": "optimizer_total_ms",
}


def avg(values):
    return sum(values) / len(values) if values else None


def parse_log(path: Path, warmup: int) -> dict:
    current_iter = None
    iter_times = {}
    timers_by_iter = {}
    last_mem = None
    failures = []

    text = path.read_text(errors="replace") if path.exists() else ""
    for line in text.splitlines():
        iter_match = ITER_RE.search(line)
        if iter_match:
            current_iter = int(iter_match.group(1))
            iter_times[current_iter] = float(iter_match.group(2))

        mem_match = MEM_RE.search(line)
        if mem_match:
            last_mem = tuple(float(item) for item in mem_match.groups())

        timer_match = TIMER_RE.search(line)
        if timer_match and current_iter is not None:
            name = timer_match.group(1)
            timers_by_iter.setdefault(current_iter, {})[name] = float(timer_match.group(3))

        fail_match = FAIL_RE.search(line)
        if fail_match:
            failures.append(fail_match.group(0))

    measured_iters = sorted(i for i in iter_times if i > warmup)
    row = {
        "status": "failed" if failures else ("ok" if measured_iters else "missing"),
        "failure_reason": failures[0] if failures else "",
        "num_logged_iters": len(iter_times),
        "avg_iteration_ms": avg([iter_times[i] for i in measured_iters]),
        "log_file": str(path),
    }

    for timer_name, column in TIMER_COLUMNS.items():
        row[column] = avg(
            [
                timers_by_iter.get(iteration, {}).get(timer_name)
                for iteration in measured_iters
                if timer_name in timers_by_iter.get(iteration, {})
            ]
        )

    copy_values = [
        value
        for value in (
            row.get("optimizer_copy_to_main_grad_ms"),
            row.get("optimizer_copy_main_to_model_ms"),
        )
        if value is not None
    ]
    row["optimizer_copy_ms"] = sum(copy_values) if copy_values else None

    if last_mem:
        (
            row["allocated_mb"],
            row["max_allocated_mb"],
            row["reserved_mb"],
            row["max_reserved_mb"],
            row["global_max_allocated_mb"],
            row["global_max_reserved_mb"],
        ) = last_mem

    return row


def metadata_rows(root: Path):
    for meta_path in sorted((root / "meta").glob("*.json")):
        meta = json.loads(meta_path.read_text())
        yield meta_path, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", action="append", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--warmup", type=int, default=None)
    args = parser.parse_args()

    rows = []
    for root_arg in args.results_root:
        root = Path(root_arg)
        for meta_path, meta in metadata_rows(root):
            warmup = args.warmup if args.warmup is not None else int(meta.get("warmup_iters", 5))
            log_file = Path(meta["log_file"])
            parsed = parse_log(log_file, warmup)
            rows.append({**meta, "meta_file": str(meta_path), "results_root": str(root), **parsed})

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2) + "\n")
    fieldnames = sorted({key for row in rows for key in row})
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
