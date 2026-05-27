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
    r"NotImplementedError|ChildFailedError|CUDA error|FAILED|error:)",
    re.IGNORECASE,
)

TIMER_COLUMNS = {
    "forward-backward": "fwd-bwd",
    "all-grads-sync": "grad sync",
    "params-all-gather": "param all-gather",
    "optimizer-inner-step": "opt inner",
    "optimizer-copy-to-main-grad": "opt copy to main",
    "optimizer-copy-main-to-model-params": "opt copy to model",
    "optimizer": "opt total",
}

OUTPUT_COLUMNS = [
    "row",
    "status",
    "iter ms",
    "fwd-bwd",
    "grad sync",
    "param all-gather",
    "opt inner",
    "opt copy",
    "opt total",
    "allocated",
    "max allocated",
    "reserved",
    "log",
]


def avg(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{value:.2f}"


def extract_failure_reason(text):
    if "RUN_TIMEOUT" in text:
        return "RUN_TIMEOUT"
    for line in reversed(text.splitlines()):
        if "torch.OutOfMemoryError:" in line or "CUDA out of memory" in line:
            return line.strip()
    for line in reversed(text.splitlines()):
        if "Cuda failure 2 'out of memory'" in line:
            return line.strip()
    for line in reversed(text.splitlines()):
        if any(
            token in line
            for token in (
                "RuntimeError:",
                "AssertionError:",
                "ValueError:",
                "NotImplementedError:",
                "CUDA error:",
                "ChildFailedError:",
            )
        ):
            return line.strip()
    match = FAIL_RE.search(text)
    return match.group(0) if match else ""


def parse_log(path, warmup):
    current_iter = None
    iter_times = {}
    timers_by_iter = {}
    last_mem = None
    text = path.read_text(errors="replace") if path.exists() else ""
    failures = list(FAIL_RE.finditer(text))

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
            timers_by_iter.setdefault(current_iter, {})[timer_match.group(1)] = float(
                timer_match.group(3)
            )

    measured_iters = sorted(iteration for iteration in iter_times if iteration > warmup)
    completed_training = "[after training is done]" in text
    if failures:
        reason = extract_failure_reason(text)
        status = "teardown timeout" if reason == "RUN_TIMEOUT" and measured_iters else f"failed: {reason}"
    elif measured_iters:
        status = "ok"
    else:
        status = "missing"

    parsed = {
        "status": status,
        "completed_training": completed_training,
        "num_logged_iters": len(iter_times),
        "iter ms": avg([iter_times[iteration] for iteration in measured_iters]),
        "log": str(path),
    }
    for timer_name, column in TIMER_COLUMNS.items():
        parsed[column] = avg(
            [
                timers_by_iter.get(iteration, {}).get(timer_name)
                for iteration in measured_iters
                if timer_name in timers_by_iter.get(iteration, {})
            ]
        )
    copy_parts = [parsed.get("opt copy to main"), parsed.get("opt copy to model")]
    parsed["opt copy"] = sum(part for part in copy_parts if part is not None) if any(
        part is not None for part in copy_parts
    ) else None
    if last_mem:
        allocated, max_allocated, reserved, _max_reserved, global_max_allocated, _global_max_reserved = (
            last_mem
        )
        parsed["allocated"] = allocated
        parsed["max allocated"] = global_max_allocated
        parsed["reserved"] = reserved
        parsed["rank max allocated"] = max_allocated
    else:
        parsed["allocated"] = None
        parsed["max allocated"] = None
        parsed["reserved"] = None
    return parsed


def load_rows(root):
    rows = []
    for meta_path in sorted((root / "meta").glob("*.json")):
        meta = json.loads(meta_path.read_text())
        warmup = int(meta.get("warmup_iters", 5))
        parsed = parse_log(Path(meta["log_file"]), warmup)
        row = {**meta, **parsed, "meta_file": str(meta_path)}
        rows.append(row)
    return rows


def write_outputs(rows, root):
    parsed_dir = root / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    json_path = parsed_dir / "deepspeed_wrapper_table.json"
    csv_path = parsed_dir / "deepspeed_wrapper_table.csv"
    md_path = parsed_dir / "deepspeed_wrapper_table.md"

    table_rows = [{column: row.get(column) for column in OUTPUT_COLUMNS} for row in rows]
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(table_rows)

    lines = [
        "| " + " | ".join(OUTPUT_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(OUTPUT_COLUMNS)) + " |",
    ]
    for row in table_rows:
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in OUTPUT_COLUMNS) + " |")
    md_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()
    root = Path(args.results_root)
    rows = load_rows(root)
    write_outputs(rows, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
