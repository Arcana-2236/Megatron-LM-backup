#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\):\s*([0-9.]+)"
)
MEM_RE = re.compile(r"global max allocated:\s*([0-9.]+).*?global max reserved:\s*([0-9.]+)")
FAIL_RE = re.compile(
    r"(RUN_TIMEOUT|torch\.OutOfMemoryError|CUDA out of memory|OutOfMemoryError|Traceback \(most recent call last\)|AssertionError|ValueError|RuntimeError|NotImplementedError|FAILED|error:)",
    re.IGNORECASE,
)


def parse_log(path: Path, warmup: int) -> dict:
    text = path.read_text(errors="replace") if path.exists() else ""
    times = [(int(i), float(ms)) for i, ms in ITER_RE.findall(text)]
    post_warmup = [ms for i, ms in times if i > warmup]
    mem = [(float(alloc), float(reserved)) for alloc, reserved in MEM_RE.findall(text)]
    failures = [m.group(0) for m in FAIL_RE.finditer(text)]
    return {
        "status": "failed" if failures else ("ok" if post_warmup else "missing"),
        "iteration_time_ms": sum(post_warmup) / len(post_warmup) if post_warmup else None,
        "peak_allocated_gib": max((m[0] for m in mem), default=None) / 1024 if mem else None,
        "peak_reserved_gib": max((m[1] for m in mem), default=None) / 1024 if mem else None,
        "failure_reason": failures[0] if failures else "",
        "num_logged_iters": len(times),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="benchmarks/cola_memory/results")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--warmup", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.results_root)
    rows = []
    for meta_path in sorted((root / "meta").glob("*.json")):
        meta = json.loads(meta_path.read_text())
        warmup = args.warmup if args.warmup is not None else int(meta.get("warmup_iters", 5))
        parsed = parse_log(Path(meta["log_file"]), warmup)
        rows.append({**meta, **parsed})

    out_csv = Path(args.out_csv) if args.out_csv else root / "summary.csv"
    out_json = Path(args.out_json) if args.out_json else root / "summary.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
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
