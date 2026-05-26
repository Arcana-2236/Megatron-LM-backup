#!/usr/bin/env python3
import argparse
import csv
import json
import sqlite3
from pathlib import Path


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def table_names(conn):
    return [
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    ]


def columns(conn, table):
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def first_table(names, candidates):
    for candidate in candidates:
        if candidate in names:
            return candidate
    for name in names:
        lowered = name.lower()
        if any(candidate.lower() in lowered for candidate in candidates):
            return name
    return None


def duration_rows(conn, table):
    if not table:
        return []
    cols = columns(conn, table)
    if "start" not in cols or "end" not in cols:
        return []
    return [
        (start, end, max(end - start, 0))
        for start, end in conn.execute(f"SELECT start, end FROM {table} WHERE end >= start")
    ]


def runtime_named_rows(conn, table):
    if not table:
        return []
    cols = columns(conn, table)
    if "start" not in cols or "end" not in cols or "nameId" not in cols:
        return []
    names = table_names(conn)
    if "StringIds" not in names:
        return []
    return [
        (name, max(end - start, 0))
        for name, start, end in conn.execute(
            f"""
            SELECT s.value, r.start, r.end
            FROM {table} r
            JOIN StringIds s ON r.nameId = s.id
            WHERE r.end >= r.start
            """
        )
    ]


def memcpy_records(conn, table):
    if not table:
        return []
    cols = columns(conn, table)
    required = {"start", "end", "bytes", "copyKind"}
    if not required.issubset(set(cols)):
        return []
    names = table_names(conn)
    if "ENUM_CUDA_MEMCPY_OPER" in names:
        query = f"""
            SELECT m.start, m.end, m.bytes, COALESCE(e.label, CAST(m.copyKind AS TEXT))
            FROM {table} m
            LEFT JOIN ENUM_CUDA_MEMCPY_OPER e ON m.copyKind = e.id
            WHERE m.end >= m.start
        """
    else:
        query = f"""
            SELECT start, end, bytes, CAST(copyKind AS TEXT)
            FROM {table}
            WHERE end >= start
        """
    return [
        {
            "duration_us": max(end - start, 0) / 1000,
            "bytes": byte_count or 0,
            "kind": kind,
        }
        for start, end, byte_count, kind in conn.execute(query)
    ]


def numeric_sum(conn, table, candidates):
    if not table:
        return None
    cols = columns(conn, table)
    for column in candidates:
        if column in cols:
            value = conn.execute(f"SELECT SUM({column}) FROM {table}").fetchone()[0]
            return value or 0
    return None


def summarize(path: Path):
    conn = sqlite3.connect(path)
    try:
        names = table_names(conn)
        kernel_table = first_table(names, ["CUPTI_ACTIVITY_KIND_KERNEL"])
        runtime_table = first_table(names, ["CUPTI_ACTIVITY_KIND_RUNTIME"])
        memcpy_table = first_table(names, ["CUPTI_ACTIVITY_KIND_MEMCPY"])

        kernel_rows = duration_rows(conn, kernel_table)
        kernel_durations_us = [duration / 1000 for _, _, duration in kernel_rows]
        runtime_rows = duration_rows(conn, runtime_table)
        runtime_durations_us = [duration / 1000 for _, _, duration in runtime_rows]
        runtime_named = runtime_named_rows(conn, runtime_table)
        launch_durations_us = [
            duration / 1000
            for name, duration in runtime_named
            if "LaunchKernel" in name or name == "cuLaunchKernel"
        ]
        sync_durations_us = [
            duration / 1000
            for name, duration in runtime_named
            if "Synchronize" in name or "cudaEventQuery" in name
        ]
        memcpy_records_by_kind = memcpy_records(conn, memcpy_table)
        memcpy_durations_us = [record["duration_us"] for record in memcpy_records_by_kind]
        memcpy_sizes = [record["bytes"] for record in memcpy_records_by_kind]

        idle_gaps_us = []
        if kernel_rows:
            previous_end = None
            for start, end, _duration in sorted(kernel_rows):
                if previous_end is not None and start > previous_end:
                    idle_gaps_us.append((start - previous_end) / 1000)
                previous_end = max(previous_end or end, end)

        memcpy_bytes = numeric_sum(conn, memcpy_table, ["bytes", "copySize", "size"])
        span_us = None
        if kernel_rows:
            span_us = (max(row[1] for row in kernel_rows) - min(row[0] for row in kernel_rows)) / 1000
        by_kind = {}
        for record in memcpy_records_by_kind:
            bucket = by_kind.setdefault(record["kind"], {"count": 0, "bytes": 0, "durations": []})
            bucket["count"] += 1
            bucket["bytes"] += record["bytes"]
            bucket["durations"].append(record["duration_us"])

        def kind_value(kind, key):
            bucket = by_kind.get(kind, {})
            if key == "avg_bytes" and bucket.get("count"):
                return bucket.get("bytes", 0) / bucket["count"]
            if key == "avg_us" and bucket.get("durations"):
                return sum(bucket["durations"]) / len(bucket["durations"])
            return bucket.get(key)

        return {
            "sqlite_file": str(path),
            "kernel_table": kernel_table,
            "kernel_count": len(kernel_durations_us),
            "avg_kernel_us": sum(kernel_durations_us) / len(kernel_durations_us)
            if kernel_durations_us
            else None,
            "p50_kernel_us": percentile(kernel_durations_us, 0.50),
            "p90_kernel_us": percentile(kernel_durations_us, 0.90),
            "p99_kernel_us": percentile(kernel_durations_us, 0.99),
            "total_kernel_us": sum(kernel_durations_us) if kernel_durations_us else 0,
            "kernel_span_us": span_us,
            "idle_gap_count": len(idle_gaps_us),
            "avg_idle_gap_us": sum(idle_gaps_us) / len(idle_gaps_us) if idle_gaps_us else None,
            "total_idle_gap_us": sum(idle_gaps_us) if idle_gaps_us else 0,
            "idle_gap_fraction": (sum(idle_gaps_us) / span_us) if idle_gaps_us and span_us else None,
            "cuda_api_count": len(runtime_durations_us),
            "avg_cuda_api_us": sum(runtime_durations_us) / len(runtime_durations_us)
            if runtime_durations_us
            else None,
            "total_cuda_api_us": sum(runtime_durations_us) if runtime_durations_us else 0,
            "cuda_launch_api_count": len(launch_durations_us),
            "avg_cuda_launch_api_us": sum(launch_durations_us) / len(launch_durations_us)
            if launch_durations_us
            else None,
            "total_cuda_launch_api_us": sum(launch_durations_us) if launch_durations_us else 0,
            "cuda_sync_api_count": len(sync_durations_us),
            "avg_cuda_sync_api_us": sum(sync_durations_us) / len(sync_durations_us)
            if sync_durations_us
            else None,
            "total_cuda_sync_api_us": sum(sync_durations_us) if sync_durations_us else 0,
            "memcpy_count": len(memcpy_durations_us),
            "avg_memcpy_us": sum(memcpy_durations_us) / len(memcpy_durations_us)
            if memcpy_durations_us
            else None,
            "p50_memcpy_us": percentile(memcpy_durations_us, 0.50),
            "p90_memcpy_us": percentile(memcpy_durations_us, 0.90),
            "p99_memcpy_us": percentile(memcpy_durations_us, 0.99),
            "total_memcpy_bytes": memcpy_bytes,
            "avg_memcpy_bytes": (memcpy_bytes / len(memcpy_durations_us))
            if memcpy_bytes is not None and memcpy_durations_us
            else None,
            "p50_memcpy_bytes": percentile(memcpy_sizes, 0.50),
            "p90_memcpy_bytes": percentile(memcpy_sizes, 0.90),
            "p99_memcpy_bytes": percentile(memcpy_sizes, 0.99),
            "h2d_memcpy_count": kind_value("Host-to-Device", "count"),
            "h2d_memcpy_bytes": kind_value("Host-to-Device", "bytes"),
            "h2d_avg_memcpy_bytes": kind_value("Host-to-Device", "avg_bytes"),
            "h2d_avg_memcpy_us": kind_value("Host-to-Device", "avg_us"),
            "d2h_memcpy_count": kind_value("Device-to-Host", "count"),
            "d2h_memcpy_bytes": kind_value("Device-to-Host", "bytes"),
            "d2h_avg_memcpy_bytes": kind_value("Device-to-Host", "avg_bytes"),
            "d2h_avg_memcpy_us": kind_value("Device-to-Host", "avg_us"),
            "d2d_memcpy_count": kind_value("Device-to-Device", "count"),
            "d2d_memcpy_bytes": kind_value("Device-to-Device", "bytes"),
            "d2d_avg_memcpy_bytes": kind_value("Device-to-Device", "avg_bytes"),
            "d2d_avg_memcpy_us": kind_value("Device-to-Device", "avg_us"),
        }
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", nargs="+")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    rows = [summarize(Path(item)) for item in args.sqlite]
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
