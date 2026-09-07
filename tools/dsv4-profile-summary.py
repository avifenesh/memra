#!/usr/bin/env python3
"""Read-only Nsight kernel-interval accounting; never infer overlap from utilization."""
import argparse
import json
import re
import sqlite3
from pathlib import Path


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            raise ValueError("invalid kernel interval")
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def duration(intervals):
    return sum(end - start for start, end in merge(intervals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    args = parser.parse_args()
    connection = sqlite3.connect(args.sqlite.resolve().as_uri() + "?mode=ro", uri=True)
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        raise SystemExit("REFUSED: no CUDA kernel timeline")
    rows = connection.execute("SELECT start, end, deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    if not rows:
        raise SystemExit("REFUSED: empty CUDA kernel timeline")
    devices = sorted({device for _, _, device in rows})
    per_device = {}
    for device in devices:
        intervals = [(start, end) for start, end, owner in rows if owner == device]
        per_device[device] = {"kernels": len(intervals), "busy_ns": duration(intervals)}
    merged_busy = duration([(start, end) for start, end, _ in rows])
    extent = max(end for _, end, _ in rows) - min(start for start, _, _ in rows)
    result = {
        "kernel_count": len(rows),
        "kernel_sum_ns": sum(end - start for start, end, _ in rows),
        "observed_kernel_extent_ns": extent,
        "any_device_kernel_busy_ns": merged_busy,
        "no_device_kernel_busy_ns_within_extent": extent - merged_busy,
        "devices": per_device,
    }
    if len(devices) == 2:
        overlap = sum(item["busy_ns"] for item in per_device.values()) - merged_busy
        result["both_devices_kernel_busy_ns"] = overlap
        result["overlap_fraction_of_union"] = overlap / merged_busy
    if "DIAGNOSTIC_EVENT" in tables:
        diagnostics = [row[0] for row in connection.execute("SELECT text FROM DIAGNOSTIC_EVENT")]
        dropped = [int(match.group(1)) for text in diagnostics
                   if (match := re.search(r"incomplete CUPTI events dropped:\s*(\d+)", text))]
        result["reported_incomplete_cupti_events_dropped"] = dropped or None
        result["diagnostic_warnings"] = [row[0] for row in connection.execute(
            "SELECT text FROM DIAGNOSTIC_EVENT WHERE severity >= 2")]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
