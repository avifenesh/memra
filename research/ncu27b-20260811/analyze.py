#!/usr/bin/env python3
"""Join unperturbed Nsight Systems time weights to one NCU counter sample per launch shape."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


PHASE_RE = re.compile(
    r"\[spec-phase\] draft=([0-9.]+)ms .*verify-issue=([0-9.]+)ms .*"
    r"verify-wait=([0-9.]+)ms .*commit-host=([0-9.]+)ms .*rounds=(\d+)"
)
STATS_RE = re.compile(
    r"\[spec-stats\] rounds=(\d+).*total=(\d+)/(\d+)=([0-9.]+)"
)
STALL_RE = re.compile(r"smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio")


def number(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def shape(value: str) -> tuple[int, int, int]:
    values = [int(item) for item in re.findall(r"\d+", value)]
    if len(values) != 3:
        raise ValueError(f"unexpected launch shape: {value!r}")
    return tuple(values)  # type: ignore[return-value]


def category(name: str) -> str:
    if name.startswith("[CUDA "):
        return "CUDA memcpy/memset"
    if name == "qmatvec_nvfp4_mmvq_mr2_rp":
        return "drafter head"
    if name.startswith("qmatvec_"):
        if "_b4" in name or "_b2" in name:
            return "target verify qmatvec/mmvq"
        return "drafter body qmatvec/mmvq"
    if name.startswith("fa_decode"):
        return "FA decode"
    if "rms" in name or "rope" in name:
        return "RMS/RoPE glue"
    if name.startswith("argmax_"):
        return "GPU sampling/argmax"
    if name.startswith(("gdn_", "ssm_", "qkv_", "q_gate_", "sigmoid_")):
        return "GDN/SSM glue"
    return "other GPU kernels"


def parse_phase(log_path: Path) -> dict[str, float | int]:
    text = log_path.read_text()
    phase_match = PHASE_RE.search(text)
    stats_match = STATS_RE.search(text)
    if not phase_match or not stats_match:
        raise ValueError(f"missing spec phase/stats line in {log_path}")
    draft_ms, issue_ms, wait_ms, commit_ms = map(float, phase_match.groups()[:4])
    rounds = int(phase_match.group(5))
    stats_rounds, accepted, drafted = map(int, stats_match.groups()[:3])
    if rounds != stats_rounds:
        raise ValueError(f"round mismatch: phase={rounds}, stats={stats_rounds}")
    return {
        "rounds": rounds,
        "accepted": accepted,
        "drafted": drafted,
        "acceptance_pct": 100.0 * accepted / drafted,
        "draft_ms": draft_ms,
        "verify_issue_ms": issue_ms,
        "verify_wait_ms": wait_ms,
        "commit_host_ms": commit_ms,
        "phase_ms": draft_ms + issue_ms + wait_ms + commit_ms,
    }


def parse_nsys(trace_path: Path):
    configs: dict[tuple[str, tuple[int, int, int], tuple[int, int, int]], dict[str, int]] = {}
    categories: dict[str, int] = defaultdict(int)
    kernel_ns = 0
    gpu_ns = 0
    intervals = []
    with trace_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["Name"]
            duration_ns = int(row["Duration (ns)"])
            start_ns = int(row["Start (ns)"])
            intervals.append((start_ns, start_ns + duration_ns))
            gpu_ns += duration_ns
            categories[category(name)] += duration_ns
            if name.startswith("[CUDA "):
                continue
            kernel_ns += duration_ns
            key = (
                name,
                (int(row["GrdX"]), int(row["GrdY"]), int(row["GrdZ"])),
                (int(row["BlkX"]), int(row["BlkY"]), int(row["BlkZ"])),
            )
            item = configs.setdefault(
                key,
                {"total_ns": 0, "launches": 0, "min_ns": 2**63 - 1, "max_ns": 0},
            )
            item["total_ns"] += duration_ns
            item["launches"] += 1
            item["min_ns"] = min(item["min_ns"], duration_ns)
            item["max_ns"] = max(item["max_ns"], duration_ns)
    intervals.sort()
    union_ns = 0
    union_start, union_end = intervals[0]
    for start_ns, end_ns in intervals[1:]:
        if start_ns <= union_end:
            union_end = max(union_end, end_ns)
        else:
            union_ns += union_end - union_start
            union_start, union_end = start_ns, end_ns
    union_ns += union_end - union_start
    span_ns = max(end_ns for _, end_ns in intervals) - intervals[0][0]
    timing = {
        "kernel_ns": kernel_ns,
        "gpu_sum_ns": gpu_ns,
        "gpu_busy_union_ns": union_ns,
        "gpu_overlap_ns": gpu_ns - union_ns,
        "span_ns": span_ns,
        "idle_gap_ns": span_ns - union_ns,
    }
    return configs, categories, timing


METRICS = {
    "gpu__time_duration.avg": "ncu_duration",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_pct",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "mem_pct",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct",
    "dram__bytes.sum.per_second": "dram_gbs",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
    "sm__maximum_warps_per_active_cycle_pct": "theoretical_occupancy_pct",
    "smsp__issue_active.avg.per_cycle_active": "issue_active",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__waves_per_multiprocessor": "waves_per_sm",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": "shared_ld_wavefronts",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum": "shared_st_wavefronts",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "shared_ld_conflicts",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": "shared_st_conflicts",
}


def parse_ncu(raw_path: Path):
    groups: dict[
        tuple[str, tuple[int, int, int], tuple[int, int, int]],
        dict[str, list[float]],
    ] = defaultdict(lambda: defaultdict(list))
    with raw_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        try:
            units = next(reader)
        except StopIteration as error:
            raise ValueError(f"empty NCU CSV: {raw_path}") from error
        duration_scale = {"ns": 0.001, "us": 1.0, "ms": 1000.0}.get(
            units.get("gpu__time_duration.avg", "us"), 1.0
        )
        bandwidth_scale = {
            "byte/s": 1e-9,
            "Kbyte/s": 1e-6,
            "Mbyte/s": 1e-3,
            "Gbyte/s": 1.0,
            "Tbyte/s": 1e3,
        }.get(units.get("dram__bytes.sum.per_second", "Gbyte/s"), 1.0)
        for row in reader:
            key = (row["Kernel Name"], shape(row["Grid Size"]), shape(row["Block Size"]))
            for source, target in METRICS.items():
                value = number(row.get(source))
                if value is None:
                    continue
                if target == "ncu_duration":
                    value *= duration_scale
                elif target == "dram_gbs":
                    value *= bandwidth_scale
                groups[key][target].append(value)
            for source, raw_value in row.items():
                match = STALL_RE.fullmatch(source or "")
                value = number(raw_value)
                if match and value is not None:
                    groups[key][f"stall_{match.group(1)}"].append(value)

    result = {}
    for key, values in groups.items():
        item = {name: statistics.median(samples) for name, samples in values.items()}
        wavefronts = item.get("shared_ld_wavefronts", 0) + item.get("shared_st_wavefronts", 0)
        conflicts = item.get("shared_ld_conflicts", 0) + item.get("shared_st_conflicts", 0)
        item["shared_bank_conflict_pct"] = 100.0 * conflicts / wavefronts if wavefronts else 0.0
        stalls = sorted(
            (
                (value, name.removeprefix("stall_"))
                for name, value in item.items()
                if name.startswith("stall_") and name != "stall_selected"
            ),
            reverse=True,
        )[:3]
        item["top_stalls"] = [{"name": name, "warps_per_issue": value} for value, name in stalls]
        result[key] = item
    return result


def rounded(value):
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return round(value, 4)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.lane / "raw"
    output = args.output or args.lane / "summary.json"

    phase = parse_phase(raw / "nsys-spec-k3-n64.log")
    configs, category_ns, trace_timing = parse_nsys(raw / "nsys-spec-k3-n64-trace.csv")
    counters = parse_ncu(raw / "ncu-spec-k3-n64-raw.csv")
    phase_ns = trace_timing["span_ns"]
    residual_ns = phase_ns - trace_timing["gpu_sum_ns"]
    category_ns["host/API/launch gaps"] = residual_ns

    rows = []
    for key, config_timing in configs.items():
        name, grid, block = key
        counter = counters.get(key)
        if counter is None:
            continue
        total_ns = config_timing["total_ns"]
        row = {
            "kernel": name,
            "grid": list(grid),
            "block": list(block),
            "category": category(name),
            "launches": config_timing["launches"],
            "nsys_total_ms": total_ns / 1e6,
            "nsys_avg_us": total_ns / config_timing["launches"] / 1e3,
            "round_ms": total_ns / int(phase["rounds"]) / 1e6,
            "round_share_pct": 100.0 * total_ns / phase_ns,
            **counter,
        }
        row["suspicious"] = row.get("sm_pct", 100) < 60 and row.get("mem_pct", 100) < 70
        rows.append({name: rounded(value) for name, value in row.items()})
    rows.sort(key=lambda item: item["nsys_total_ms"], reverse=True)

    by_kernel = []
    for kernel in sorted({item["kernel"] for item in rows}):
        members = [item for item in rows if item["kernel"] == kernel]
        total_ms = sum(float(item["nsys_total_ms"]) for item in members)
        weighted = {}
        for metric in ("sm_pct", "mem_pct", "dram_pct", "occupancy_pct"):
            present = [item for item in members if item.get(metric) is not None]
            weight = sum(float(item["nsys_total_ms"]) for item in present)
            if weight:
                weighted[metric] = sum(
                    float(item[metric]) * float(item["nsys_total_ms"]) for item in present
                ) / weight
        kernel_row = {
            "kernel": kernel,
            "category": members[0]["category"],
            "launch_shapes": len(members),
            "launches": sum(int(item["launches"]) for item in members),
            "nsys_total_ms": round(total_ms, 4),
            "round_ms": round(total_ms / int(phase["rounds"]), 4),
            "round_share_pct": round(100.0 * total_ms * 1e6 / phase_ns, 4),
            **{name: round(value, 4) for name, value in weighted.items()},
        }
        kernel_row["suspicious"] = (
            kernel_row.get("sm_pct", 100) < 60 and kernel_row.get("mem_pct", 100) < 70
        )
        by_kernel.append(kernel_row)
    by_kernel.sort(key=lambda item: item["nsys_total_ms"], reverse=True)

    suspicious = [item for item in rows if item["suspicious"]]
    coverage_ns = sum(configs[key]["total_ns"] for key in counters if key in configs)
    categories = []
    for name, duration_ns in sorted(category_ns.items(), key=lambda item: item[1], reverse=True):
        categories.append(
            {
                "category": name,
                "total_ms": round(duration_ns / 1e6, 4),
                "round_ms": round(duration_ns / int(phase["rounds"]) / 1e6, 4),
                "round_share_pct": round(100.0 * duration_ns / phase_ns, 4),
            }
        )

    document = {
        "contract": phase,
        "nsys": {
            "trace_span_ms": round(trace_timing["span_ns"] / 1e6, 4),
            "round_ms": round(trace_timing["span_ns"] / int(phase["rounds"]) / 1e6, 4),
            "kernel_ms": round(trace_timing["kernel_ns"] / 1e6, 4),
            "gpu_sum_ms_including_memops": round(trace_timing["gpu_sum_ns"] / 1e6, 4),
            "gpu_busy_union_ms": round(trace_timing["gpu_busy_union_ns"] / 1e6, 4),
            "gpu_overlap_ms": round(trace_timing["gpu_overlap_ns"] / 1e6, 4),
            "idle_gap_ms": round(trace_timing["idle_gap_ns"] / 1e6, 4),
            "selected_counter_coverage_pct_of_kernel_time": round(
                100.0 * coverage_ns / trace_timing["kernel_ns"], 4
            ),
        },
        "categories": categories,
        "kernels": by_kernel,
        "launch_configs": rows,
        "suspicious_launch_configs": suspicious,
    }
    output.write_text(json.dumps(document, indent=2) + "\n")
    print(
        f"wrote {output}: {len(rows)} profiled launch shapes, "
        f"{document['nsys']['selected_counter_coverage_pct_of_kernel_time']:.2f}% kernel-time coverage, "
        f"{len(suspicious)} suspicious shapes"
    )


if __name__ == "__main__":
    main()
