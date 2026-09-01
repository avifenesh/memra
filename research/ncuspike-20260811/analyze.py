#!/usr/bin/env python3
"""Reduce box1 Nsys timing and selected NCU counters into one auditable summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


GEN_RE = re.compile(r"generated (\d+) tokens in ([0-9.]+)s = ([0-9.]+) tok/s")
WINDOW_RE = re.compile(
    r"(?:serving )?decode window @d(\d+): (\d+) tokens in ([0-9.]+)s = ([0-9.]+) tok/s"
)
STALL_RE = re.compile(r"smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio")
ACTIVE_EXPERT_BYTES_PER_TOKEN = 6_101_901_312
MODEL_REFERENCE_BW_GBS = 1_790.0
SERVER_EDITION_BW_GBS = 1_597.0


def category(name: str) -> str:
    if name.startswith("[CUDA "):
        return "CUDA memcpy/memset"
    if name.startswith("moe_router_sigmoid"):
        return "device sigmoid router"
    if name.startswith("moe_gate_up"):
        return "routed expert gate/up/SwiGLU"
    if name.startswith("moe_down8"):
        return "routed expert weighted-down"
    if name.startswith("moe_pairs"):
        return "clamped routed-expert tail"
    if name.startswith("fa_decode"):
        return "attention decode"
    if name.startswith("qmatvec_"):
        return "trunk/shared/head matvec"
    if name.startswith("append_quantize_kv"):
        return "KV append"
    if any(part in name for part in ("rms", "rope", "quantize", "silu", "add_")):
        return "norm/quant/activation glue"
    return "other GPU kernels"


def shape(value: str) -> tuple[int, int, int]:
    numbers = tuple(int(item) for item in re.findall(r"\d+", value))
    if len(numbers) != 3:
        raise ValueError(f"unexpected launch shape: {value!r}")
    return numbers


def device_id(value: str) -> str:
    """Normalize the device labels emitted by Nsys and NCU to a CUDA ordinal."""

    value = value.strip()
    if value.isdigit():
        return value
    match = re.search(r"\((\d+)\)$", value)
    return match.group(1) if match else value


def number(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals.sort()
    total = 0
    begin, end = intervals[0]
    for next_begin, next_end in intervals[1:]:
        if next_begin <= end:
            end = max(end, next_end)
        else:
            total += end - begin
            begin, end = next_begin, next_end
    return total + end - begin


def parse_nsys(trace_path: Path, run_log: Path) -> dict:
    log_text = run_log.read_text()
    match = GEN_RE.search(log_text)
    window_match = WINDOW_RE.search(log_text)
    if match:
        context_depth = None
        n_tokens = int(match.group(1))
        logged_seconds = float(match.group(2))
        logged_tok_s = float(match.group(3))
    elif window_match:
        context_depth = int(window_match.group(1))
        n_tokens = int(window_match.group(2))
        logged_seconds = float(window_match.group(3))
        logged_tok_s = float(window_match.group(4))
    else:
        raise ValueError(f"missing generated-token timing in {run_log}")

    configs: dict[tuple[str, str, tuple[int, int, int], tuple[int, int, int]], dict] = {}
    kernels: dict[str, dict] = {}
    categories: dict[str, dict] = {}
    device_kernels: dict[str, dict[str, dict]] = defaultdict(dict)
    device_categories: dict[str, dict[str, dict]] = defaultdict(dict)
    intervals: list[tuple[int, int]] = []
    device_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    gpu_sum_ns = 0
    kernel_sum_ns = 0

    with trace_path.open(newline="") as handle:
        reader = None
        for line in handle:
            if line.startswith("Start (ns),"):
                reader = csv.DictReader(handle, fieldnames=next(csv.reader([line])))
                break
        if reader is None:
            raise ValueError(f"missing Nsys CSV header in {trace_path}")
        raw_rows = list(reader)

    # Legacy run-gen ends every B=1 Step token with one 515,584-byte full-logit D2H and then
    # performs an extra steady-state replay. Slice that diagnostic at its first N endpoints.
    # The serving harness uses device sampling + lean logits, and its cudaProfilerApi range is
    # already exactly N steps, so it correctly has no full-vocabulary D2H endpoint at all.
    decode_endpoints = [
        row
        for row in raw_rows
        if row["Name"] == "[CUDA memcpy Device-to-Host]"
        and number(row.get("Bytes (MB)")) is not None
        and number(row.get("Bytes (MB)")) >= 0.5
    ]
    window_start_ns = min(int(row["Start (ns)"]) for row in raw_rows)
    if match:
        if len(decode_endpoints) < n_tokens:
            raise ValueError(
                f"only {len(decode_endpoints)} full-logit D2H endpoints for {n_tokens} tokens"
            )
        selected_endpoints = decode_endpoints[:n_tokens]
        window_end_ns = int(selected_endpoints[-1]["Start (ns)"]) + int(
            selected_endpoints[-1]["Duration (ns)"]
        )
        window_selection = "first_n_full_vocab_dtoh"
    else:
        selected_endpoints = []
        window_end_ns = max(
            int(row["Start (ns)"]) + int(row["Duration (ns)"])
            for row in raw_rows
        )
        window_selection = "exact_cuda_profiler_api_range"
    trace_rows = [
        row
        for row in raw_rows
        if int(row["Start (ns)"]) >= window_start_ns
        and int(row["Start (ns)"]) + int(row["Duration (ns)"]) <= window_end_ns
    ]

    for row in trace_rows:
        start = int(row["Start (ns)"])
        duration = int(row["Duration (ns)"])
        end = start + duration
        name = row["Name"]
        device = device_id(row["Device"])
        intervals.append((start, end))
        device_intervals[device].append((start, end))
        gpu_sum_ns += duration
        is_memop = name.startswith("[CUDA ")
        if not is_memop:
            kernel_sum_ns += duration
            key = (
                name,
                device,
                (int(row["GrdX"]), int(row["GrdY"]), int(row["GrdZ"])),
                (int(row["BlkX"]), int(row["BlkY"]), int(row["BlkZ"])),
            )
            item = configs.setdefault(
                key,
                {"total_ns": 0, "launches": 0, "min_ns": 2**63 - 1, "max_ns": 0},
            )
            item["total_ns"] += duration
            item["launches"] += 1
            item["min_ns"] = min(item["min_ns"], duration)
            item["max_ns"] = max(item["max_ns"], duration)
        for bucket, mapping in ((name, kernels), (category(name), categories)):
            item = mapping.setdefault(bucket, {"total_ns": 0, "launches": 0})
            item["total_ns"] += duration
            item["launches"] += 1
        for bucket, mapping in (
            (name, device_kernels[device]),
            (category(name), device_categories[device]),
        ):
            item = mapping.setdefault(bucket, {"total_ns": 0, "launches": 0})
            item["total_ns"] += duration
            item["launches"] += 1

    if not intervals:
        raise ValueError(f"no GPU activities in {trace_path}")
    first = min(start for start, _ in intervals)
    last = max(end for _, end in intervals)
    span_ns = last - first
    busy_ns = union_ns(intervals)
    launch_gap_ns = span_ns - busy_ns

    def rows(mapping: dict[str, dict]) -> list[dict]:
        result = []
        for name, item in mapping.items():
            result.append(
                {
                    "name": name,
                    "launches": item["launches"],
                    "launches_per_token": item["launches"] / n_tokens,
                    "total_ms": item["total_ns"] / 1e6,
                    "per_token_ms": item["total_ns"] / n_tokens / 1e6,
                    "wall_share_pct": 100.0 * item["total_ns"] / span_ns,
                }
            )
        return sorted(result, key=lambda item: item["total_ms"], reverse=True)

    config_rows = []
    for (name, device, grid, block), item in configs.items():
        config_rows.append(
            {
                "kernel": name,
                "device": device,
                "grid": list(grid),
                "block": list(block),
                "category": category(name),
                "launches": item["launches"],
                "total_ms": item["total_ns"] / 1e6,
                "per_token_ms": item["total_ns"] / n_tokens / 1e6,
                "avg_us": item["total_ns"] / item["launches"] / 1e3,
                "min_us": item["min_ns"] / 1e3,
                "max_us": item["max_ns"] / 1e3,
            }
        )
    config_rows.sort(key=lambda item: item["total_ms"], reverse=True)

    per_device = {}
    for device in sorted(device_intervals):
        busy = union_ns(device_intervals[device])
        per_device[device] = {
            "busy_ms": busy / 1e6,
            "busy_per_token_ms": busy / n_tokens / 1e6,
            "categories": rows(device_categories[device]),
            "kernels": rows(device_kernels[device]),
        }

    return {
        "n_tokens": n_tokens,
        "context_depth": context_depth,
        "window_selection": window_selection,
        "raw_trace_activities": len(raw_rows),
        "raw_full_vocab_dtoh_endpoints": len(decode_endpoints),
        "selected_trace_activities": len(trace_rows),
        "selected_full_vocab_dtoh_endpoints": len(selected_endpoints),
        "excluded_post_generation_decode_endpoints": (
            len(decode_endpoints) - n_tokens if match else 0
        ),
        "logged_gen_seconds": logged_seconds,
        "logged_gen_tok_s": logged_tok_s,
        "trace_span_ms": span_ns / 1e6,
        "wall_per_token_ms": span_ns / n_tokens / 1e6,
        "gpu_sum_ms": gpu_sum_ns / 1e6,
        "kernel_sum_ms": kernel_sum_ns / 1e6,
        "gpu_busy_union_ms": busy_ns / 1e6,
        "gpu_busy_per_token_ms": busy_ns / n_tokens / 1e6,
        "launch_gap_ms": launch_gap_ns / 1e6,
        "launch_gap_per_token_ms": launch_gap_ns / n_tokens / 1e6,
        "launch_gap_pct": 100.0 * launch_gap_ns / span_ns,
        "gpu_overlap_ms": gpu_sum_ns / 1e6 - busy_ns / 1e6,
        "per_device_busy_ms": {
            device: union_ns(device_rows) / 1e6
            for device, device_rows in sorted(device_intervals.items())
        },
        "per_device": per_device,
        "categories": rows(categories),
        "kernels": rows(kernels),
        "launch_configs": config_rows,
        "_config_timing": configs,
    }


NCU_METRICS = {
    "gpu__time_duration.avg": "ncu_duration_us",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_pct",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "memory_pct",
    "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed": "memory_access_pct",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct",
    "dram__bytes.sum.per_second": "dram_gbs",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
    "sm__maximum_warps_per_active_cycle_pct": "theoretical_occupancy_pct",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__waves_per_multiprocessor": "waves_per_sm",
}


def parse_ncu(raw_paths: list[Path], nsys: dict, lane_root: Path) -> dict:
    samples: dict[
        tuple[str, str, tuple[int, int, int], tuple[int, int, int]], dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    units: dict[str, str] = {}
    for raw_path in raw_paths:
        with raw_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            try:
                file_units = next(reader)
            except StopIteration as error:
                raise ValueError(f"empty NCU CSV: {raw_path}") from error
            if not units:
                units = file_units
            for row in reader:
                key = (
                    row["Kernel Name"],
                    device_id(row["Device"]),
                    shape(row["Grid Size"]),
                    shape(row["Block Size"]),
                )
                for source, target in NCU_METRICS.items():
                    value = number(row.get(source))
                    if value is not None:
                        samples[key][target].append(value)
                for source, raw_value in row.items():
                    match = STALL_RE.fullmatch(source or "")
                    value = number(raw_value)
                    if match and value is not None:
                        samples[key][f"stall_{match.group(1)}"].append(value)

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

    config_rows = []
    config_metrics = {}
    for key, values in samples.items():
        item = {metric: statistics.median(points) for metric, points in values.items()}
        if "ncu_duration_us" in item:
            item["ncu_duration_us"] *= duration_scale
        if "dram_gbs" in item:
            item["dram_gbs"] *= bandwidth_scale
            item["dram_pct_of_1597"] = 100.0 * item["dram_gbs"] / 1597.0
        stalls = sorted(
            (
                {"name": metric.removeprefix("stall_"), "warps_per_issue": value}
                for metric, value in item.items()
                if metric.startswith("stall_")
            ),
            key=lambda row: row["warps_per_issue"],
            reverse=True,
        )[:3]
        item["top_stalls"] = stalls
        config_metrics[key] = item
        name, device, grid, block = key
        timing = nsys["_config_timing"].get(key)
        config_rows.append(
            {
                "kernel": name,
                "device": device,
                "grid": list(grid),
                "block": list(block),
                "nsys_total_ms": timing["total_ns"] / 1e6 if timing else None,
                "nsys_per_token_ms": timing["total_ns"] / nsys["n_tokens"] / 1e6 if timing else None,
                **item,
            }
        )
    config_rows.sort(key=lambda item: item["nsys_total_ms"] or -1, reverse=True)

    def aggregate(keys: list[tuple], identity: dict) -> dict:
        timed = [(key, nsys["_config_timing"].get(key)) for key in keys]
        timed = [(key, timing) for key, timing in timed if timing]
        row = {**identity, "launch_shapes": len(keys)}
        if timed:
            weight = sum(timing["total_ns"] for _, timing in timed)
            row["nsys_total_ms"] = weight / 1e6
            row["nsys_per_token_ms"] = weight / nsys["n_tokens"] / 1e6
            for metric in (
                "sm_pct",
                "memory_pct",
                "memory_access_pct",
                "dram_pct",
                "dram_gbs",
                "dram_pct_of_1597",
                "occupancy_pct",
                "theoretical_occupancy_pct",
                "waves_per_sm",
            ):
                present = [
                    (key, timing)
                    for key, timing in timed
                    if metric in config_metrics[key]
                ]
                metric_weight = sum(timing["total_ns"] for _, timing in present)
                if metric_weight:
                    row[metric] = sum(
                        config_metrics[key][metric] * timing["total_ns"]
                        for key, timing in present
                    ) / metric_weight
        return row

    by_kernel = []
    for kernel in sorted({key[0] for key in config_metrics}):
        keys = [key for key in config_metrics if key[0] == kernel]
        by_kernel.append(aggregate(keys, {"kernel": kernel}))
    by_kernel.sort(key=lambda item: item.get("nsys_total_ms", -1), reverse=True)

    by_kernel_device = []
    for kernel, device in sorted({(key[0], key[1]) for key in config_metrics}):
        keys = [key for key in config_metrics if key[0] == kernel and key[1] == device]
        by_kernel_device.append(
            aggregate(keys, {"kernel": kernel, "device": device})
        )
    by_kernel_device.sort(key=lambda item: item.get("nsys_total_ms", -1), reverse=True)

    coverage_ns = sum(
        timing["total_ns"]
        for key, timing in nsys["_config_timing"].items()
        if key in config_metrics
    )
    selected_names = {key[0] for key in config_metrics}
    selected_symbol_ns = sum(
        timing["total_ns"]
        for key, timing in nsys["_config_timing"].items()
        if key[0] in selected_names
    )
    return {
        "counter_sources": [str(path.relative_to(lane_root)) for path in raw_paths],
        "profiled_devices": sorted({key[1] for key in config_metrics}),
        "selected_symbols": sorted(selected_names),
        "selected_symbol_coverage_pct_of_kernel_time": 100.0
        * selected_symbol_ns
        / (nsys["kernel_sum_ms"] * 1e6),
        "selected_symbol_coverage_pct_of_wall": 100.0
        * selected_symbol_ns
        / (nsys["trace_span_ms"] * 1e6),
        "selected_counter_coverage_pct_of_kernel_time": 100.0
        * coverage_ns
        / (nsys["kernel_sum_ms"] * 1e6),
        "selected_counter_coverage_pct_of_selected_symbol_time": 100.0
        * coverage_ns
        / selected_symbol_ns,
        "by_kernel": by_kernel,
        "by_kernel_device": by_kernel_device,
        "launch_configs": config_rows,
    }


def parse_sol_comparison(receipt_path: Path, lane_root: Path) -> dict:
    """Apply both the frozen lane model and actual box1 card denominators."""

    receipt = json.loads(receipt_path.read_text())
    c1 = receipt["points"]["c1"]
    baseline = float(c1["inc1_median_tok_s"])
    resident = float(c1["default_median_tok_s"])

    denominators = []
    for label, bandwidth_gbs in (
        ("frozen_sol_model_1.79_TBps", MODEL_REFERENCE_BW_GBS),
        ("rtx_pro_6000_server_edition_1.597_TBps", SERVER_EDITION_BW_GBS),
    ):
        sol_tok_s = bandwidth_gbs * 1e9 / ACTIVE_EXPERT_BYTES_PER_TOKEN
        denominators.append(
            {
                "name": label,
                "bandwidth_gbs": bandwidth_gbs,
                "sol_tok_s": sol_tok_s,
                "increment1_pct_of_sol": 100.0 * baseline / sol_tok_s,
                "resident_pct_of_sol": 100.0 * resident / sol_tok_s,
                "resident_delta_pct_points": 100.0 * (resident - baseline) / sol_tok_s,
            }
        )

    return {
        "receipt": str(receipt_path.relative_to(lane_root)),
        "runs_per_arm": int(receipt["runs_per_arm"]),
        "active_expert_bytes_per_token": ACTIVE_EXPERT_BYTES_PER_TOKEN,
        "increment1_median_tok_s": baseline,
        "resident_median_tok_s": resident,
        "relative_gain_pct": float(c1["median_of_arm_medians_delta_pct"]),
        "paired_wins": int(c1["paired_wins"]),
        "denominators": denominators,
        "server_card_memory": "GDDR7",
        "server_spec_url": (
            "https://www.nvidia.com/en-us/data-center/"
            "rtx-pro-6000-blackwell-server-edition/"
        ),
    }


def rounded(value):
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return round(value, 6)
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items() if key != "_config_timing"}
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    box = args.lane / "raw" / "box1"
    nsys = parse_nsys(box / "nsys" / "cuda-gpu-trace.csv", box / "nsys" / "nsys-run.log")
    document = {
        "schema": "memra.ncuspike.box1.v1",
        "nsys": nsys,
        "sol_comparison": parse_sol_comparison(
            box / "sigrouter-perf-receipt" / "summary.json", args.lane
        ),
    }
    ncu_paths = [
        path
        for path in (box / "ncu-device0" / "raw.csv", box / "ncu-device1" / "raw.csv")
        if path.exists()
    ]
    if ncu_paths:
        document["ncu"] = parse_ncu(ncu_paths, nsys, args.lane)
    output = args.lane / "summary.json"
    output.write_text(json.dumps(rounded(document), indent=2) + "\n")
    print(
        f"tokens={nsys['n_tokens']} wall={nsys['wall_per_token_ms']:.3f} ms/token "
        f"busy={nsys['gpu_busy_per_token_ms']:.3f} ms/token "
        f"gap={nsys['launch_gap_per_token_ms']:.3f} ms/token "
        f"({nsys['launch_gap_pct']:.2f}%)"
    )
    for row in nsys["kernels"][:12]:
        print(
            f"{row['per_token_ms']:8.4f} ms/token {row['wall_share_pct']:6.2f}% "
            f"{row['launches_per_token']:7.2f} launches/token {row['name']}"
        )


if __name__ == "__main__":
    main()
