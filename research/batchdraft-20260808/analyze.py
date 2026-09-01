#!/usr/bin/env python3
"""Join client, scheduler-phase, GPU-sampler, and verify-width receipts.

The counterfactual deliberately changes only target verification. Draft, host commit,
prompt/session setup, and scheduler overhead remain at their measured values.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


PHASE_RE = re.compile(
    r"\[spec-phase\] draft=(?P<draft>[0-9.]+)ms .*?"
    r"verify-issue=(?P<issue>[0-9.]+)ms .*?"
    r"verify-wait=(?P<wait>[0-9.]+)ms .*?"
    r"commit-host=(?P<commit>[0-9.]+)ms .*?rounds=(?P<rounds>\d+)"
)
TICK_RE = re.compile(
    r"\[tick-spec\] seq=(?P<seq>\d+) slot=(?P<slot>\d+) trace=(?P<trace>\S+) "
    r"start_ms=(?P<start>[0-9.]+) gap_ms=(?P<gap>[0-9.]+) "
    r"wall_ms=(?P<wall>[0-9.]+) generated=(?P<generated0>\d+)->(?P<generated1>\d+) "
    r"rounds=(?P<rounds>\d+) drafted=(?P<drafted>\d+) accepted=(?P<accepted>\d+) "
    r"k=(?P<k>\d+)"
)
TRACE_RE = re.compile(r"r(?P<rep>\d+)-(?P<arm>sync|divergent)-q(?P<request>\d+)")
MSCALE_RE = re.compile(
    r"verify m=(?P<m>\d+) @d(?P<depth>\d+): median\s+(?P<median>[0-9.]+) us\s+"
    r"p10\s+(?P<p10>[0-9.]+)\s+p90\s+(?P<p90>[0-9.]+)"
)


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def parse_iso(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_client(path: Path) -> dict[tuple[int, str], dict]:
    points = {}
    for raw in path.read_text().splitlines():
        row = json.loads(raw)
        if row.get("kind") != "point" or row.get("arm") not in {"sync", "divergent"}:
            continue
        points[(int(row["rep"]), row["arm"])] = row
    return points


def parse_server(path: Path) -> list[dict]:
    calls = []
    pending_phase = None
    tick_id = -1
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        phase_match = PHASE_RE.search(line)
        if phase_match:
            pending_phase = {
                key: float(phase_match.group(key))
                for key in ("draft", "issue", "wait", "commit")
            }
            pending_phase["rounds"] = int(phase_match.group("rounds"))
            pending_phase["line"] = lineno
            continue
        tick_match = TICK_RE.search(line)
        if not tick_match:
            continue
        if pending_phase is None:
            raise ValueError(f"tick-spec at {path}:{lineno} has no preceding spec-phase")
        row = tick_match.groupdict()
        seq = int(row["seq"])
        if seq == 1:
            tick_id += 1
        trace_match = TRACE_RE.fullmatch(row["trace"])
        call = {
            "tick_id": tick_id,
            "line": lineno,
            "seq": seq,
            "slot": int(row["slot"]),
            "trace": row["trace"],
            "start_ms": float(row["start"]),
            "gap_ms": float(row["gap"]),
            "wall_ms": float(row["wall"]),
            "generated0": int(row["generated0"]),
            "generated1": int(row["generated1"]),
            "rounds": int(row["rounds"]),
            "drafted": int(row["drafted"]),
            "accepted": int(row["accepted"]),
            "k": int(row["k"]),
            "phase": pending_phase,
            "point": None,
        }
        if trace_match:
            call["point"] = (int(trace_match.group("rep")), trace_match.group("arm"))
        if call["rounds"] != pending_phase["rounds"]:
            raise ValueError(
                f"round mismatch at {path}:{lineno}: tick={call['rounds']} "
                f"phase={pending_phase['rounds']}"
            )
        calls.append(call)
        pending_phase = None
    return calls


def parse_mscale(path: Path) -> dict[int, dict]:
    widths = {}
    for line in path.read_text().splitlines():
        match = MSCALE_RE.search(line)
        if match:
            widths[int(match.group("m"))] = {
                "depth": int(match.group("depth")),
                "median_us": float(match.group("median")),
                "p10_us": float(match.group("p10")),
                "p90_us": float(match.group("p90")),
            }
    if set(widths) != {4, 8, 12, 16}:
        raise ValueError(f"expected m=4,8,12,16 in {path}; got {sorted(widths)}")
    return widths


def parse_gpu(path: Path) -> list[dict]:
    samples = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, skipinitialspace=True):
            samples.append({
                "time": dt.datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
                .replace(tzinfo=dt.timezone.utc),
                "index": int(row["index"]),
                "util": float(row["utilization.gpu [%]"]),
            })
    return samples


def max_zero_run_ms(samples: list[dict]) -> float:
    if not samples:
        return 0.0
    best = 0.0
    start = None
    previous = None
    for row in sorted(samples, key=lambda item: item["time"]):
        if row["util"] == 0:
            if start is None or (previous and (row["time"] - previous).total_seconds() > 0.25):
                start = row["time"]
            best = max(best, (row["time"] - start).total_seconds() * 1000.0 + 100.0)
        else:
            start = None
        previous = row["time"]
    return best


def gpu_for_point(samples: list[dict], point: dict) -> dict:
    start = parse_iso(point["start_utc"])
    end = parse_iso(point["end_utc"])
    window = [row for row in samples if start <= row["time"] <= end]
    result = {}
    for index in (0, 1):
        rows = [row for row in window if row["index"] == index]
        result[f"gpu{index}_samples"] = len(rows)
        result[f"gpu{index}_mean_util_pct"] = statistics.mean(row["util"] for row in rows)
        result[f"gpu{index}_zero_sample_pct"] = (
            100.0 * sum(row["util"] == 0 for row in rows) / len(rows)
        )
        result[f"gpu{index}_max_zero_run_ms"] = max_zero_run_ms(rows)

    # Bucket to the sampler's requested 100 ms cadence. A bucket is jointly idle only when
    # both stage devices supplied a sample and both reported zero utilization.
    buckets: dict[int, dict[int, float]] = defaultdict(dict)
    for row in window:
        buckets[int(row["time"].timestamp() * 10)][row["index"]] = row["util"]
    paired = sorted(
        (bucket, values) for bucket, values in buckets.items() if set(values) == {0, 1}
    )
    idle_buckets = [bucket for bucket, values in paired if values[0] == 0 and values[1] == 0]
    longest = 0
    run = 0
    previous = None
    for bucket in idle_buckets:
        run = run + 1 if previous is not None and bucket == previous + 1 else 1
        longest = max(longest, run)
        previous = bucket
    result["paired_samples"] = len(paired)
    result["joint_zero_sample_pct"] = 100.0 * len(idle_buckets) / len(paired) if paired else 0.0
    result["joint_max_zero_run_ms"] = longest * 100.0
    return result


def projection_for_point(
    point_key: tuple[int, str],
    point: dict,
    calls: list[dict],
    widths: dict[int, dict],
) -> dict:
    selected = [call for call in calls if call["point"] == point_key]
    grouped: dict[int, list[dict]] = defaultdict(list)
    for call in selected:
        if call["rounds"]:
            grouped[call["tick_id"]].append(call)

    t4 = widths[4]["median_us"]
    phase_ms = 0.0
    verify_ms = 0.0
    draft_ms = 0.0
    commit_ms = 0.0
    proxy_serial_us = 0.0
    proxy_fused_us = 0.0
    proxy_ideal_us = 0.0
    round_slots = defaultdict(int)
    for tick_calls in grouped.values():
        rounds = [call["rounds"] for call in tick_calls]
        for call in tick_calls:
            phase = call["phase"]
            draft_ms += phase["draft"]
            verify_ms += phase["issue"] + phase["wait"]
            commit_ms += phase["commit"]
            phase_ms += phase["draft"] + phase["issue"] + phase["wait"] + phase["commit"]
            proxy_serial_us += call["rounds"] * t4
        for wave in range(1, max(rounds) + 1):
            live = sum(round_count >= wave for round_count in rounds)
            round_slots[live] += 1
            proxy_fused_us += widths[live * 4]["median_us"]
            # Ideal four-for-one ceiling: one single-session verify cost at any live B.
            proxy_ideal_us += t4

    fused_ratio = proxy_fused_us / proxy_serial_us
    ideal_ratio = proxy_ideal_us / proxy_serial_us
    flattened_saved_s = verify_ms * (1.0 - fused_ratio) / 1000.0
    ideal_saved_s = verify_ms * (1.0 - ideal_ratio) / 1000.0
    zero_verify_saved_s = verify_ms / 1000.0

    def phase_speedup(verify_cost_ratio: float) -> float:
        projected_phase_ms = phase_ms - verify_ms + verify_ms * verify_cost_ratio
        return phase_ms / projected_phase_ms

    def projected_rate(saved_s: float) -> float:
        projected_wall = point["wall_s"] - saved_s
        return point["completion_tokens"] / projected_wall

    steady = [call for call in selected if call["rounds"] and call["generated0"] > 0]
    boundary_gaps = [call["gap_ms"] for call in selected if call["seq"] > 1]
    steady_unaccounted = [
        call["wall_ms"]
        - sum(call["phase"][name] for name in ("draft", "issue", "wait", "commit"))
        for call in steady
    ]
    return {
        "n_spec_calls": len(selected),
        "n_phase_calls": sum(call["rounds"] > 0 for call in selected),
        "n_rounds": sum(call["rounds"] for call in selected),
        "draft_ms": draft_ms,
        "verify_ms": verify_ms,
        "commit_ms": commit_ms,
        "phase_ms": phase_ms,
        "verify_phase_pct": 100.0 * verify_ms / phase_ms,
        "round_width_histogram": dict(sorted(round_slots.items())),
        "flattened_verify_cost_ratio": fused_ratio,
        "ideal_verify_cost_ratio": ideal_ratio,
        "flattened_saved_s": flattened_saved_s,
        "ideal_saved_s": ideal_saved_s,
        "zero_verify_saved_s": zero_verify_saved_s,
        "flattened_phase_speedup": phase_speedup(fused_ratio),
        "ideal_phase_speedup": phase_speedup(ideal_ratio),
        "zero_verify_phase_speedup": phase_speedup(0.0),
        "flattened_projected_tok_s": projected_rate(flattened_saved_s),
        "ideal_projected_tok_s": projected_rate(ideal_saved_s),
        "zero_verify_projected_tok_s": projected_rate(zero_verify_saved_s),
        "scheduler_gap_median_ms": median(boundary_gaps),
        "scheduler_gap_max_ms": max(boundary_gaps, default=0.0),
        "steady_unaccounted_median_ms": median(steady_unaccounted),
        "steady_unaccounted_max_ms": max(steady_unaccounted, default=0.0),
    }


def summarize(points: dict[tuple[int, str], dict], rows: list[dict]) -> dict:
    summary = {}
    for arm in ("sync", "divergent"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        fields = (
            "aggregate_output_tok_s",
            "wall_s",
            "verify_phase_pct",
            "flattened_verify_cost_ratio",
            "flattened_phase_speedup",
            "flattened_projected_tok_s",
            "ideal_phase_speedup",
            "ideal_projected_tok_s",
            "zero_verify_phase_speedup",
            "zero_verify_projected_tok_s",
            "scheduler_gap_median_ms",
            "scheduler_gap_max_ms",
            "steady_unaccounted_median_ms",
            "gpu0_mean_util_pct",
            "gpu1_mean_util_pct",
            "joint_zero_sample_pct",
            "joint_max_zero_run_ms",
        )
        entry = {"n": len(arm_rows), "completion_tokens": [row["completion_tokens"] for row in arm_rows]}
        for field in fields:
            values = [float(row[field]) for row in arm_rows]
            entry[field] = {
                "median": median(values),
                "min": min(values),
                "max": max(values),
            }
        entry["round_width_histogram"] = {
            str(width): sum(int(row["round_width_histogram"].get(width, 0)) for row in arm_rows)
            for width in range(1, 5)
        }
        summary[arm] = entry
    return summary


def summarize_requests(calls: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for call in calls:
        if call["point"] is not None and call["point"][0] > 0 and call["rounds"]:
            grouped[call["trace"]].append(call)

    summary = {}
    for arm in ("sync", "divergent"):
        rows = []
        for trace_calls in grouped.values():
            if trace_calls[0]["point"][1] != arm:
                continue
            draft_ms = sum(call["phase"]["draft"] for call in trace_calls)
            verify_ms = sum(
                call["phase"]["issue"] + call["phase"]["wait"] for call in trace_calls
            )
            commit_ms = sum(call["phase"]["commit"] for call in trace_calls)
            phase_ms = draft_ms + verify_ms + commit_ms
            rows.append({
                "spec_calls": len(trace_calls),
                "rounds": sum(call["rounds"] for call in trace_calls),
                "draft_ms": draft_ms,
                "verify_ms": verify_ms,
                "commit_ms": commit_ms,
                "phase_ms": phase_ms,
                "verify_phase_pct": 100.0 * verify_ms / phase_ms,
            })
        entry = {"n_requests": len(rows), "n_point_replicates": 5}
        for field in rows[0]:
            values = [float(row[field]) for row in rows]
            entry[field] = {"median": median(values), "min": min(values), "max": max(values)}
        summary[arm] = entry
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("client", type=Path)
    parser.add_argument("server", type=Path)
    parser.add_argument("gpu", type=Path)
    parser.add_argument("mscale", type=Path)
    args = parser.parse_args()

    points = parse_client(args.client)
    calls = parse_server(args.server)
    widths = parse_mscale(args.mscale)
    gpu = parse_gpu(args.gpu)
    if set(points) != {(rep, arm) for rep in range(1, 6) for arm in ("sync", "divergent")}:
        raise ValueError(f"expected N=5 for both arms, got {sorted(points)}")

    rows = []
    for key, point in sorted(points.items()):
        row = dict(point)
        row.update(projection_for_point(key, point, calls, widths))
        row.update(gpu_for_point(gpu, point))
        rows.append(row)

    output = {
        "inputs": {
            "client": str(args.client),
            "server": str(args.server),
            "gpu": str(args.gpu),
            "mscale": str(args.mscale),
        },
        "verify_widths": widths,
        "points": rows,
        "summary": summarize(points, rows),
        "request_phase_summary": summarize_requests(calls),
        "projection_contract": {
            "flattened": (
                "For each observed scheduler tick and speculative round wave, replace B serial "
                "m=4 target verifies with the measured contiguous m=4B cost at depth 256; "
                "scale only measured verify wall."
            ),
            "ideal": (
                "For each live round wave, charge one measured m=4 verify regardless of B; "
                "this is a four-for-one target-only ceiling, not an implementation forecast."
            ),
            "held_constant": "draft, prompt/session setup, host commit, scheduling, and output",
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
