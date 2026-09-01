#!/usr/bin/env python3
"""Summarize the fixed OPTIPIPE increment-2 c=2 receipt without changing raw data."""

import argparse
import json
import re
import statistics
from pathlib import Path


ARMS = ("plain", "serial", "seam", "q0", "q05", "q07", "q09")
Q_ARMS = ("q0", "q05", "q07", "q09")


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def arm_from_label(label: str) -> str:
    return label.rsplit("-", 1)[-1]


def thermal_summary(root: Path):
    temperatures = []
    clocks = []
    powers = []
    for path in sorted(root.glob("c2-r*-thermal-*.log")):
        for line in path.read_text(errors="replace").splitlines()[2:]:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 11:
                continue
            temperatures.append(float(fields[6]))
            clocks.append(float(fields[8].split()[0]))
            powers.append(float(fields[9].split()[0]))
    if not temperatures:
        raise SystemExit("no N=5 thermal snapshots found")
    return {
        "sample_count": len(temperatures),
        "temperature_c_min": min(temperatures),
        "temperature_c_max": max(temperatures),
        "sm_clock_mhz_min": min(clocks),
        "sm_clock_mhz_max": max(clocks),
        "power_w_min": min(powers),
        "power_w_max": max(powers),
    }


def q0_trajectory_counterfactual(root: Path):
    """Price thresholds on one fixed trajectory instead of mixing q-arm state histories."""
    observations = []
    paths = sorted(root.glob("c2-r*-q0-server.log"))
    if len(paths) != 5:
        raise SystemExit(f"q0 trajectory: expected five logs, got {len(paths)}")
    for path in paths:
        text = path.read_text(errors="replace")
        rows = [
            (float(q), hit == "true")
            for hit, q in re.findall(
                r"\[opti-controller\] resolve .* hit=(true|false) q=([0-9.]+)", text
            )
        ]
        if not rows:
            raise SystemExit(f"q0 trajectory: no resolved labels in {path}")
        observations.extend(rows)

    thresholds = {}
    for threshold in (0.0, 0.5, 0.7, 0.9):
        selected = [hit for q, hit in observations if q >= threshold]
        thresholds[f"{threshold:.1f}"] = {
            "selected": len(selected),
            "hits": sum(selected),
            "hit_rate": sum(selected) / len(selected) if selected else None,
        }
    q_values = [q for q, _ in observations]
    return {
        "note": "counterfactual thresholds over the fixed unconditional trajectory; not arm throughput",
        "observations": len(observations),
        "q_proxy_min": min(q_values),
        "q_proxy_median": statistics.median(q_values),
        "q_proxy_max": max(q_values),
        "thresholds": thresholds,
    }


def controller_resolution_summary(root: Path, arm: str):
    values = []
    paths = sorted(root.glob(f"c2-r*-{arm}-server.log"))
    if len(paths) != 5:
        raise SystemExit(f"{arm}: expected five server logs, got {len(paths)}")
    for path in paths:
        values.extend(
            float(value)
            for value in re.findall(r"resolution_ms=([0-9.]+)", path.read_text(errors="replace"))
        )
    return {
        "n": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "mean": statistics.mean(values) if values else None,
        "max": max(values) if values else None,
    }


def trace_anatomy(root: Path):
    pattern = re.compile(
        r"\[spec-anatomy\] per-round draft=([0-9.]+)ms "
        r"pp-verify=([0-9.]+)ms verify-accept=([0-9.]+)ms "
        r"commit-rollback=([0-9.]+)ms other=([0-9.]+)ms rounds=(\d+)"
    )
    fields = ("draft_ms", "pp_verify_ms", "verify_accept_ms", "commit_rollback_ms", "other_ms")
    result = {}
    telemetry = {row["arm"]: row for row in rows(root / "controller-telemetry.jsonl")
                 if row["label"].startswith("c2-trace-")}
    for arm in ("q0", "q07"):
        path = root / f"c2-trace-{arm}-server.log"
        matches = pattern.findall(path.read_text(errors="replace"))
        if not matches:
            raise SystemExit(f"trace anatomy: no rows in {path}")
        series = {field: [] for field in fields}
        round_counts = []
        for match in matches:
            for field, value in zip(fields, match[:5]):
                series[field].append(float(value))
            round_counts.append(int(match[5]))
        result[arm] = {
            "burst_count": len(matches),
            "rounds_total": sum(round_counts),
            "per_round_medians_ms": {
                field: statistics.median(values) for field, values in series.items()
            },
            "per_round_weighted_ms": {
                field: sum(value * rounds for value, rounds in zip(values, round_counts))
                / sum(round_counts)
                for field, values in series.items()
            },
            "controller": telemetry.get(arm),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    root = args.receipt

    points = [row for row in rows(root / "points.jsonl") if "-trace-" not in row["label"]]
    grouped = {arm: [] for arm in ARMS}
    for row in points:
        grouped[arm_from_label(row["label"])].append(row)
    for arm, arm_rows in grouped.items():
        if len(arm_rows) != 5:
            raise SystemExit(f"{arm}: expected N=5, got {len(arm_rows)}")
        if any(row["n_ok"] != 8 or row["n_err"] or row["n_shed"] for row in arm_rows):
            raise SystemExit(f"{arm}: non-clean load point")

    plain_median = statistics.median(row["agg_tok_s"] for row in grouped["plain"])
    serial_median = statistics.median(row["agg_tok_s"] for row in grouped["serial"])
    seam_median = statistics.median(row["agg_tok_s"] for row in grouped["seam"])
    arms = {}
    for arm in ARMS:
        values = [row["agg_tok_s"] for row in grouped[arm]]
        median = statistics.median(values)
        arms[arm] = {
            "n": len(values),
            "agg_tok_s_values": values,
            "agg_tok_s_median": median,
            "agg_tok_s_min": min(values),
            "agg_tok_s_max": max(values),
            "vs_plain_percent": (median / plain_median - 1.0) * 100.0,
            "vs_serial_percent": (median / serial_median - 1.0) * 100.0,
            "vs_seam_percent": (median / seam_median - 1.0) * 100.0,
        }

    telemetry_rows = [
        row for row in rows(root / "controller-telemetry.jsonl")
        if "-trace-" not in row["label"]
    ]
    telem_grouped = {arm: [] for arm in Q_ARMS}
    for row in telemetry_rows:
        telem_grouped[row["arm"]].append(row)
    count_fields = (
        "checks", "admits", "rejects", "hits", "misses", "reconciles", "tail_drains",
        "breaker_trips", "shadow_reject_hits", "shadow_reject_misses",
        "opportunity_labels", "wasted_draft_tokens", "shadow_draft_tokens",
    )
    for arm, arm_rows in telem_grouped.items():
        if len(arm_rows) != 5:
            raise SystemExit(f"{arm}: expected five telemetry rows, got {len(arm_rows)}")
        totals = {field: sum(row[field] for row in arm_rows) for field in count_fields}
        resolved = totals["hits"] + totals["misses"]
        opportunity_hits = totals["hits"] + totals["shadow_reject_hits"]
        totals.update({
            "admission_rate": totals["admits"] / totals["checks"],
            "admitted_hit_rate": totals["hits"] / resolved if resolved else None,
            "admitted_miss_rate": totals["misses"] / resolved if resolved else None,
            "opportunity_hit_rate": opportunity_hits / totals["opportunity_labels"],
            "resolution_ms_run_medians": [row["resolution_ms_median"] for row in arm_rows],
            "resolution_ms": controller_resolution_summary(root, arm),
        })
        arms[arm]["controller"] = totals

    result = {
        "protocol": {
            "concurrency": 2,
            "requests_per_point": 8,
            "max_tokens": 128,
            "warmup_requests": 2,
            "n": 5,
            "interleaved_single_lock_hold": True,
            "instrumented_trace_points_excluded": True,
        },
        "thermal_regime": thermal_summary(root),
        "q0_trajectory_counterfactual": q0_trajectory_counterfactual(root),
        "instrumented_trace_anatomy": trace_anatomy(root),
        "arms": arms,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
