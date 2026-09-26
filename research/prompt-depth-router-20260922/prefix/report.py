"""Paired fixed-three comparisons from complete audited native requests."""

import argparse
import collections
import json
from pathlib import Path
import random
import statistics

from audit import save, sha
from pipeline import ARMS, orders


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (position - lo)


def comparisons(matched):
    if not matched:
        return {"pairs": 0, "arms": {}}
    result = {"pairs": len(matched), "arms": {}}
    baseline = [group["fixed3"] for group in matched]
    base_rate = sum(row["output_tokens"] for row in baseline) / sum(row["elapsed_s"] for row in baseline)
    for label in ARMS:
        rows = [group[label] for group in matched]
        tokens, seconds = sum(r["output_tokens"] for r in rows), sum(r["elapsed_s"] for r in rows)
        gains = [
            100 * ((row["output_tokens"] / row["elapsed_s"])
                   / (base["output_tokens"] / base["elapsed_s"]) - 1)
            for row, base in zip(rows, baseline)
        ]
        result["arms"][label] = {
            "tokens": tokens, "complete_request_seconds": seconds, "tokens_per_second": tokens / seconds,
            "pooled_gain_percent": 100 * (tokens / seconds / base_rate - 1),
            "paired_gain_percent": gains, "median_paired_gain_percent": statistics.median(gains),
            "wins": sum(gain > 0 for gain in gains),
            "latency_ratio_to_fixed3": seconds / sum(r["elapsed_s"] for r in baseline),
            "output_token_ratio_to_fixed3": tokens / sum(r["output_tokens"] for r in baseline),
            "format_covered": sum(row["format"]["requested_format_covered"] for row in rows),
            "final_tokens_with_bytes": sum(row["format"]["final_tokens_with_bytes"] for row in rows),
            "nonfinal_tokens_with_bytes": sum(row["format"]["nonfinal_tokens_with_bytes"] for row in rows),
            "k_counts": dict(collections.Counter(row["k"] for row in rows)),
            "prediction_counts": dict(collections.Counter(row["prediction"] for row in rows)),
            "routing_us": {
                "median": statistics.median(r["routing_ns"] for r in rows) / 1000,
                "p95": quantile([r["routing_ns"] for r in rows], .95) / 1000,
                "max": max(r["routing_ns"] for r in rows) / 1000,
            },
        }
        # Resample whole matched scenarios, preserving both policies in a pair.
        # These are pointwise exploratory intervals, not a multiple-test gate.
        if len(rows) >= 3 and label != "fixed3":
            rng = random.Random(20730922)
            gains = []
            for _ in range(5000):
                indices = [rng.randrange(len(rows)) for _ in rows]
                adapted_rate = (sum(rows[i]["output_tokens"] for i in indices)
                                / sum(rows[i]["elapsed_s"] for i in indices))
                fixed_rate = (sum(baseline[i]["output_tokens"] for i in indices)
                              / sum(baseline[i]["elapsed_s"] for i in indices))
                gains.append(100 * (adapted_rate / fixed_rate - 1))
            result["arms"][label]["paired_bootstrap_95_percent"] = [
                quantile(gains, .025), quantile(gains, .975)]
        else:
            result["arms"][label]["paired_bootstrap_95_percent"] = None
    return result


def report(root):
    native = root / "native"
    state = json.loads((native / "status.json").read_text())
    freeze = json.loads((native / "FREEZE.json").read_text())
    if state["status"] != "completed":
        raise ValueError("native matrix is incomplete; no final performance report")
    if freeze["arms"] != ARMS or freeze["orders"] != orders():
        raise ValueError("native arm registration changed")
    if sha(root / "workloads/manifest.json") != freeze["workloads_sha256"]:
        raise ValueError("frozen workload manifest changed")
    models = {}
    for family in ("qwen", "gemma"):
        cells = collections.defaultdict(list)
        exclusions = []
        for scenario in range(6):
            path = native / family / "scored"
            records = {label: json.loads((path / f"{scenario:02}-{label}.audit.json").read_text())
                       for label in ARMS}
            for label, record in records.items():
                if (record["arm"] != ARMS[label] or record["max_new"] != state["budgets"][family]
                        or record["gate"] or record["family"] != family):
                    raise ValueError("wrong scored policy, budget, or model")
                exit_row = json.loads((path / f"{scenario:02}-{label}.exit.json").read_text())
                if exit_row["returncode"] != 0 or exit_row["contamination"]:
                    raise ValueError("failed or contaminated scored arm")
            for turn in range(1, 9):
                matched = {label: record["requests"][turn - 1] for label, record in records.items()}
                kinds = {(row["kind"], row["length_target"]) for row in matched.values()}
                if len(kinds) != 1 or len({record["seed"] for record in records.values()}) != 1:
                    raise ValueError("matched request identities differ")
                key = next(iter(kinds))
                if any(row["loop"] for row in matched.values()):
                    exclusions.append({"scenario": scenario, "turn": turn, "kind": key[0],
                                       "length_target": key[1], "scope": "all four arms"})
                else:
                    cells[key].append(matched)
        model = {"selected_max_new": state["budgets"][family], "loop_exclusions": exclusions, "cells": {}}
        for kind in ("prose", "code"):
            for length in (256, 1024, 4096, 16384):
                matched = cells[kind, length]
                covered = [group for group in matched
                           if all(row["format"]["requested_format_covered"] for row in group.values())]
                model["cells"][f"{kind}-{length}"] = {
                    "requested_format": comparisons(matched),
                    "all_arms_format_covered": comparisons(covered),
                }
        models[family] = model
    return {
        "status": "measured", "models": models, "freeze": freeze,
        "identity": json.loads((native / "identity.json").read_text()),
        "scope": "single-GPU native independent requests; no HTTP/concurrency or continuous-session claim",
        "interpretation": "primary requested-format rows keep capped/format-mismatched outputs; covered matched subset is separate; six scenarios per cell are exploratory",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.root)
    save(args.out, result)
    print(json.dumps({"status": result["status"], "models": list(result["models"])}))
