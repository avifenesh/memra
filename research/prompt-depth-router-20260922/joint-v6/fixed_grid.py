"""Measure static MTP draft K, draft length D and after-offer C."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import struct

from run_native import TOP_K, input_manifests, old_code_probes, run_one, save, sha, v4


def entries(manifest):
    return manifest["groups"]["calibration"] + manifest["groups"]["heldout"][:3]


def eligible_k(root):
    qualifier = json.loads((root / "qualification-result.json").read_text())
    return [k for k in TOP_K if qualifier["eligible"][str(k)]]


def depth_grid(args, old):
    allowed = eligible_k(args.out)
    if 20 not in allowed:
        raise ValueError("draft top-k=20 failed qualification")
    records = []
    for index, entry in enumerate(entries(old)):
        arms = [
            (f"topk{k}-d{d}-c0", k, d)
            for k in allowed for d in (1, 2, 4)
        ]
        order = arms[index % len(arms):] + arms[:index % len(arms)]
        if index % 2:
            order.reverse()
        for label, k, d in order:
            records.append(run_one(
                args, entry, args.development, f"depth-grid-{index}",
                label, k, f"fixed:{d}", d,
            ))
    save(args.out / "depth-grid-summary.json", {
        "schema": 1, "phase": "depth-grid", "records": records,
    })
    return records


def q_values(root, k):
    development = json.loads((root / "development-summary.json").read_text())
    q = {0: [], 1: []}
    sources = {}
    for row in development["records"]:
        if row["draft_top_k"] != k or row["arm"] != "explore-d":
            continue
        session = root / row["name"]
        sources[row["name"]] = [
            (session / f"turn-{turn}.confidence.tsv").stat().st_size
            for turn in range(1, 9)
        ]
        for turn in range(1, 9):
            for offered in v4.table(session / f"turn-{turn}.confidence.tsv"):
                if offered["eligible"] != "true":
                    continue
                bits = offered["q_bits"].split(",")
                if len(bits) != int(offered["drafted"]):
                    raise ValueError("sampled confidence receipt changed")
                for position in q:
                    if position < len(bits):
                        value = struct.unpack("<f", struct.pack("<I", int(bits[position])))[0]
                        if not math.isfinite(value) or not 0 <= value <= 1:
                            raise ValueError("nonfinite sampled confidence candidate")
                        q[position].append(value)
    if len(sources) != 6 or any(len(rows) < 150 for rows in q.values()):
        raise ValueError("fixed C quantiles lack randomized top-k exposure")
    return q, sources


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def summarize_rows(rows, old, root):
    by_variant = defaultdict(list)
    for record in rows:
        by_variant[record["variant"]].append(record)
    metrics = {}
    topics = entries(old)
    for variant, group in by_variant.items():
        if len(group) != 6:
            raise ValueError("static C/K/D arm lacks six complete conversations")
        group_by_index = {
            next(int(part) for part in row["name"].split("-") if part.isdigit()): row
            for row in group
        }
        if set(group_by_index) != set(range(6)):
            raise ValueError("static grid lacks a matched development topic")
        tokens = sum(row["tokens"] for row in group)
        seconds = sum(row["seconds"] for row in group)
        functional = sum(
            old_code_probes(root / group_by_index[index]["name"], topics[index])
            for index in range(3, 6)
        )
        looped = sum(row["loops"] for row in group)
        metrics[variant] = {
            "tokens": tokens, "seconds": seconds,
            "tok_s": tokens / seconds if looped == 0 else None,
            "format_pass": sum(row["format"] for row in group),
            "functional_pass_last_three_topics": functional,
            "loops": looped,
        }
    return metrics


def choose_best_depth(args, old):
    development = json.loads((args.out / "development-summary.json").read_text())
    depth = json.loads((args.out / "depth-grid-summary.json").read_text())
    all_rows = [
        row for row in development["records"]
        if isinstance(row["draft_top_k"], int) and row["arm"] == "fixed:3"
    ] + depth["records"]
    metrics = summarize_rows(all_rows, old, args.out)
    allowed = [
        name for name, score in metrics.items()
        if int(name.split("-")[0].removeprefix("topk")) in eligible_k(args.out)
        and score["format_pass"] == 48
        and score["functional_pass_last_three_topics"] == 24
        and score["loops"] == 0
    ]
    if "topk20-d3-c0" not in allowed:
        raise ValueError("recommended fixed K20/D3 fails development quality")
    winner = max(allowed, key=lambda name: metrics[name]["tok_s"])
    save(args.out / "depth-grid-analysis.json", {
        "schema": 1, "status": "quality-gated static depth selection",
        "metrics": metrics, "best": winner,
    })
    return winner


def confidence_grid(args, old):
    chosen = json.loads((args.out / "depth-grid-analysis.json").read_text())["best"]
    k = int(chosen.split("-")[0].removeprefix("topk"))
    q, sources = q_values(args.out, k)
    low0, low1 = quantile(q[0], 0.25), quantile(q[1], 0.25)
    mid0, mid1 = quantile(q[0], 0.5), quantile(q[1], 0.5)
    cutoffs = {
        "low-both": [low0, low1],
        "median-both": [mid0, mid1],
        "low-first": [low0, 0.0],
        "low-second": [0.0, low1],
    }
    save(args.out / "fixed-confidence-candidates.json", {
        "schema": 1,
        "draft_top_k": k,
        "basis": "quarter/median empirical q from six C0 randomized-D conversations",
        "source_trace_bytes": sources,
        "cutoffs": cutoffs,
    })
    records = []
    for index, entry in enumerate(entries(old)):
        variants = list(cutoffs)
        order = variants[index % len(variants):] + variants[:index % len(variants)]
        if index % 2:
            order.reverse()
        for variant in order:
            pair = cutoffs[variant]
            records.append(run_one(
                args, entry, args.development, f"confidence-grid-{index}",
                f"topk{k}-d3-c-{variant}", k, "fixed-c3", 3,
                extra=(f"confidence-fixed={pair[0]:.9g},{pair[1]:.9g}",),
            ))
    save(args.out / "confidence-grid-summary.json", {
        "schema": 1, "phase": "confidence-grid", "records": records,
    })
    return records


def choose_best_fixed(args, old):
    depth = json.loads((args.out / "depth-grid-analysis.json").read_text())
    confidence = json.loads((args.out / "confidence-grid-summary.json").read_text())
    cm = summarize_rows(confidence["records"], old, args.out)
    metrics = {**depth["metrics"], **cm}
    allowed = [
        name for name, score in metrics.items()
        if int(name.split("-")[0].removeprefix("topk")) in eligible_k(args.out)
        and score["format_pass"] == 48
        and score["functional_pass_last_three_topics"] == 24
        and score["loops"] == 0
    ]
    winner = max(allowed, key=lambda name: metrics[name]["tok_s"])
    save(args.out / "selected-fixed.json", {
        "schema": 1, "status": "quality-gated measured C/K/D grid",
        "best": winner, "metrics": metrics,
        "confidence_candidates_sha256": sha(
            args.out / "fixed-confidence-candidates.json"
        ),
    })
    return winner


def main():
    parser = argparse.ArgumentParser()
    for name in ("binary", "model", "development", "fresh", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument(
        "--phase",
        choices=("depth", "depth-analysis", "confidence", "fixed-analysis"),
        required=True,
    )
    args = parser.parse_args()
    for name in ("binary", "model", "development", "fresh", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(exist_ok=True)
    old, _ = input_manifests(args)
    result = {
        "depth": lambda: len(depth_grid(args, old)),
        "depth-analysis": lambda: choose_best_depth(args, old),
        "confidence": lambda: len(confidence_grid(args, old)),
        "fixed-analysis": lambda: choose_best_fixed(args, old),
    }[args.phase]()
    print(json.dumps({"phase": args.phase, "result": result}))


if __name__ == "__main__":
    main()
