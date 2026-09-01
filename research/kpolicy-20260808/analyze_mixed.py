#!/usr/bin/env python3
"""Validate and summarize the before/after mixed-workload receipt."""

import argparse
import collections
import json
import pathlib
import statistics


def med(values):
    return statistics.median(values)


def rng(values):
    return f"{min(values):.2f}-{max(values):.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("points")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.points) if line.strip()]
    grouped = collections.defaultdict(list)
    errors = []
    for row in rows:
        grouped[row["arm"]].append(row)
        if row["requests"] != 8:
            errors.append(f"{row['arm']} r{row['rep']}: requests={row['requests']}")
        names = {request["name"] for request in row["rows"]}
        expected = {
            "seq-short",
            "seq-long",
            "cached-setup",
            "seq-cached",
            "wave-short-a",
            "wave-cached",
            "wave-long",
            "wave-short-b",
        }
        if names != expected:
            errors.append(f"{row['arm']} r{row['rep']}: request set mismatch")
        for request in row["rows"]:
            cached = int(request["cached_tokens"])
            if request["name"] in {"seq-cached", "wave-cached"}:
                if cached < 1024:
                    errors.append(
                        f"{row['arm']} r{row['rep']} {request['name']}: cached={cached}"
                    )
            elif cached != 0:
                errors.append(
                    f"{row['arm']} r{row['rep']} {request['name']}: unexpected cached={cached}"
                )

    for arm in ("before", "after"):
        reps = sorted(int(row["rep"]) for row in grouped.get(arm, []))
        expected = list(range(1, args.reps + 1))
        if reps != expected:
            errors.append(f"{arm}: reps={reps}, expected={expected}")
    if errors:
        raise SystemExit("invalid mixed receipt:\n  " + "\n  ".join(errors))

    lines = [
        "# Mixed-workload before/after",
        "",
        f"N={args.reps} independent server boots per arm, rep-major alternating order.",
        "Each rep includes cold-short, cold-long, cached-long setup+continuation, then a "
        "staggered c=4 wave with two short, one cold-long, and one cached-long request.",
        "",
        "| arm | aggregate tok/s median (range) | c=4 wave tok/s median (range) | "
        "workload wall median | cached tok median |",
        "|---|---:|---:|---:|---:|",
    ]
    metrics = {}
    for arm in ("before", "after"):
        cell = grouped[arm]
        aggregate = [float(row["agg_tok_s"]) for row in cell]
        wave = [float(row["wave_tok_s"]) for row in cell]
        wall = [float(row["workload_wall_s"]) for row in cell]
        cached = [int(row["cached_tokens_total"]) for row in cell]
        metrics[arm] = {
            "aggregate": med(aggregate),
            "wave": med(wave),
            "wall": med(wall),
        }
        lines.append(
            f"| {arm} | {med(aggregate):.2f} ({rng(aggregate)}) | "
            f"{med(wave):.2f} ({rng(wave)}) | {med(wall):.3f}s | "
            f"{int(med(cached))} |"
        )

    aggregate_delta = metrics["after"]["aggregate"] / metrics["before"]["aggregate"] - 1.0
    wave_delta = metrics["after"]["wave"] / metrics["before"]["wave"] - 1.0
    wall_delta = metrics["after"]["wall"] / metrics["before"]["wall"] - 1.0
    lines.extend([
        "",
        f"- Aggregate delta: {aggregate_delta * 100:+.2f}%.",
        f"- c=4 wave delta: {wave_delta * 100:+.2f}%.",
        f"- Workload wall delta: {wall_delta * 100:+.2f}%.",
        "",
        "## Per-request modes",
        "",
    ])
    for arm in ("before", "after"):
        counts = collections.Counter()
        for row in grouped[arm]:
            for request in row["rows"]:
                counts[(request["name"], request["mode"])] += 1
        rendered = ", ".join(
            f"{name}:{mode}={count}"
            for (name, mode), count in sorted(counts.items())
        )
        lines.append(f"- {arm}: {rendered}")

    text = "\n".join(lines) + "\n"
    if args.out:
        pathlib.Path(args.out).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
