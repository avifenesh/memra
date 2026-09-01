#!/usr/bin/env python3
"""Validate and summarize the request-conditioned K matrix."""

import argparse
import collections
import json
import pathlib
import statistics


def median(values):
    return statistics.median(values)


def span(values):
    return f"{min(values):.2f}-{max(values):.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("points")
    parser.add_argument("--expected-reps", type=int, default=3)
    parser.add_argument("--expect", action="append", default=[],
                        help="required model:comma-separated-Ks cell set")
    parser.add_argument("--classes", default="cold-short,cold-long,cached-long")
    parser.add_argument("--thermal-note")
    parser.add_argument("--out")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.points) if line.strip()]
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["class"], int(row["k"]))].append(row)

    errors = []
    expected_classes = [value for value in args.classes.split(",") if value]
    expected_keys = set()
    for spec in args.expect:
        model, separator, values = spec.partition(":")
        if not separator or not model or not values:
            errors.append(f"invalid --expect value {spec!r}")
            continue
        for prompt_class in expected_classes:
            for value in values.split(","):
                expected_keys.add((model, prompt_class, int(value)))
    if expected_keys:
        missing = sorted(expected_keys - set(grouped))
        unexpected = sorted(set(grouped) - expected_keys)
        if missing:
            errors.append(f"missing cells: {missing}")
        if unexpected:
            errors.append(f"unexpected cells: {unexpected}")
    elif not grouped:
        errors.append("matrix has no rows")

    for key, cell in sorted(grouped.items()):
        reps = sorted(int(row["rep"]) for row in cell)
        expected = list(range(1, args.expected_reps + 1))
        if reps != expected:
            errors.append(f"{key}: reps {reps}, expected {expected}")
        for row in cell:
            if row["class"] == "cached-long" and row["cached_tokens"] < 1024:
                errors.append(f"{row['label']}: cached_tokens={row['cached_tokens']}")
            if row["class"] != "cached-long" and row["cached_tokens"] != 0:
                errors.append(f"{row['label']}: unexpected cache hit")
            if row["k"] == 0 and row["spec"] is not None:
                errors.append(f"{row['label']}: K=0 has spec usage")
            if row["k"] > 0 and row["spec"] is None:
                errors.append(f"{row['label']}: positive K has no spec usage")
    if errors:
        raise SystemExit("incomplete or invalid matrix:\n  " + "\n  ".join(errors))

    lines = [
        "# K-policy matrix summary",
        "",
        f"Rows: {len(rows)}. N={args.expected_reps} independent server boots per cell.",
        "Primary rate is completion tokens divided by client-observed request wall time.",
    ]
    if args.thermal_note:
        lines.append(f"Thermal regime: {args.thermal_note}")
    lines.extend([
        "",
        "| model | class | K | net tok/s median (range) | server tok/s median | "
        "acceptance median | prompt tok | cached tok |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for (model, prompt_class, k), cell in sorted(grouped.items()):
        net = [float(row["net_tok_s"]) for row in cell]
        server = [float(row["server_tok_s"]) for row in cell]
        accepts = [
            float(row["acceptance_rate"])
            for row in cell
            if row["acceptance_rate"] is not None
        ]
        prompt_tokens = [int(row["prompt_tokens"]) for row in cell]
        cached_tokens = [int(row["cached_tokens"]) for row in cell]
        acceptance = f"{median(accepts) * 100:.2f}%" if accepts else "plain"
        lines.append(
            f"| {model} | {prompt_class} | {k} | {median(net):.2f} "
            f"({span(net)}) | {median(server):.2f} | {acceptance} | "
            f"{int(median(prompt_tokens))} | {int(median(cached_tokens))} |"
        )

    lines.extend([
        "",
        "## Per-class ordering",
        "",
    ])
    by_shape = collections.defaultdict(list)
    for (model, prompt_class, k), cell in grouped.items():
        by_shape[(model, prompt_class)].append(
            (median([float(row["net_tok_s"]) for row in cell]), k)
        )
    for (model, prompt_class), values in sorted(by_shape.items()):
        ordering = ", ".join(
            f"K={k} {rate:.2f}" for rate, k in sorted(values, reverse=True)
        )
        lines.append(f"- {model} {prompt_class}: {ordering}")

    text = "\n".join(lines) + "\n"
    if args.out:
        pathlib.Path(args.out).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
