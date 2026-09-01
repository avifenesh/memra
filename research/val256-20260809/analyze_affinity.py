#!/usr/bin/env python3
"""Summarize the box1 256k plain-affinity A/B receipt."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics


def requests(path: pathlib.Path) -> list[dict]:
    return [
        row
        for line in path.read_text().splitlines()
        if line.strip() and (row := json.loads(line)).get("type") == "request"
    ]


def key(row: dict) -> tuple[str, int, int]:
    return (row["phase"], row["phase_index"], row["request_index"])


def slope(xs: list[float], ys: list[float]) -> float:
    xbar = statistics.mean(xs)
    ybar = statistics.mean(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--learning-turns", type=int, default=2)
    parser.add_argument("--deep-token-floor", type=int, default=32768)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    arms = {
        name: {key(row): row for row in requests(args.root / name / "requests.jsonl")}
        for name in ("on-1", "off-1", "on-2", "off-2", "on-3", "off-3")
    }
    on_names = ("on-1", "on-2", "on-3")
    off_names = ("off-1", "off-2", "off-3")
    record_rows = requests(args.root / "record-on" / "requests.jsonl")
    record_sequential = [row for row in record_rows if row.get("phase") == "sequential"]
    rewritten_turns = sum(bool(row.get("history_rewritten")) for row in record_sequential)
    baseline_keys = set(arms["on-1"])
    failures: list[str] = []
    if rewritten_turns != len(record_sequential):
        failures.append(
            f"record workload rewrote {rewritten_turns}/{len(record_sequential)} sequential turns"
        )
    if any(set(rows) != baseline_keys for rows in arms.values()):
        failures.append("request keys differ across A/B repetitions")

    nondeterministic = []
    for item in sorted(baseline_keys):
        hashes = {arms[name][item].get("text_sha256") for name in on_names}
        if len(hashes) != 1:
            nondeterministic.append(list(item))
    if nondeterministic:
        failures.append(
            f"affinity-ON output differed across servers on {len(nondeterministic)} requests"
        )

    seq_keys = sorted(item for item in baseline_keys if item[0] == "sequential")
    turns = []
    for item in seq_keys:
        prompt_tokens = [arms[name][item].get("prompt_tokens") for name in on_names + off_names]
        if any(not isinstance(value, int) for value in prompt_tokens):
            failures.append(f"missing prompt token count at {item}")
            continue
        turns.append(
            {
                "turn": item[1],
                "prompt_tokens": int(statistics.median(prompt_tokens)),
                "on_ttft_s_median_n3": round(
                    statistics.median(arms[name][item]["ttft_s"] for name in on_names), 6
                ),
                "off_ttft_s_median_n3": round(
                    statistics.median(arms[name][item]["ttft_s"] for name in off_names), 6
                ),
                "on_cached_tokens_median_n3": int(
                    statistics.median((arms[name][item].get("cached_tokens") or 0) for name in on_names)
                ),
                "off_cached_tokens_median_n3": int(
                    statistics.median((arms[name][item].get("cached_tokens") or 0) for name in off_names)
                ),
            }
        )

    min_prompt = min((turn["prompt_tokens"] for turn in turns), default=0)
    max_prompt = max((turn["prompt_tokens"] for turn in turns), default=0)
    if max_prompt <= args.deep_token_floor:
        failures.append(
            f"conversation never crossed {args.deep_token_floor} prompt tokens (max={max_prompt})"
        )

    fitted = turns[args.learning_turns :]
    xs = [float(turn["prompt_tokens"]) for turn in fitted]
    on_ys = [float(turn["on_ttft_s_median_n3"]) for turn in fitted]
    off_ys = [float(turn["off_ttft_s_median_n3"]) for turn in fitted]
    on_slope = slope(xs, on_ys) if len(xs) >= 2 else 0.0
    off_slope = slope(xs, off_ys) if len(xs) >= 2 else 0.0
    ratio = on_slope / off_slope if off_slope > 0 else None
    if off_slope <= 0:
        failures.append("affinity-OFF TTFT slope was not positive")
    elif on_slope > off_slope * 0.5:
        failures.append(
            "affinity-ON TTFT slope did not collapse by at least 2x versus affinity-OFF"
        )

    rewinds = {}
    for name in on_names + off_names:
        metrics = json.loads((args.root / name / "metrics-final.json").read_text())
        rewinds[name] = metrics.get("plain_affinity_rewinds")
    if any(not isinstance(rewinds[name], int) or rewinds[name] <= 0 for name in on_names):
        failures.append("one or more affinity-ON servers recorded no plain affinity rewind")
    if any(rewinds[name] != 0 for name in off_names):
        failures.append("one or more affinity-OFF servers unexpectedly recorded a rewind")

    receipt = {
        "n_per_arm": 3,
        "arm_order": ["on-1", "off-1", "on-2", "off-2", "on-3", "off-3"],
        "sequential_turns": len(seq_keys),
        "history_rewritten_turns": rewritten_turns,
        "deep_token_floor": args.deep_token_floor,
        "min_prompt_tokens": min_prompt,
        "max_prompt_tokens": max_prompt,
        "deterministic_across_on_servers": not nondeterministic,
        "nondeterministic_requests": nondeterministic,
        "plain_affinity_rewinds": rewinds,
        "learning_turns_excluded": args.learning_turns,
        "on_slope_ms_per_added_prompt_token": round(on_slope * 1000.0, 6),
        "off_slope_ms_per_added_prompt_token": round(off_slope * 1000.0, 6),
        "on_over_off_slope_ratio": None if ratio is None else round(ratio, 6),
        "slope_collapse_at_least_2x": off_slope > 0 and on_slope <= off_slope * 0.5,
        "turns": turns,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
