"""Optimistic full-offer stopping bound and data-selected fixed C controls."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

from costs import read_costs
from learn import loop_candidate, read_rounds


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_sessions(roots):
    sessions, sources, excluded = [], [], []
    for root in roots:
        included = []
        for turn in range(1, 9):
            trace = root / f"turn-{turn}.confidence.tsv"
            tape = root / f"turn-{turn}.output.ids"
            ids = [int(token) for token in tape.read_text().split()]
            sources.append({"trace": str(trace), "sha256": sha(trace),
                            "output_sha256": sha(tape)})
            if loop_candidate(ids):
                excluded.append({"session": root.name, "turn": turn})
                continue
            for row in read_rounds(trace):
                if row["eligible"]:
                    if row["drafted"] != 3:
                        raise ValueError("oracle requires uncensored K=3 offers")
                    included.append(row)
        if not included:
            raise ValueError("calibration session has no eligible full offers")
        sessions.append(included)
    if len(sessions) < 2:
        raise ValueError("oracle needs independent calibration conversations")
    return sessions, sources, excluded


def pooled(rows, depths, costs):
    tokens = sum(min(row["accepted"], depth) + 1
                 for row, depth in zip(rows, depths))
    ns = sum(costs[depth] for depth in depths)
    return tokens / ns


def hindsight_depths(rows, costs):
    rate = pooled(rows, [3] * len(rows), costs)
    for _ in range(128):
        depths = [
            max((1, 2, 3), key=lambda k: (
                min(row["accepted"], k) + 1 - rate * costs[k], -k
            ))
            for row in rows
        ]
        update = pooled(rows, depths, costs)
        if math.isclose(update, rate, rel_tol=1e-13):
            return update, depths
        rate = update
    raise ValueError("hindsight cost oracle did not converge")


def observed_cutpoints(values):
    ordered = sorted(set(values))
    if not ordered:
        raise ValueError("no observed confidence cutpoints")
    # The number of empirical quantiles grows with information, never
    # a frozen list of candidate confidence values.
    stride = max(1, math.isqrt(len(ordered)))
    return sorted({0.0, 1.0, *(ordered[::stride])})


def fixed_depths(rows, cutoffs):
    first, second = cutoffs
    return [
        1 if row["q"][0] < first else
        2 if row["q"][1] < second else 3
        for row in rows
    ]


def best_fixed(rows, costs):
    first = observed_cutpoints(row["q"][0] for row in rows)
    second = observed_cutpoints(row["q"][1] for row in rows)
    best = None
    for c0 in first:
        for c1 in second:
            depths = fixed_depths(rows, (c0, c1))
            rate = pooled(rows, depths, costs)
            item = (rate, -(c0 + c1), c0, c1, depths)
            if best is None or item[:2] > best[:2]:
                best = item
    return {
        "cutoffs": [best[2], best[3]], "rate_per_ns": best[0],
        "depth_counts": {str(k): best[4].count(k) for k in (1, 2, 3)},
        "candidate_counts": [len(first), len(second)],
    }


def report(sessions, costs, sources, excluded):
    rows = [row for session in sessions for row in session]
    baseline = pooled(rows, [3] * len(rows), costs)
    upper, choices = hindsight_depths(rows, costs)
    fixed = best_fixed(rows, costs)
    rng = random.Random(20760031)
    bootstrap = []
    for _ in range(2048):
        sampled = [
            row for _ in sessions
            for row in sessions[rng.randrange(len(sessions))]
        ]
        base = pooled(sampled, [3] * len(sampled), costs)
        oracle_rate, _ = hindsight_depths(sampled, costs)
        selected = pooled(sampled, fixed_depths(sampled, fixed["cutoffs"]), costs)
        bootstrap.append({
            "oracle_gain_percent": 100 * (oracle_rate / base - 1),
            "selected_fixed_gain_percent": 100 * (selected / base - 1),
        })
    bands = {}
    for name in ("oracle_gain_percent", "selected_fixed_gain_percent"):
        values = sorted(row[name] for row in bootstrap)
        bands[name] = [values[51], values[1996]]
    return {
        "status": "optimistic-calibration-oracle",
        "scope": "uncensored eligible full K=3 native rounds; hindsight is a bound, not an executable online policy",
        "sessions": len(sessions), "rounds": len(rows),
        "baseline_rate_per_ns": baseline,
        "oracle_rate_per_ns": upper,
        "oracle_gain_percent": 100 * (upper / baseline - 1),
        "oracle_depth_counts": {str(k): choices.count(k) for k in (1, 2, 3)},
        "best_calibrated_fixed": fixed,
        "best_fixed_gain_percent": 100 * (fixed["rate_per_ns"] / baseline - 1),
        "conversation_bootstrap_95_percent": bands,
        "matched_loop_exclusions": excluded,
        "source_files": sources,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, action="append", required=True)
    parser.add_argument("--costs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    costs = read_costs(args.costs)
    sessions, sources, excluded = read_sessions(args.session)
    result = report(sessions, costs, sources, excluded)
    result["costs_sha256"] = sha(args.costs)
    with args.out.open("x") as output:
        output.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "oracle_gain_percent": result["oracle_gain_percent"],
        "best_fixed": result["best_calibrated_fixed"],
    }))


if __name__ == "__main__":
    main()
