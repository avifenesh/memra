"""Update a live Qwen K=3 confidence policy from verified native rounds.

The only cutoff constants are probability-domain endpoints, 0 and 1.
All interior C values come from observed proposal confidence values.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import time


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def loop_candidate(ids):
    for end in sorted({len(ids), *range(512, len(ids) + 1, 256)}):
        for period in range(1, 129):
            repeated = max(4, (256 + period - 1) // period)
            size = repeated * period
            if end >= size and ids[end-size:end] == ids[end-period:end] * repeated:
                return {"end_token": end, "period": period, "repeated_tokens": size}
    return None


def read_rounds(path):
    rows = []
    with Path(path).open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {
            "round", "drafted", "accepted_prefix", "emitted",
            "elapsed_ns", "eligible", "q_bits",
        }
        if set(reader.fieldnames or ()) != required:
            raise ValueError("confidence trace has another schema")
        for row in reader:
            q = [
                struct.unpack("<f", struct.pack("<I", int(bits)))[0]
                for bits in row["q_bits"].split(",")
            ]
            drafted, accepted = int(row["drafted"]), int(row["accepted_prefix"])
            elapsed, emitted = int(row["elapsed_ns"]), int(row["emitted"])
            if (
                not 1 <= drafted <= 3 or len(q) != drafted
                or not 0 <= accepted <= drafted or not 1 <= emitted <= drafted + 1
                or elapsed <= 0 or row["eligible"] not in ("true", "false")
                or any(not math.isfinite(p) or not 0 <= p <= 1 for p in q)
            ):
                raise ValueError("invalid offered sampled confidence round")
            rows.append({
                "round": int(row["round"]), "q": q, "accepted": accepted,
                "drafted": drafted, "emitted": emitted, "elapsed_ns": elapsed,
                "eligible": row["eligible"] == "true",
            })
    if not rows or len({row["round"] for row in rows}) != len(rows):
        raise ValueError("confidence trace is empty or duplicates a round")
    return rows


def read_costs(path):
    data = json.loads(Path(path).read_text())
    if data.get("schema") != 1 or set(data.get("round_ns", {})) != {"1", "2", "3"}:
        raise ValueError("measured K-cost table has another schema")
    costs = {int(k): float(v) for k, v in data["round_ns"].items()}
    if any(not math.isfinite(v) or v <= 0 for v in costs.values()):
        raise ValueError("invalid measured K cost")
    if not costs[1] < costs[2] < costs[3]:
        raise ValueError("K-cost table must increase after its measured uncertainty check")
    return costs


def eligible_labels(rounds, position):
    return [
        (row["q"][position], int(row["accepted"] > position))
        for row in rounds
        if row["eligible"] and row["drafted"] > position
        and row["accepted"] >= position
    ]


def isotonic_blocks(samples):
    """Pool adjacent q ranges until observed acceptance is nondecreasing."""
    ordered = sorted(samples)
    blocks = []
    for q, accepted in ordered:
        if blocks and blocks[-1]["high"] == q:
            blocks[-1]["count"] += 1
            blocks[-1]["success"] += accepted
        else:
            blocks.append({
                "low": q, "high": q, "count": 1, "success": accepted
            })
        while len(blocks) > 1:
            left, right = blocks[-2:]
            if left["success"] * right["count"] <= right["success"] * left["count"]:
                break
            blocks[-2:] = [{
                "low": left["low"], "high": right["high"],
                "count": left["count"] + right["count"],
                "success": left["success"] + right["success"],
            }]
    return blocks


def cutoff_from_evidence(current, following, rate, marginal_cost):
    if not current or not following:
        return 0.0, {"reason": "uncensored evidence needed"}
    next_success = sum(accepted for _, accepted in following)
    next_count = len(following)
    # Jeffreys posterior has no acceptance-band or C hyperparameter.
    next_accept = (next_success + 0.5) / (next_count + 1)
    required = rate * marginal_cost / next_accept
    blocks = isotonic_blocks(current)
    cutoff = 1.0
    for index, block in enumerate(blocks):
        if block["success"] / block["count"] >= required:
            cutoff = 0.0 if index == 0 else (
                blocks[index - 1]["high"] + block["low"]
            ) / 2
            break
    return cutoff, {
        "eligible_current": len(current), "eligible_next": next_count,
        "next_accept_posterior": next_accept, "required_current_accept": required,
        "isotonic_blocks": len(blocks),
    }


def decide(rounds, costs, previous):
    eligible = [row for row in rounds if row["eligible"]]
    if not eligible:
        return [0.0, 0.0], {"reason": "no eligible verified rounds", "probe": False}
    rate = sum(row["emitted"] for row in eligible) / sum(
        row["elapsed_ns"] for row in eligible
    )
    evidence = [eligible_labels(eligible, position) for position in range(3)]
    cutoffs, details = [], []
    for position in (0, 1):
        cutoff, detail = cutoff_from_evidence(
            evidence[position], evidence[position + 1], rate,
            costs[position + 2] - costs[position + 1],
        )
        cutoffs.append(cutoff)
        details.append(detail)
    # A full-offer turn is an uncertainty-driven probe, never a periodic schedule.
    probe = False
    if any(previous) and any(cutoffs):
        current_cost = sum(costs[row["drafted"]] for row in eligible) / len(eligible)
        extra_cost = max(0.0, costs[3] - current_cost)
        uncertainty_time = 0.0
        for next_position in (1, 2):
            n = len(evidence[next_position])
            if not n:
                uncertainty_time = math.inf
                break
            accepted = sum(value for _, value in evidence[next_position])
            a, b = accepted + 0.5, n - accepted + 0.5
            variance = a * b / ((a + b) ** 2 * (a + b + 1))
            uncertainty_time = max(uncertainty_time, math.sqrt(variance) / rate)
        probe = uncertainty_time > extra_cost
    return ([0.0, 0.0] if probe else cutoffs), {
        "observed_tokens_per_ns": rate, "position_models": details,
        "candidate_cutoffs": cutoffs, "probe": probe,
        "reason": "uncertainty value exceeds full-offer cost" if probe else "cost-calibrated evidence",
    }


def update(prior, trace, costs, cost_sha, turn, loop):
    if prior is None:
        prior = {
            "schema": 1, "cost_sha256": cost_sha, "turn": 0,
            "cutoffs": [0.0, 0.0], "rounds": [],
        }
    if (
        prior.get("schema") != 1 or prior.get("cost_sha256") != cost_sha
        or prior.get("turn") != turn - 1 or len(prior.get("cutoffs", [])) != 2
    ):
        raise ValueError("learner state does not belong to this continuing session")
    observations = prior["rounds"] + ([] if loop else [
        row for row in trace if row["eligible"]
    ])
    cutoffs, detail = decide(observations, costs, prior["cutoffs"])
    state = {
        "schema": 1, "cost_sha256": cost_sha, "turn": turn,
        "cutoffs": cutoffs, "rounds": observations,
    }
    return state, {
        "turn": turn, "cutoffs_before": prior["cutoffs"],
        "cutoffs_after": cutoffs, "eligible_rounds_added": (
            0 if loop else sum(row["eligible"] for row in trace)
        ), "excluded_loop": loop, **detail,
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("trace", "costs", "output_ids", "state", "state_out", "decision_out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=name != "state")
    parser.add_argument("--turn", type=int, required=True)
    args = parser.parse_args()
    if args.turn < 1:
        raise ValueError("turns start at one")
    started = time.monotonic_ns()
    prior = json.loads(args.state.read_text()) if args.state else None
    trace = read_rounds(args.trace)
    costs = read_costs(args.costs)
    ids = [int(value) for value in args.output_ids.read_text().split()]
    loop = loop_candidate(ids) is not None
    state, decision = update(prior, trace, costs, sha(args.costs), args.turn, loop)
    decision["policy_cpu_ns"] = time.monotonic_ns() - started
    args.state_out.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n")
    args.decision_out.write_text(json.dumps(decision, sort_keys=True, indent=2) + "\n")
    print(f'{state["cutoffs"][0]:.9g} {state["cutoffs"][1]:.9g}')


if __name__ == "__main__":
    main()
