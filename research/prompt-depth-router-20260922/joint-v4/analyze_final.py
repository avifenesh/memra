"""Audit fresh, continuing-session E2E throughput against fixed controls."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random


def table(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def score(root):
    summary = json.loads((root / "heldout-summary.json").read_text())
    selected = json.loads((root / "selected-model.json").read_text())
    selected_c = json.loads((root / "selected-confidence.json").read_text())
    quality_report = json.loads((root / "quality.json").read_text())
    if summary["schema"] != selected["schema"] != 1 or summary["phase"] != "heldout":
        raise ValueError("fresh evaluation or frozen model schema differs")
    expected = {
        "k3-c0", "native-adapt",
        f"{selected['variant']}-trained", f"{selected['variant']}-noop",
        "c-trained", "c-noop", "cd-trained", "cd-noop",
    }
    if selected["fixed_control"] != "k3-c0":
        expected.add(selected["fixed_control"])
    by_variant = defaultdict(dict)
    for record in summary["records"]:
        name = record["name"]
        parts = name.split("-", 2)
        if len(parts) != 3 or parts[0] != "heldout":
            raise ValueError("another conversation name in fresh evaluation")
        index = int(parts[1])
        if index not in range(6) or record["variant"] not in expected:
            raise ValueError("unexpected fresh evaluation arm")
        if index in by_variant[record["variant"]]:
            raise ValueError("duplicate conversation and arm")
        by_variant[record["variant"]][index] = record
    if set(by_variant) != expected or any(len(rows) != 6 for rows in by_variant.values()):
        raise ValueError("fresh evaluation lacks a complete matched arm grid")
    probed = {
        row["session"]: row["passed"]
        for row in quality_report["sessions"] if row["session"].startswith("heldout-")
    }
    if set(probed) != {
        by_variant[arm][index]["name"]
        for arm in expected for index in range(6)
    }:
        raise ValueError("fresh evaluation lacks matched functional probes")
    excluded = [
        index for index in range(6)
        if any(by_variant[arm][index]["loops"] for arm in expected)
    ]
    included = [index for index in range(6) if index not in excluded]
    if not included:
        raise ValueError("every fresh conversation has a looped arm")

    def totals(arm, sample):
        records = [by_variant[arm][index] for index in sample]
        tokens = sum(row["tokens"] for row in records)
        seconds = sum(row["seconds"] for row in records)
        return tokens, seconds, tokens / seconds

    control = "k3-c0"
    rows = {}
    for arm in sorted(expected):
        tokens, seconds, rate = totals(arm, included)
        quality = sum(by_variant[arm][index]["format"] for index in range(6))
        depth = Counter()
        policy_ns = 0
        confidence_decisions = confidence_stops = 0
        offered = accepted = rounds = 0
        for index in included:
            session = root / by_variant[arm][index]["name"]
            for row in table(session / "rounds.tsv"):
                if row["eligible_for_learning"] == "true":
                    depth[int(row["draft_depth"])] += 1
            for row in table(session / "turns.tsv"):
                policy_ns += int(row["depth_policy_ns"])
                confidence_decisions += int(row["confidence_decisions"])
                confidence_stops += int(row["confidence_stops"])
                offered += int(row["drafted"])
                accepted += int(row["accepted"])
                rounds += int(row["policy_rounds"])
        rows[arm] = {
            "tokens": tokens,
            "seconds": seconds,
            "tok_s": rate,
            "all_48_format_pass": quality == 48,
            "format_pass": quality,
            "functional_pass": sum(
                probed[by_variant[arm][index]["name"]] for index in range(6)
            ),
            "loops_all_six": sum(by_variant[arm][index]["loops"] for index in range(6)),
            "eligible_d_distribution": dict(sorted(depth.items())),
            "depth_policy_seconds": policy_ns / 1e9,
            "confidence_decisions": confidence_decisions,
            "confidence_stops": confidence_stops,
            "offered_draft_tokens": offered,
            "accepted_draft_tokens": accepted,
            "accepted_over_offered": accepted / offered if offered else None,
            "policy_rounds": rounds,
        }
    base_tokens, base_seconds, base_rate = totals(control, included)
    rng = random.Random(20779001)
    paired = {}
    for arm in sorted(expected - {control}):
        tokens, seconds, rate = totals(arm, included)
        bootstrap = []
        for _ in range(20000):
            sample = [rng.choice(included) for _ in included]
            _, _, a = totals(arm, sample)
            _, _, b = totals(control, sample)
            bootstrap.append(100 * (a / b - 1))
        bootstrap.sort()
        paired[arm] = {
            "throughput_delta_percent": 100 * (rate / base_rate - 1),
            "bootstrap_95_percent": [
                bootstrap[int(0.025 * len(bootstrap))],
                bootstrap[int(0.975 * len(bootstrap))],
            ],
            "output_token_ratio": tokens / base_tokens,
            "total_elapsed_ratio": seconds / base_seconds,
            "per_conversation_rate_delta_percent": [
                100 * (
                    by_variant[arm][index]["tokens"]
                    / by_variant[arm][index]["seconds"]
                    / (
                        by_variant[control][index]["tokens"]
                        / by_variant[control][index]["seconds"]
                    ) - 1
                )
                for index in included
            ],
        }
    learned_d = f"{selected['variant']}-trained"
    learned = "cd-trained"
    return {
        "schema": 1,
        "scope": "fresh Qwen code native eight-turn conversations on one GPU",
        "objective": "sum returned output tokens / sum complete native request seconds",
        "selected_model_sha256": selected["model_sha256"],
        "selected_variant": selected["variant"],
        "selected_confidence_sha256": selected_c["model_sha256"],
        "selected_confidence_variant": selected_c["variant"],
        "excluded_conversation_indices_due_to_any_loop": excluded,
        "included_conversation_indices": included,
        "arms": rows,
        "paired_vs_k3_c0": paired,
        "learned_quality_gate": (
            rows[learned]["all_48_format_pass"]
            and rows[learned]["functional_pass"] == 48
        ),
        "depth_only_quality_gate": (
            rows[learned_d]["all_48_format_pass"]
            and rows[learned_d]["functional_pass"] == 48
        ),
        "confidence_only_quality_gate": (
            rows["c-trained"]["all_48_format_pass"]
            and rows["c-trained"]["functional_pass"] == 48
        ),
        "learned_beats_all_executed_fixed": (
            rows[learned]["tok_s"] > max(
                rows[arm]["tok_s"]
                for arm in expected
                if arm in ("k3-c0", selected["fixed_control"])
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.root)
    with args.out.open("x") as target:
        json.dump(result, target, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps({
        "variant": result["selected_variant"],
        "quality": result["learned_quality_gate"],
        "beats_fixed": result["learned_beats_all_executed_fixed"],
    }))


if __name__ == "__main__":
    main()
