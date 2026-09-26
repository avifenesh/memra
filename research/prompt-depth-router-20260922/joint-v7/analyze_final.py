"""Score fresh native C/K/D policies by pooled complete-request throughput."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from run_native import v4, save


def score(root):
    summary = json.loads((root / "heldout-summary.json").read_text())
    quality = json.loads((root / "quality.json").read_text())
    fixed = json.loads((root / "selected-fixed.json").read_text())
    selected = json.loads((root / "selected-joint.json").read_text())
    if summary["schema"] != 1 or summary["phase"] != "heldout":
        raise ValueError("fresh C/K/D summary has another schema")
    by_arm = defaultdict(dict)
    for row in summary["records"]:
        index = int(row["name"].split("-")[1])
        if index not in range(6) or index in by_arm[row["variant"]]:
            raise ValueError("duplicate or unexpected fresh conversation")
        by_arm[row["variant"]][index] = row
    if any(set(rows) != set(range(6)) for rows in by_arm.values()):
        raise ValueError("fresh arm lacks six matched conversations")
    probed = {row["session"]: row["passed"] for row in quality["sessions"]}
    if set(probed) != {
        row["name"] for rows in by_arm.values() for row in rows.values()
    }:
        raise ValueError("fresh code probe inventory differs from native arms")

    arms = {}
    for arm, sessions in by_arm.items():
        looped = [
            index for index, row in sessions.items() if row["loops"]
        ]
        all_rows = list(sessions.values())
        tokens = sum(row["tokens"] for row in all_rows)
        seconds = sum(row["seconds"] for row in all_rows)
        k_actions = Counter()
        d_actions = Counter()
        k_ns = depth_ns = c_decisions = c_stops = 0
        drafted = accepted = 0
        continuation = 0
        for row in all_rows:
            session = root / row["name"]
            for turn in v4.table(session / "turns.tsv"):
                if int(turn["sampler_top_k"]) != 20:
                    raise ValueError("target sampler top-k changed")
                k_actions[int(turn["draft_top_k"])] += 1
                k_ns += int(turn["k_model_ns"])
                depth_ns += int(turn["depth_policy_ns"])
                c_decisions += int(turn["confidence_decisions"])
                c_stops += int(turn["confidence_stops"])
                drafted += int(turn["drafted"])
                accepted += int(turn["accepted"])
                if int(turn["turn"]) > 1:
                    if (
                        turn["resumed"] != "true"
                        or int(turn["cached_tokens"]) <= 0
                        or int(turn["new_input_tokens"]) <= 0
                    ):
                        raise ValueError("later turn lacks native KV continuation")
                    continuation += 1
            for round_row in v4.table(session / "rounds.tsv"):
                if round_row["eligible_for_learning"] == "true":
                    d_actions[int(round_row["draft_depth"])] += 1
        if continuation != 42:
            raise ValueError("fresh arm lacks 42 later-turn cache receipts")
        arms[arm] = {
            "returned_tokens": tokens,
            "complete_request_seconds": seconds,
            "pooled_tok_s": tokens / seconds if not looped else None,
            "format_pass": sum(row["format"] for row in all_rows),
            "functional_pass": sum(probed[row["name"]] for row in all_rows),
            "looped_conversations": looped,
            "draft_k_turns": dict(sorted(k_actions.items())),
            "draft_d_eligible_rounds": dict(sorted(d_actions.items())),
            "k_model_seconds": k_ns / 1e9,
            "cd_model_seconds": depth_ns / 1e9,
            "c_decisions": c_decisions,
            "c_stops": c_stops,
            "drafted": drafted,
            "accepted": accepted,
            "acceptance_diagnostic": accepted / drafted if drafted else None,
            "later_turns_with_native_kv_reuse": continuation,
        }
    if "topk20-d3-c0" not in arms or fixed["best"] not in arms:
        raise ValueError("fresh battery lacks recommended or selected fixed control")

    def paired(candidate, control):
        pair = [
            index for index in range(6)
            if not by_arm[candidate][index]["loops"]
            and not by_arm[control][index]["loops"]
        ]
        if len(pair) < 3:
            return {"included_conversations": pair, "status": "too-few-unlooped"}

        def totals(arm, chosen):
            tokens = sum(by_arm[arm][index]["tokens"] for index in chosen)
            seconds = sum(by_arm[arm][index]["seconds"] for index in chosen)
            return tokens, seconds, tokens / seconds

        a_tokens, a_seconds, a_rate = totals(candidate, pair)
        b_tokens, b_seconds, b_rate = totals(control, pair)
        rng = random.Random(20795001)
        samples = []
        for _ in range(20000):
            chosen = [rng.choice(pair) for _ in pair]
            _, _, a = totals(candidate, chosen)
            _, _, b = totals(control, chosen)
            samples.append(100 * (a / b - 1))
        samples.sort()
        return {
            "included_conversations": pair,
            "throughput_delta_percent": 100 * (a_rate / b_rate - 1),
            "bootstrap_95_percent": [
                samples[int(0.025 * len(samples))],
                samples[int(0.975 * len(samples))],
            ],
            "output_token_ratio": a_tokens / b_tokens,
            "total_elapsed_ratio": a_seconds / b_seconds,
        }

    comparisons = {
        control: {
            arm: paired(arm, control)
            for arm in sorted(arms) if arm != control
        }
        for control in {"topk20-d3-c0", fixed["best"]}
    }
    joint = arms.get("joint-learned")
    joint_pair = comparisons[fixed["best"]].get("joint-learned")
    return {
        "schema": 1,
        "scope": "six fresh eight-turn Qwen code conversations with target top-k=20 and MTP draft-only K",
        "objective": "sum returned output tokens / sum complete native request seconds",
        "best_development_fixed": fixed["best"],
        "selected_joint_variant": selected["variant"],
        "arms": arms,
        "paired": comparisons,
        "joint_quality_pass": bool(
            joint and joint["format_pass"] == 48
            and joint["functional_pass"] == 48
            and not joint["looped_conversations"]
        ),
        "joint_95_percent_lower_bound_above_best_fixed": bool(
            joint_pair and joint_pair.get("bootstrap_95_percent", [0])[0] > 0
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.root)
    save(args.root / "final-analysis.json", result)
    print(json.dumps({
        "joint_quality": result["joint_quality_pass"],
        "joint_ci_above_fixed": result["joint_95_percent_lower_bound_above_best_fixed"],
    }))


if __name__ == "__main__":
    main()
