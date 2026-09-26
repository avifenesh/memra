"""Score two-order prose judgments with conversation-level uncertainty."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


CHOICES = {"A++", "A+", "A=B", "B+", "B++"}
TEMPLATE_SHA = "ccd57bd8c4c73f4f83cf8963ef3c2697c1c7b9e907ead91e0d0512cca4ae7a11"
DRAW_COUNT = 20000


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def jsonl(path):
    return [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]


def orientation(choice, candidate_in_a):
    if choice == "A=B":
        return 0
    a_wins = choice in ("A++", "A+")
    return 1 if a_wins == candidate_in_a else -1


def bootstrap(items, seed):
    grouped = defaultdict(list)
    for item in items:
        grouped[item["conversation"]].append(item["vote"])
    conversations = sorted(grouped)
    if not conversations or any(len(grouped[index]) != 8
                                for index in conversations):
        raise ValueError("prose judge lacks whole native conversations")
    rng = random.Random(seed)
    draws = []
    for _ in range(DRAW_COUNT):
        sample = [
            rng.choice(conversations) for _ in conversations
        ]
        votes = [
            value for conversation in sample
            for value in grouped[conversation]
        ]
        draws.append(
            (sum(value == 1 for value in votes)
             + 0.5 * sum(value == 0 for value in votes)) / len(votes)
        )
    draws.sort()
    return [draws[int(0.025 * DRAW_COUNT)],
            draws[int(0.975 * DRAW_COUNT)]]


def score(packets_dir, results_dir, config_path,
          prior_manifest_path=None):
    packets_manifest = json.loads(
        (packets_dir / "manifest.json").read_text()
    )
    results_manifest = json.loads(
        (results_dir / "manifest.json").read_text()
    )
    config = json.loads(config_path.read_text())
    packet_path = packets_dir / "packets.jsonl"
    result_path = results_dir / "results.jsonl"
    if (
        packets_manifest["schema"] != 1
        or packets_manifest["template_sha256"] != TEMPLATE_SHA
        or sha(packet_path) != packets_manifest["packets_sha256"]
        or results_manifest["schema"] != 1
        or results_manifest["packets_sha256"] != sha(packet_path)
        or results_manifest["config_sha256"] != sha(config_path)
        or sha(result_path) != results_manifest["results_sha256"]
        or results_manifest["model_id"] != config["model_id"]
        or results_manifest["pricing_sha256"]
        != sha(results_dir / "pricing.json")
        or config["template_sha256"] != TEMPLATE_SHA
        or not config["model_id"]
        or not all(
            isinstance(config[key], (int, float))
            and math.isfinite(config[key])
            and config[key] > 0
            for key in (
                "input_usd_per_million_budget",
                "output_usd_per_million_budget",
                "total_usd_cap",
            )
        )
    ):
        raise ValueError("prose judgment lacks pinned prompt/model lineage")
    packets = jsonl(packet_path)
    results = jsonl(result_path)
    if len(packets) != packets_manifest["packet_count"] or (
        len(results) != len(packets)
    ):
        raise ValueError("prose judgment inventory is incomplete")
    grouped = defaultdict(dict)
    usage = {"input_tokens": 0, "output_tokens": 0}
    response_ids = set()
    for index, (packet, result) in enumerate(zip(packets, results)):
        if (
            result["packet_index"] != index
            or result["prompt_sha256"] != packet["prompt_sha256"]
            or result["model_id"] != config["model_id"]
            or result["choice"] not in CHOICES
            or not result["provider_response_id"]
            or not isinstance(result["input_tokens"], int)
            or not isinstance(result["output_tokens"], int)
            or result["input_tokens"] <= 0
            or result["output_tokens"] <= 0
            or result["provider_response_id"] in response_ids
        ):
            raise ValueError("prose judgment differs from frozen packet")
        response_ids.add(result["provider_response_id"])
        usage["input_tokens"] += result["input_tokens"]
        usage["output_tokens"] += result["output_tokens"]
        key = (
            packet["conversation"], packet["turn"],
            packet["candidate"], packet["control"],
        )
        if packet["order"] in grouped[key]:
            raise ValueError("duplicate prose order")
        grouped[key][packet["order"]] = orientation(
            result["choice"],
            packet["response_a"] == packet["candidate"],
        )
    projected_usd = (
        usage["input_tokens"] * config["input_usd_per_million_budget"]
        + usage["output_tokens"]
        * config["output_usd_per_million_budget"]
    ) / 1_000_000
    prior_sha = None
    total_usage = dict(usage)
    if packets_manifest["phase"] == "final":
        if prior_manifest_path is None:
            raise ValueError("final prose score lacks prior judge usage")
        prior = json.loads(prior_manifest_path.read_text())
        if (
            prior["status"] != "complete"
            or prior["config_sha256"] != sha(config_path)
            or prior["model_id"] != config["model_id"]
            or results_manifest["prior_judge_manifest_sha256"]
            != sha(prior_manifest_path)
        ):
            raise ValueError("final prose judge budget lineage differs")
        prior_sha = sha(prior_manifest_path)
        for key in total_usage:
            total_usage[key] += prior["usage"][key]
    elif (
        packets_manifest["phase"] != "validation"
        or prior_manifest_path is not None
        or results_manifest["prior_judge_manifest_sha256"] is not None
    ):
        raise ValueError("prose judge validation budget phase differs")
    cumulative_usd = (
        total_usage["input_tokens"]
        * config["input_usd_per_million_budget"]
        + total_usage["output_tokens"]
        * config["output_usd_per_million_budget"]
    ) / 1_000_000
    if not math.isfinite(cumulative_usd) or (
        cumulative_usd > config["total_usd_cap"]
        or abs(
            cumulative_usd - results_manifest["budgeted_usd_ceiling"]
        ) > 1e-8
    ):
        raise ValueError("prose judge cost cap exceeded")
    rows = defaultdict(list)
    for key, by_order in grouped.items():
        if set(by_order) != {0, 1}:
            raise ValueError("prose comparison lacks reversed judgment")
        vote = by_order[0] if (
            by_order[0] == by_order[1]
        ) else 0
        conversation, turn, candidate, control = key
        rows[candidate, control].append({
            "conversation": conversation, "turn": turn,
            "vote": vote,
            "order_agreed": by_order[0] == by_order[1],
        })
    comparisons = {}
    for (candidate, control), items in sorted(rows.items()):
        items.sort(key=lambda row: (row["conversation"], row["turn"]))
        wins = sum(row["vote"] == 1 for row in items)
        losses = sum(row["vote"] == -1 for row in items)
        ties = len(items) - wins - losses
        comparison = {
            "candidate": candidate, "control": control,
            "task_count": len(items),
            "wins": wins, "losses": losses, "ties": ties,
            "reverse_order_disagreements":
            sum(not row["order_agreed"] for row in items),
            "point_win_fraction": (wins + 0.5 * ties) / len(items),
            "bootstrap_95_win_fraction": bootstrap(
                items,
                260926 + sum(map(ord, candidate + control)),
            ),
            "tasks": items,
        }
        comparisons[f"{candidate}::vs::{control}"] = comparison
    if set(comparisons) != {
        f"{candidate}::vs::{control}"
        for candidate, control in packets_manifest["pairs"]
    }:
        raise ValueError("prose comparison pair inventory changed")
    return {
        "schema": 1, "phase": packets_manifest["phase"],
        "scope": "frozen checklist pairwise, both A/B orders, whole-conversation CI",
        "packets_manifest_sha256": sha(packets_dir / "manifest.json"),
        "judge_results_manifest_sha256": sha(results_dir / "manifest.json"),
        "arms_sha256": packets_manifest["arms_sha256"],
        "workloads_sha256": packets_manifest["workloads_sha256"],
        "template_sha256": TEMPLATE_SHA,
        "judge_config_sha256": sha(config_path),
        "judge_model_id": config["model_id"],
        "judge_usage": usage,
        "prior_judge_manifest_sha256": prior_sha,
        "budgeted_usd_ceiling": projected_usd,
        "cumulative_budgeted_usd_ceiling": cumulative_usd,
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("packets", "results", "config", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prior-manifest", type=Path)
    args = parser.parse_args()
    result = score(
        args.packets.resolve(), args.results.resolve(),
        args.config.resolve(),
        args.prior_manifest.resolve() if args.prior_manifest else None,
    )
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({
        "phase": result["phase"],
        "judge_model_id": result["judge_model_id"],
        "comparisons": len(result["comparisons"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
