"""Score non-code native transfer separately for prose and math."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random


DOMAINS = ("ifeval", "gsm8k")
REFERENCE = "fixed-k20-d3-c0"
CONVERSATIONS = 16
TURNS = 8


def pooled(rows, indices):
    tokens = sum(rows[i]["tokens"] for i in indices)
    seconds = sum(rows[i]["seconds"] for i in indices)
    return {
        "tokens": tokens,
        "seconds": seconds,
        "tok_s": tokens / seconds,
    }


def paired(rows, candidate, control, seed):
    included = [
        index for index in range(CONVERSATIONS)
        if not rows[candidate][index]["loops"]
        and not rows[control][index]["loops"]
    ]
    if len(included) < CONVERSATIONS - 2:
        return {"status": "too-many-looped-conversations", "included": included}
    a = pooled(rows[candidate], included)
    b = pooled(rows[control], included)
    rng = random.Random(seed)
    draws = []
    for _ in range(20000):
        sample = [rng.choice(included) for _ in included]
        draws.append(
            100 * (
                pooled(rows[candidate], sample)["tok_s"]
                / pooled(rows[control], sample)["tok_s"] - 1
            )
        )
    draws.sort()
    return {
        "status": "paired",
        "included": included,
        "delta_percent": 100 * (a["tok_s"] / b["tok_s"] - 1),
        "bootstrap_95_percent": [
            draws[int(0.025 * len(draws))],
            draws[int(0.975 * len(draws))],
        ],
        "output_token_ratio": a["tokens"] / b["tokens"],
        "elapsed_ratio": a["seconds"] / b["seconds"],
    }


def action_counts(rows, key):
    counts = Counter()
    for row in rows:
        counts.update({int(k): n for k, n in row[key].items()})
    return dict(sorted(counts.items()))


def assert_noop_identity(root, domain, labels):
    noops = sorted(
        label for label in labels
        if label.startswith(("cd-noop-", "joint-noop-"))
    )
    for index in range(CONVERSATIONS):
        baseline = root / f"heldout-{domain}-{index}-{REFERENCE}"
        for turn in range(1, TURNS + 1):
            expected = (baseline / f"turn-{turn}.output.ids").read_bytes()
            for label in noops:
                session = root / f"heldout-{domain}-{index}-{label}"
                if (session / f"turn-{turn}.output.ids").read_bytes() != expected:
                    raise ValueError(
                        f"{domain} sampled no-op differs: {label}/{index}/{turn}"
                    )
    return noops


def domain_score(root, quality, arms, domain, seed):
    labels = [row["label"] for row in arms["arms"]]
    if len(labels) != 10 or len(set(labels)) != len(labels) or REFERENCE not in labels:
        raise ValueError("non-code fixed or learned arm inventory differs")
    by_name = {
        item["session"]: item
        for item in quality["domains"][domain]["sessions"]
    }
    if len(by_name) != CONVERSATIONS * len(labels):
        raise ValueError("non-code grader session count differs")
    rows = {}
    scores = {}
    for label in labels:
        group = []
        graded = []
        for index in range(CONVERSATIONS):
            name = f"heldout-{domain}-{index}-{label}"
            native = json.loads((root / f"{name}.result.json").read_text())
            if native["name"] != name or native["cached_later_turns"] != 7:
                raise ValueError("non-code native continuation differs")
            group.append(native)
            graded.append(by_name[name])
        rows[label] = group
        unlooped = [
            i for i in range(CONVERSATIONS) if not group[i]["loops"]
        ]
        total = pooled(group, unlooped) if unlooped else {
            "tokens": 0, "seconds": 0, "tok_s": None,
        }
        finished = Counter(
            reason for row in group for reason in row["finished"]
        )
        tokens_drafted = sum(row["drafted"] for row in group)
        scores[label] = {
            **total,
            "task_pass": sum(row["task_pass"] for row in graded),
            "instruction_pass": sum(row["instruction_pass"] for row in graded),
            "instruction_total": sum(row["instruction_total"] for row in graded),
            "numeric_answer_present": sum(
                row["numeric_answer_present"] for row in graded
            ),
            "capped_turns": finished["length"],
            "finish_reasons": dict(sorted(finished.items())),
            "looped_turns": sum(row["loops"] for row in group),
            "looped_conversations": [
                i for i in range(CONVERSATIONS) if group[i]["loops"]
            ],
            "k_actions": action_counts(group, "k_actions"),
            "d_actions": action_counts(group, "d_actions"),
            "c_decisions": sum(row["c_decisions"] for row in group),
            "c_stops": sum(row["c_stops"] for row in group),
            "k_model_seconds": sum(row["k_model_s"] for row in group),
            "cd_model_seconds": sum(row["cd_model_s"] for row in group),
            "accepted": sum(row["accepted"] for row in group),
            "drafted": tokens_drafted,
            "acceptance_diagnostic": (
                sum(row["accepted"] for row in group) / tokens_drafted
                if tokens_drafted else None
            ),
        }
    assert_noop_identity(root, domain, labels)
    baseline = scores[REFERENCE]
    margin = 0.05 * CONVERSATIONS * TURNS
    for item in scores.values():
        item["quality_eligible"] = (
            item["looped_turns"] <= baseline["looped_turns"]
            and item["task_pass"] + margin >= baseline["task_pass"]
            and item["capped_turns"] <= baseline["capped_turns"] + margin
        )
    fixed = [
        label for label in labels
        if label.startswith("fixed-") and scores[label]["quality_eligible"]
    ]
    if REFERENCE not in fixed:
        raise ValueError("non-code fixed reference is ineligible")
    comparisons = {}
    for label in labels:
        if label == REFERENCE:
            continue
        comparisons[label] = {
            "vs_reference": paired(rows, label, REFERENCE, seed),
        }
        if label.startswith(("cd-", "joint-")) and "-noop-" not in label:
            noop = label.replace("-", "-noop-", 1)
            comparisons[label]["vs_own_noop"] = paired(
                rows, label, noop, seed
            )
            comparisons[label]["vs_each_eligible_fixed"] = {
                control: paired(rows, label, control, seed)
                for control in fixed
            }
            own = comparisons[label]["vs_own_noop"]
            against = comparisons[label]["vs_each_eligible_fixed"]
            scores[label]["transfer_win"] = (
                scores[label]["quality_eligible"]
                and own["status"] == "paired"
                and own["bootstrap_95_percent"][0] > 0
                and all(
                    result["status"] == "paired"
                    and result["bootstrap_95_percent"][0] > 0
                    for result in against.values()
                )
            )
    return {
        "arms": scores,
        "comparisons": comparisons,
        "eligible_fixed": fixed,
        "byte_identical_noops": [
            label for label in labels
            if label.startswith(("cd-noop-", "joint-noop-"))
        ],
    }


def score(root, quality, arms):
    if quality["schema"] != 1 or quality["phase"] != "heldout":
        raise ValueError("non-code quality result differs")
    return {
        "schema": 1,
        "scope": "Qwen code-trained C/K/D transfer, domain-specific native request tok/s",
        "domains": {
            domain: domain_score(root, quality, arms, domain, 25193001 + index)
            for index, domain in enumerate(DOMAINS)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = score(
        args.root,
        json.loads(args.quality.read_text()),
        json.loads(args.arms.read_text()),
    )
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({
        domain: {
            name: {
                "tok_s": row["tok_s"],
                "task_pass": row["task_pass"],
                "quality_eligible": row["quality_eligible"],
            }
            for name, row in report["arms"].items()
        }
        for domain, report in result["domains"].items()
    }, sort_keys=True))


if __name__ == "__main__":
    main()
