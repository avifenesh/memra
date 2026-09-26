"""Score v11 non-code validation and final native requests by topic."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


DOMAINS = ("ifeval", "gsm8k")
REFERENCE = "fixed-k20-d3-c0"
PHASE_SIZE = {"validation": 8, "heldout": 16}
TURNS = 8


def pooled(rows, indices):
    tokens = sum(rows[i]["tokens"] for i in indices)
    seconds = sum(rows[i]["seconds"] for i in indices)
    return {
        "tokens": tokens,
        "seconds": seconds,
        "tok_s": tokens / seconds,
    }


def paired(rows, candidate, control, seed, count):
    included = [
        index for index in range(count)
        if not rows[candidate][index]["loops"]
        and not rows[control][index]["loops"]
    ]
    if len(included) < count - 2:
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


def assert_noop_identity(root, domain, labels, phase, count):
    noops = sorted(
        label for label in labels
        if "-noop-" in label
    )
    for index in range(count):
        baseline = root / f"{phase}-{domain}-{index}-{REFERENCE}"
        for turn in range(1, TURNS + 1):
            expected = (baseline / f"turn-{turn}.output.ids").read_bytes()
            for label in noops:
                session = root / f"{phase}-{domain}-{index}-{label}"
                if (session / f"turn-{turn}.output.ids").read_bytes() != expected:
                    raise ValueError(
                        f"{domain} sampled no-op differs: {label}/{index}/{turn}"
                    )
    return noops


def noop_for(label):
    for kind in ("cd-", "joint-", "k-"):
        if label.startswith(kind) and not label.startswith(kind + "noop-"):
            return kind + "noop-" + label.removeprefix(kind)
    return None


def domain_score(root, quality, arms, domain, phase, seed):
    count = PHASE_SIZE[phase]
    labels = [row["label"] for row in arms["arms"]]
    if (
        len(set(labels)) != len(labels)
        or REFERENCE not in labels
        or len([label for label in labels if label.startswith("fixed-")]) != 7
        or (phase == "validation" and len(labels) != 19)
    ):
        raise ValueError("non-code fixed or learned arm inventory differs")
    by_name = {
        item["session"]: item
        for item in quality["domains"][domain]["sessions"]
    }
    if len(by_name) != count * len(labels):
        raise ValueError("non-code grader session count differs")
    rows = {}
    scores = {}
    for label in labels:
        group = []
        graded = []
        for index in range(count):
            name = f"{phase}-{domain}-{index}-{label}"
            native = json.loads((root / f"{name}.result.json").read_text())
            if native["name"] != name or native["cached_later_turns"] != 7:
                raise ValueError("non-code native continuation differs")
            group.append(native)
            graded.append(by_name[name])
        rows[label] = group
        unlooped = [
            i for i in range(count) if not group[i]["loops"]
        ]
        unlooped_group = [group[i] for i in unlooped]
        total = pooled(group, unlooped) if unlooped else {
            "tokens": 0, "seconds": 0, "tok_s": None,
        }
        finished = Counter(
            reason for row in group for reason in row["finished"]
        )
        tokens_drafted = sum(row["drafted"] for row in group)
        k_actions = action_counts(group, "k_actions")
        d_actions = action_counts(group, "d_actions")
        c_decisions = sum(row["c_decisions"] for row in group)
        c_stops = sum(row["c_stops"] for row in group)
        behavior_k = action_counts(unlooped_group, "k_actions")
        behavior_d = action_counts(unlooped_group, "d_actions")
        behavior_decisions = sum(
            row["c_decisions"] for row in unlooped_group
        )
        behavior_stops = sum(
            row["c_stops"] for row in unlooped_group
        )
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
                i for i in range(count) if group[i]["loops"]
            ],
            "k_actions": k_actions,
            "d_actions": d_actions,
            "c_decisions": c_decisions,
            "c_stops": c_stops,
            "dynamic_k": len(behavior_k) > 1,
            "dynamic_d": len(behavior_d) > 1,
            "dynamic_c": 0 < behavior_stops < behavior_decisions,
            "k_model_seconds": sum(row["k_model_s"] for row in group),
            "cd_model_seconds": sum(row["cd_model_s"] for row in group),
            "accepted": sum(row["accepted"] for row in group),
            "drafted": tokens_drafted,
            "acceptance_diagnostic": (
                sum(row["accepted"] for row in group) / tokens_drafted
                if tokens_drafted else None
            ),
        }
    noops = assert_noop_identity(root, domain, labels, phase, count)
    baseline = scores[REFERENCE]
    margin = 0.05 * count * TURNS
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
            "vs_reference": paired(rows, label, REFERENCE, seed, count),
        }
        noop = noop_for(label)
        if noop is not None:
            if noop not in labels:
                raise ValueError(f"learned arm lacks no-op: {label}")
            comparisons[label]["vs_own_noop"] = paired(
                rows, label, noop, seed, count
            )
            comparisons[label]["vs_each_eligible_fixed"] = {
                control: paired(rows, label, control, seed, count)
                for control in fixed
            }
            own = comparisons[label]["vs_own_noop"]
            against = comparisons[label]["vs_each_eligible_fixed"]
            common = [
                index for index in range(count)
                if all(
                    not rows[arm][index]["loops"]
                    for arm in (label, noop, *fixed)
                )
            ]
            matched = [rows[label][index] for index in common]
            matched_decisions = sum(
                row["c_decisions"] for row in matched
            )
            matched_stops = sum(
                row["c_stops"] for row in matched
            )
            scores[label]["paired_behavior_conversations"] = common
            scores[label]["paired_dynamic_k"] = (
                len(action_counts(matched, "k_actions")) > 1
            )
            scores[label]["paired_dynamic_d"] = (
                len(action_counts(matched, "d_actions")) > 1
            )
            scores[label]["paired_dynamic_c"] = (
                0 < matched_stops < matched_decisions
            )
            if phase == "heldout":
                scores[label]["exploratory_win"] = (
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
        "byte_identical_noops": noops,
    }


def score(root, quality, arms):
    phase = quality["phase"]
    if (
        quality["schema"] != 1
        or phase not in PHASE_SIZE
        or arms["schema"] != 1
        or arms["phase"] != phase
        or set(quality["domains"]) != set(DOMAINS)
    ):
        raise ValueError("non-code quality result differs")
    return {
        "schema": 1,
        "phase": phase,
        "scope": "Qwen v11 learned C/K/D, domain-specific native request tok/s",
        "domains": {
            domain: domain_score(
                root, quality, arms, domain, phase, 25197001 + index
            )
            for index, domain in enumerate(DOMAINS)
        },
    }


def attach_receipts(result, quality_path, arms_path):
    result["quality_sha256"] = hashlib.sha256(
        quality_path.read_bytes()
    ).hexdigest()
    result["arms_sha256"] = hashlib.sha256(
        arms_path.read_bytes()
    ).hexdigest()
    if result["phase"] == "heldout":
        selected = arms_path.with_name("selected.json")
        validation = arms_path.with_name("validation-score.json")
        choice = json.loads(selected.read_text())
        arm_specs = json.loads(arms_path.read_text())
        inventory = hashlib.sha256(json.dumps(
            arm_specs["arms"], sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if (
            arm_specs["selected_from_validation"]
            != hashlib.sha256(selected.read_bytes()).hexdigest()
            or choice["phase"] != "validation-selected"
            or choice["status"] != "selected"
            or hashlib.sha256(validation.read_bytes()).hexdigest()
            != choice["validation_score_sha256"]
            or sorted(choice["final_arms"])
            != sorted(item["label"] for item in arm_specs["arms"])
            or choice["final_arm_inventory_sha256"] != inventory
        ):
            raise ValueError("v11 final arms lack frozen validation selection")
        for domain in DOMAINS:
            primary = choice["chosen_by_domain"][domain]["primary"]
            report = result["domains"][domain]
            if primary is not None and (
                primary not in report["arms"]
                or "exploratory_win" not in report["arms"][primary]
            ):
                raise ValueError("v11 confirmatory arm differs from validation")
            report["primary_label"] = primary
            report["confirmatory_win"] = (
                report["arms"][primary]["exploratory_win"]
                if primary is not None else False
            )
            report["confirmatory_adaptive_win"] = (
                report["confirmatory_win"]
                and len(
                    report["arms"][primary][
                        "paired_behavior_conversations"
                    ]
                ) >= PHASE_SIZE["heldout"] - 2
                and any(
                    report["arms"][primary][key]
                    for key in (
                        "paired_dynamic_k", "paired_dynamic_d",
                        "paired_dynamic_c",
                    )
                )
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = attach_receipts(
        score(
            args.root,
            json.loads(args.quality.read_text()),
            json.loads(args.arms.read_text()),
        ),
        args.quality, args.arms,
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
