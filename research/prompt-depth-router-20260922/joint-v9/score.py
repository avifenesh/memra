"""Score paired complete native requests and freeze validation choices."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random


PHASE_SIZE = {"validation": 8, "heldout": 16}
REFERENCE = "fixed-k20-d3-c0"


def by_arm(root, quality, phase):
    count = PHASE_SIZE[phase]
    records = defaultdict(dict)
    for path in root.glob(f"{phase}-*.result.json"):
        row = json.loads(path.read_text())
        index = int(row["name"].split("-")[1])
        arm = row["variant"]
        if index not in range(count) or index in records[arm]:
            raise ValueError("duplicate or unexpected scored conversation")
        records[arm][index] = row
    if not records or any(set(group) != set(range(count)) for group in records.values()):
        raise ValueError("an evaluation arm lacks a full conversation set")
    quality_rows = {
        row["session"]: row for row in quality["sessions"]
    }
    if set(quality_rows) != {
        row["name"] for group in records.values() for row in group.values()
    }:
        raise ValueError("code quality inventory differs from scored native requests")
    return records, quality_rows


def totals(rows, indices):
    tokens = sum(rows[index]["tokens"] for index in indices)
    seconds = sum(rows[index]["seconds"] for index in indices)
    return {
        "tokens": tokens,
        "seconds": seconds,
        "tok_s": tokens / seconds,
    }


def paired(records, candidate, control, count):
    indices = [
        index for index in range(count)
        if not records[candidate][index]["loops"]
        and not records[control][index]["loops"]
    ]
    if len(indices) < count - 2:
        return {"status": "too-many-looped-conversations", "included": indices}
    candidate_totals = totals(records[candidate], indices)
    control_totals = totals(records[control], indices)
    rng = random.Random(20927001)
    draws = []
    for _ in range(20000):
        chosen = [rng.choice(indices) for _ in indices]
        a = totals(records[candidate], chosen)["tok_s"]
        b = totals(records[control], chosen)["tok_s"]
        draws.append(100 * (a / b - 1))
    draws.sort()
    return {
        "status": "paired",
        "included": indices,
        "delta_percent": 100 * (
            candidate_totals["tok_s"] / control_totals["tok_s"] - 1
        ),
        "bootstrap_95_percent": [
            draws[int(0.025 * len(draws))],
            draws[int(0.975 * len(draws))],
        ],
        "output_token_ratio": (
            candidate_totals["tokens"] / control_totals["tokens"]
        ),
        "elapsed_ratio": (
            candidate_totals["seconds"] / control_totals["seconds"]
        ),
    }


def score(root, quality_path, arms_path, phase):
    if phase not in PHASE_SIZE:
        raise ValueError("only validation and final phases can be scored")
    count = PHASE_SIZE[phase]
    quality = json.loads(quality_path.read_text())
    arms = json.loads(arms_path.read_text())
    if arms["phase"] != phase:
        raise ValueError("arm manifest belongs to another phase")
    records, quality_rows = by_arm(root, quality, phase)
    expected = {spec["label"] for spec in arms["arms"]}
    if set(records) != expected or REFERENCE not in records:
        raise ValueError("scored arms differ from frozen candidates")
    result = {}
    for arm, group in records.items():
        unlooped = [
            index for index in range(count) if not group[index]["loops"]
        ]
        result[arm] = {
            **(totals(group, unlooped) if unlooped else {
                "tokens": 0, "seconds": 0, "tok_s": None,
            }),
            "looped_turns": sum(row["loops"] for row in group.values()),
            "looped_conversations": [
                index for index in range(count) if group[index]["loops"]
            ],
            "finish_reasons": dict(sorted(Counter(
                reason
                for conversation in group.values()
                for reason in conversation["finished"]
            ).items())),
            "syntax_pass": sum(
                quality_rows[row["name"]]["syntax_pass"]
                for row in group.values()
            ),
            "all_tests_pass": sum(
                quality_rows[row["name"]]["all_tests_pass"]
                for row in group.values()
            ),
            "c_decisions": sum(row["c_decisions"] for row in group.values()),
            "c_stops": sum(row["c_stops"] for row in group.values()),
        }
    baseline = result[REFERENCE]
    margin = 0.05 * count * 8
    for arm, row in result.items():
        row["capped_turns"] = row["finish_reasons"].get("length", 0)
        row["quality_eligible"] = (
            row["looped_turns"] <= baseline["looped_turns"]
            and row["syntax_pass"] + margin >= baseline["syntax_pass"]
            and row["all_tests_pass"] + margin >= baseline["all_tests_pass"]
            and row["capped_turns"] <= baseline["finish_reasons"].get("length", 0) + margin
        )
    comparisons = {}
    for arm in records:
        if arm == REFERENCE:
            continue
        comparisons[arm] = {"vs_fixed_k20": paired(records, arm, REFERENCE, count)}
        if arm.startswith("k-") and not arm.startswith("k-noop-"):
            noop = "k-noop-" + arm.removeprefix("k-")
            comparisons[arm]["vs_own_noop"] = paired(records, arm, noop, count)
        if arm.startswith("cd-") and not arm.startswith("cd-noop-"):
            noop = "cd-noop-" + arm.removeprefix("cd-")
            comparisons[arm]["vs_own_noop"] = paired(records, arm, noop, count)
        if arm.startswith("joint-") and not arm.startswith("joint-noop-"):
            noop = "joint-noop-" + arm.removeprefix("joint-")
            comparisons[arm]["vs_own_noop"] = paired(records, arm, noop, count)
    if phase == "heldout":
        selection_path = arms_path.with_name("selected.json")
        selection = json.loads(selection_path.read_text())
        if (
            selection["phase"] != "validation-selected"
            or arms["selected_from_validation"]
            != hashlib.sha256(selection_path.read_bytes()).hexdigest()
        ):
            raise ValueError("final score lacks a frozen validation selection")
        fixed = selection["chosen"]["fixed"]
        if fixed not in records:
            raise ValueError("selected fixed control missing from final battery")
        eligible_fixed = [
            name for name in records
            if name.startswith("fixed-") and result[name]["quality_eligible"]
        ]
        if not eligible_fixed:
            raise ValueError("no quality-eligible fixed arm on final requests")
        for arm in records:
            if arm == fixed:
                continue
            comparisons.setdefault(arm, {})["vs_selected_fixed"] = paired(
                records, arm, fixed, count
            )
        for arm in records:
            if not arm.startswith(("k-", "cd-", "joint-")) or "-noop-" in arm:
                continue
            versus = comparisons[arm]
            own_noop = versus["vs_own_noop"]
            selected_fixed = versus["vs_selected_fixed"]
            fixed_pairs = {
                name: paired(records, arm, name, count)
                for name in eligible_fixed
            }
            versus["vs_each_eligible_fixed"] = fixed_pairs
            result[arm]["learned_win"] = (
                result[arm]["quality_eligible"]
                and own_noop["status"] == "paired"
                and selected_fixed["status"] == "paired"
                and own_noop["bootstrap_95_percent"][0] > 0
                and all(
                    pair["status"] == "paired"
                    and pair["bootstrap_95_percent"][0] > 0
                    for pair in fixed_pairs.values()
                )
            )
    return {
        "schema": 1,
        "phase": phase,
        "arms_sha256": hashlib.sha256(arms_path.read_bytes()).hexdigest(),
        "quality_sha256": hashlib.sha256(quality_path.read_bytes()).hexdigest(),
        "arms": result,
        "comparisons": comparisons,
    }


def select(report, arms_path, out):
    if report["phase"] != "validation":
        raise ValueError("policy selection must use validation only")
    candidates = json.loads(arms_path.read_text())["arms"]
    options = {item["label"]: item for item in candidates}

    def best(names):
        eligible = [
            name for name in names
            if report["arms"][name]["quality_eligible"]
            and report["arms"][name]["tok_s"] is not None
        ]
        return max(eligible, key=lambda name: report["arms"][name]["tok_s"]) if eligible else None

    chosen = {
        "fixed": best([
            name for name in options if name.startswith("fixed-")
        ]),
        "k": best(("k-new-only", "k-augmented")),
        "cd": best(("cd-new-only", "cd-augmented", "cd-token", "cd-prior")),
        "joint": best(("joint-new-only", "joint-augmented")),
    }
    if chosen["fixed"] is None:
        raise ValueError("no quality-eligible fixed control")
    final_names = {
        "fixed-k3-d3-c0", "fixed-k10-d3-c0", "fixed-k20-d3-c0",
        chosen["fixed"],
    }
    for key in ("k", "cd", "joint"):
        name = chosen[key]
        if name is None:
            continue
        final_names.add(name)
        if key in ("k", "cd", "joint"):
            final_names.add(name.replace(f"{key}-", f"{key}-noop-", 1))
    final_arms = [
        options[name] for name in sorted(final_names)
    ]
    selection = {
        "schema": 1,
        "phase": "validation-selected",
        "validation_arms_sha256": report["arms_sha256"],
        "chosen": chosen,
        "final_arms": sorted(final_names),
    }
    out.mkdir(exist_ok=False)
    (out / "selected.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n"
    )
    (out / "heldout-arms.json").write_text(
        json.dumps({
            "schema": 1,
            "phase": "heldout",
            "selected_from_validation": hashlib.sha256(
                (out / "selected.json").read_bytes()
            ).hexdigest(),
            "arms": final_arms,
        }, indent=2, sort_keys=True) + "\n"
    )
    return selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--phase", choices=("validation", "heldout"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection-out", type=Path)
    args = parser.parse_args()
    report = score(args.root, args.quality, args.arms, args.phase)
    with args.out.open("x") as target:
        json.dump(report, target, indent=2, sort_keys=True)
        target.write("\n")
    result = {
        "phase": args.phase,
        "arms": {
            name: {
                "tok_s": row["tok_s"],
                "quality_eligible": row["quality_eligible"],
            }
            for name, row in report["arms"].items()
        },
    }
    if args.selection_out is not None:
        result["selection"] = select(report, args.arms, args.selection_out)["chosen"]
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
