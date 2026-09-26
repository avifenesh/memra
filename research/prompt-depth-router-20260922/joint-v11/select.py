"""Select bounded final v11 arms using validation receipts only."""

import argparse
import hashlib
import json
from pathlib import Path


DOMAINS = ("ifeval", "gsm8k")
CD_OPTIONS = ("cd-noncode-only", "cd-mixed", "cd-augmented")
OTHER_OPTIONS = ("joint-mixed", "k-mixed", "k-noncode-only")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arm_digest(arms):
    return hashlib.sha256(json.dumps(
        arms, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def save(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def noop_for(label):
    for kind in ("cd-", "joint-", "k-"):
        if label.startswith(kind):
            return kind + "noop-" + label.removeprefix(kind)
    raise ValueError(f"not a learned policy: {label}")


def beats_all_fixed(result, label, eligible):
    comparisons = result["comparisons"][label]["vs_each_eligible_fixed"]
    return all(
        comparisons[fixed]["status"] == "paired"
        and comparisons[fixed]["delta_percent"] > 0
        for fixed in eligible
    )


def choose(score_path, arms_path, qualifier_path):
    report = json.loads(score_path.read_text())
    arms = json.loads(arms_path.read_text())
    qualifier = json.loads(qualifier_path.read_text())
    labels = [item["label"] for item in arms["arms"]]
    if (
        report["schema"] != 1
        or report["phase"] != "validation"
        or report["arms_sha256"] != sha(arms_path)
        or arms["schema"] != 1
        or arms["phase"] != "validation"
        or len(labels) != 19
        or len(set(labels)) != 19
        or qualifier["status"] != "qualified"
        or qualifier["arm_inventory_sha256"] != arm_digest(arms["arms"])
        or qualifier["model_manifest_sha256"]
        != arms["model_manifest_sha256"]
        or qualifier["training_manifest_sha256"]
        != arms["training_manifest_sha256"]
        or set(report["domains"]) != set(DOMAINS)
    ):
        raise ValueError("v11 validation source or qualifier differs")
    fixed = [
        item for item in arms["arms"] if item["label"].startswith("fixed-")
    ]
    if len(fixed) != 7:
        raise ValueError("v11 final must retain seven fixed controls")
    chosen = {}
    learned = set()
    for domain in DOMAINS:
        result = report["domains"][domain]
        eligible = [
            label for label in result["eligible_fixed"]
            if result["arms"][label]["tok_s"] is not None
        ]
        if not eligible or "fixed-k20-d3-c0" not in eligible:
            raise ValueError("validation lacks quality-eligible fixed reference")
        best_fixed = max(
            eligible, key=lambda label: result["arms"][label]["tok_s"]
        )
        cd = [
            label for label in CD_OPTIONS
            if result["arms"][label]["quality_eligible"]
            and result["comparisons"][label]["vs_own_noop"]["status"]
            == "paired"
            and result["comparisons"][label]["vs_own_noop"][
                "delta_percent"
            ] > 0
            and beats_all_fixed(result, label, eligible)
        ]
        best_cd = (
            max(
                cd,
                key=lambda label: result["comparisons"][label][
                    "vs_each_eligible_fixed"
                ][best_fixed]["delta_percent"],
            )
            if cd else None
        )
        candidate = [best_cd] if best_cd is not None else []
        if best_cd is not None:
            learned.add(best_cd)
        for label in OTHER_OPTIONS:
            if (
                result["arms"][label]["quality_eligible"]
                and result["comparisons"][label]["vs_own_noop"]["status"]
                == "paired"
                and result["comparisons"][label]["vs_own_noop"][
                    "delta_percent"
                ] > 0
                and beats_all_fixed(result, label, eligible)
            ):
                learned.add(label)
                candidate.append(label)
        primary = (
            max(
                candidate,
                key=lambda label: result["comparisons"][label][
                    "vs_each_eligible_fixed"
                ][best_fixed]["delta_percent"],
            )
            if candidate else None
        )
        chosen[domain] = {
            "fixed": best_fixed, "cd": best_cd, "primary": primary,
        }
    selected_labels = {
        *(item["label"] for item in fixed),
        *learned,
        *(noop_for(label) for label in learned),
    }
    final = [
        item for item in arms["arms"] if item["label"] in selected_labels
    ] if learned else []
    if learned and {item["label"] for item in final} != selected_labels:
        raise ValueError("selected learned no-op or fixed arm absent")
    return {
        "schema": 1,
        "phase": "validation-selected",
        "status": "selected" if learned else "no-go",
        "reason": None if learned else "no learned arm passed validation point checks",
        "validation_score_sha256": sha(score_path),
        "validation_arms_sha256": sha(arms_path),
        "qualification_arm_inventory_sha256": arm_digest(arms["arms"]),
        "model_manifest_sha256": arms["model_manifest_sha256"],
        "training_manifest_sha256": arms["training_manifest_sha256"],
        "chosen_by_domain": chosen,
        "final_arms": sorted(selected_labels) if learned else [],
        "final_arm_inventory_sha256": arm_digest(final),
    }, final, arms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--qualifier", type=Path, required=True)
    args = parser.parse_args()
    if args.score.resolve() != args.arms.with_name(
        "validation-score.json"
    ).resolve():
        raise ValueError("validation score must be frozen beside arm manifest")
    selected_path = args.arms.with_name("selected.json")
    final_path = args.arms.with_name("heldout-arms.json")
    if selected_path.exists() or final_path.exists():
        raise ValueError("v11 final selection already frozen")
    selection, final, arms = choose(
        args.score, args.arms, args.qualifier
    )
    save(selected_path, selection)
    if selection["status"] == "selected":
        save(final_path, {
            "schema": 1,
            "phase": "heldout",
            "model_manifest_sha256": arms["model_manifest_sha256"],
            "training_manifest_sha256": arms["training_manifest_sha256"],
            "selected_from_validation": sha(selected_path),
            "arms": final,
        })
    print(json.dumps({
        "status": selection["status"],
        "final_arms": selection["final_arms"],
        "chosen_by_domain": selection["chosen_by_domain"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
