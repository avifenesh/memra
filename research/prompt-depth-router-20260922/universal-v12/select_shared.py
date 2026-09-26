"""Freeze one mixed-domain C/K/D policy from validation, or fail closed."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


DOMAINS = ("code", "prose", "math")
VALIDATION_SHA = "bf920b82e0176c4304bcc562ccce34a28384082c888355a280620163a45e5313"
PAIR_STATUS = "paired"
BASELINE = "fixed-k20-d3-c0"
REQUIRED_FIXED = {
    BASELINE, "fixed-k3-d3-c0", "fixed-k10-d3-c0",
    "fixed-k20-d1-c0", "fixed-k20-d2-c0", "fixed-k20-d4-c0",
}
SHA = re.compile(r"[a-f0-9]{64}")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def paired_delta(value):
    if value.get("status") != PAIR_STATUS:
        return None
    delta = value.get("delta_percent")
    return (
        float(delta)
        if isinstance(delta, (int, float)) and math.isfinite(delta)
        else None
    )


def pooled(domains, label):
    tokens = sum(domains[domain]["arms"][label]["tokens"] for domain in DOMAINS)
    seconds = sum(domains[domain]["arms"][label]["seconds"] for domain in DOMAINS)
    if not positive(tokens) or not positive(seconds):
        raise ValueError(f"{label} lacks complete native request time")
    return tokens / seconds


def inspect(validation, arms_path):
    score = json.loads(validation.read_text())
    arms = json.loads(arms_path.read_text())
    specs = arms["arms"]
    labels = [item["label"] for item in specs]
    by_label = {item["label"]: item for item in specs}
    if (
        score["schema"] != 1
        or score["phase"] != "validation"
        or score["arms_sha256"] != sha(arms_path)
        or score["source_manifest_sha256"] != VALIDATION_SHA
        or arms["schema"] != 1
        or arms["phase"] != "validation"
        or set(score["domains"]) != set(DOMAINS)
        or len(labels) != len(by_label)
        or len(labels) < 8
        or not SHA.fullmatch(score["quality_sha256"])
        or not SHA.fullmatch(score["judge_config_sha256"])
        or not SHA.fullmatch(arms["source_manifest_sha256"])
        or not SHA.fullmatch(arms["model_manifest_sha256"])
        or not SHA.fullmatch(arms["qualification_arms_sha256"])
        or score["model_manifest_sha256"]
        != arms["model_manifest_sha256"]
    ):
        raise ValueError("mixed validation source, quality or arms differ")
    fixed = {
        label for label, item in by_label.items()
        if item["role"] == "fixed"
    }
    learned = {
        label for label, item in by_label.items()
        if item["role"] == "learned"
    }
    noops = {
        label for label, item in by_label.items()
        if item["role"] == "noop"
    }
    if not REQUIRED_FIXED.issubset(fixed) or not learned or not noops:
        raise ValueError("mixed validation lacks bounded fixed or learned controls")
    selectable = {
        label for label in learned
        if by_label[label].get("selectable") is True
        and by_label[label].get("arm") == "joint-ckd"
    }
    if not selectable or any(
        by_label[label].get("selectable") not in (False, True)
        for label in learned
    ):
        raise ValueError("mixed validation has no shared C/K/D candidate")
    if any(
        not SHA.fullmatch(by_label[label]["policy_sha256"])
        or by_label[label]["noop_label"] not in noops
        or by_label[by_label[label]["noop_label"]]["policy_sha256"]
        != by_label[label]["policy_sha256"]
        for label in learned
    ):
        raise ValueError("learned arm and model-running no-op differ")
    for domain in DOMAINS:
        report = score["domains"][domain]
        if (
            set(report["arms"]) != set(labels)
            or not set(report["eligible_fixed"]).issubset(fixed)
            or BASELINE not in report["eligible_fixed"]
            or not set(by_label[label]["noop_label"] for label in learned)
            .issubset(set(report["byte_identical_noops"]))
        ):
            raise ValueError(f"{domain} arm inventory or no-op identity differs")
        for label in report["eligible_fixed"]:
            row = report["arms"][label]
            if (
                not row["quality_eligible"]
                or not positive(row["tokens"])
                or not positive(row["seconds"])
                or not positive(row["tok_s"])
            ):
                raise ValueError(f"{domain} fixed quality or rate differs: {label}")
    if not SHA.fullmatch(score["domains"]["prose"]["judge_receipt_sha256"]):
        raise ValueError("prose quality lacks pinned checklist judge")
    return score, arms, by_label, fixed, learned, selectable


def choose(validation, arms_path):
    score, arms, by_label, fixed, learned, selectable = inspect(
        validation, arms_path,
    )
    domains = score["domains"]
    global_fixed_options = [
        label for label in fixed
        if all(label in domains[domain]["eligible_fixed"] for domain in DOMAINS)
    ]
    if not global_fixed_options:
        raise ValueError("no globally quality-eligible fixed control")
    global_fixed = max(
        sorted(global_fixed_options),
        key=lambda label: pooled(domains, label),
    )
    domain_best_fixed = {
        domain: max(
            sorted(domains[domain]["eligible_fixed"]),
            key=lambda label: domains[domain]["arms"][label]["tok_s"],
        )
        for domain in DOMAINS
    }
    candidates = []
    for label in sorted(selectable):
        margins = {}
        for domain in DOMAINS:
            result = domains[domain]
            row = result["arms"][label]
            own = paired_delta(
                result["comparisons"][label]["vs_own_noop"]
            )
            fixed_pairs = result["comparisons"][label][
                "vs_each_eligible_fixed"
            ]
            if (
                not row["quality_eligible"]
                or own is None or own <= 0
                or set(fixed_pairs) != set(result["eligible_fixed"])
            ):
                break
            deltas = {
                fixed_label: paired_delta(fixed_pairs[fixed_label])
                for fixed_label in result["eligible_fixed"]
            }
            if any(value is None or value <= 0 for value in deltas.values()):
                break
            margins[domain] = deltas[domain_best_fixed[domain]]
        if len(margins) != len(DOMAINS):
            continue
        pooled_margin = 100 * (
            pooled(domains, label) / pooled(domains, global_fixed) - 1
        )
        if pooled_margin <= 0:
            continue
        candidates.append({
            "label": label,
            "policy_sha256": by_label[label]["policy_sha256"],
            "domain_margin_percent": margins,
            "worst_domain_margin_percent": min(margins.values()),
            "pooled_margin_percent": pooled_margin,
        })
    chosen = (
        max(
            candidates,
            key=lambda row: (
                row["worst_domain_margin_percent"],
                row["pooled_margin_percent"],
                row["label"],
            ),
        )
        if candidates else None
    )
    final_labels = (
        {
            chosen["label"],
            by_label[chosen["label"]]["noop_label"],
            BASELINE,
            global_fixed,
            *domain_best_fixed.values(),
        }
        if chosen else set()
    )
    final_arms = [
        item for item in arms["arms"] if item["label"] in final_labels
    ]
    if chosen and {item["label"] for item in final_arms} != final_labels:
        raise ValueError("one shared final arm or control is missing")
    record = {
        "schema": 1,
        "status": "selected" if chosen else "global-no-go",
        "scope": "one immutable C/K/D controller, no domain route",
        "validation_score_sha256": sha(validation),
        "validation_arms_sha256": sha(arms_path),
        "source_manifest_sha256": arms["source_manifest_sha256"],
        "model_manifest_sha256": arms["model_manifest_sha256"],
        "quality_sha256": score["quality_sha256"],
        "judge_config_sha256": score["judge_config_sha256"],
        "global_fixed": global_fixed,
        "domain_best_fixed_diagnostic": domain_best_fixed,
        "selected_policy": chosen,
        "candidate_inventory": candidates,
        "final_arm_labels": sorted(final_labels),
    }
    return record, final_arms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    args = parser.parse_args()
    validation = args.validation.resolve()
    arms = args.arms.resolve()
    if validation != arms.with_name("validation-score.json"):
        raise ValueError("mixed validation score must be frozen by its arms")
    result_path = arms.with_name("shared-selected.json")
    final_path = arms.with_name("final-arms.json")
    if result_path.exists() or final_path.exists():
        raise ValueError("shared policy was already selected")
    selection, final = choose(validation, arms)
    save(result_path, selection)
    if selection["status"] == "selected":
        save(final_path, {
            "schema": 1,
            "phase": "final",
            "source_manifest_sha256":
            selection["source_manifest_sha256"],
            "model_manifest_sha256":
            selection["model_manifest_sha256"],
            "selected_from_validation": sha(result_path),
            "qualification_arms_sha256":
            arms["qualification_arms_sha256"],
            "domains": list(DOMAINS),
            "arms": final,
        })
    print(json.dumps({
        "status": selection["status"],
        "selected_policy": (
            selection["selected_policy"]["label"]
            if selection["selected_policy"] else None
        ),
        "final_arm_labels": selection["final_arm_labels"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
