"""Score mixed native validation on paired complete-request tok/s."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


DOMAINS = ("code", "prose", "math")
COUNT = 8
REFERENCE = "fixed-k20-d3-c0"
WORKLOAD_SHA = "bf920b82e0176c4304bcc562ccce34a28384082c888355a280620163a45e5313"
DRAW_COUNT = 20000


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def pooled(rows, indices):
    tokens = sum(rows[index]["tokens"] for index in indices)
    seconds = sum(rows[index]["seconds"] for index in indices)
    if not tokens or seconds <= 0:
        raise ValueError("mixed native request has no rate")
    return {
        "tokens": tokens, "seconds": seconds,
        "tok_s": tokens / seconds,
    }


def paired(rows, candidate, control, seed):
    included = [
        index for index in range(COUNT)
        if not rows[candidate][index]["loops"]
        and not rows[control][index]["loops"]
    ]
    if len(included) < COUNT - 2:
        return {
            "status": "too-many-looped-conversations",
            "included": included,
        }
    observed = pooled(rows[candidate], included)
    reference = pooled(rows[control], included)
    rng = random.Random(seed)
    draws = []
    for _ in range(DRAW_COUNT):
        sample = [rng.choice(included) for _ in included]
        draws.append(100 * (
            pooled(rows[candidate], sample)["tok_s"]
            / pooled(rows[control], sample)["tok_s"] - 1
        ))
    draws.sort()
    return {
        "status": "paired", "included": included,
        "delta_percent":
        100 * (observed["tok_s"] / reference["tok_s"] - 1),
        "bootstrap_95_percent": [
            draws[int(0.025 * DRAW_COUNT)],
            draws[int(0.975 * DRAW_COUNT)],
        ],
        "output_token_ratio": observed["tokens"] / reference["tokens"],
        "elapsed_ratio": observed["seconds"] / reference["seconds"],
    }


def actions(rows, key):
    counts = Counter()
    for row in rows:
        counts.update({
            int(action): n for action, n in row[key].items()
        })
    return dict(sorted(counts.items()))


def identity(root, domain, noops):
    for index in range(COUNT):
        baseline = root / f"validation-{domain}-{index}-{REFERENCE}"
        for turn in range(1, 9):
            expected = (baseline / f"turn-{turn}.output.ids").read_bytes()
            for label in noops:
                if (
                    root / f"validation-{domain}-{index}-{label}"
                    / f"turn-{turn}.output.ids"
                ).read_bytes() != expected:
                    raise ValueError(
                        f"{domain} model-running no-op output changed"
                    )


def report(root, domain, arms, tasks, prose):
    by_label = {item["label"]: item for item in arms["arms"]}
    labels = list(by_label)
    noops = [
        label for label in labels
        if by_label[label]["role"] == "noop"
    ]
    identity(root, domain, noops)
    if domain != "prose":
        graded = {
            item["session"]: item
            for item in tasks["domains"][domain]["sessions"]
        }
        if len(graded) != COUNT * len(labels):
            raise ValueError(f"{domain} task grader inventory differs")
    rows = {}
    scores = {}
    for label in labels:
        native = [
            json.loads((root / (
                f"validation-{domain}-{index}-{label}.result.json"
            )).read_text())
            for index in range(COUNT)
        ]
        if any(
            row["name"] != f"validation-{domain}-{index}-{label}"
            or row["cached_later_turns"] != 7
            or row["seconds"] <= 0
            or row["tokens"] <= 0
            for index, row in enumerate(native)
        ):
            raise ValueError(f"{domain} native continuation differs")
        rows[label] = native
        included = [
            index for index, row in enumerate(native)
            if not row["loops"]
        ]
        total = pooled(native, included) if included else {
            "tokens": 0, "seconds": 0, "tok_s": 0,
        }
        finished = Counter(
            reason for row in native for reason in row["finished"]
        )
        drafted = sum(row["drafted"] for row in native)
        scores[label] = {
            **total,
            "task_pass": (
                sum(
                    graded[f"validation-{domain}-{index}-{label}"][
                        "task_pass"
                    ]
                    for index in range(COUNT)
                ) if domain != "prose" else None
            ),
            "capped_turns": finished["length"],
            "finish_reasons": dict(sorted(finished.items())),
            "loop_hits": sum(row["loops"] for row in native),
            "looped_conversations": [
                index for index, row in enumerate(native)
                if row["loops"]
            ],
            "k_actions": actions(native, "k_actions"),
            "d_actions": actions(native, "d_actions"),
            "c_decisions": sum(row["c_decisions"] for row in native),
            "c_stops": sum(row["c_stops"] for row in native),
            "k_model_seconds": sum(row["k_model_s"] for row in native),
            "cd_model_seconds": sum(row["cd_model_s"] for row in native),
            "accepted": sum(row["accepted"] for row in native),
            "drafted": drafted,
            "acceptance_diagnostic": (
                sum(row["accepted"] for row in native) / drafted
                if drafted else None
            ),
        }
    baseline = scores[REFERENCE]
    if len(baseline["looped_conversations"]) > 2:
        raise ValueError(f"{domain} baseline has too many looped sessions")
    for label in labels:
        item = scores[label]
        eligible = (
            len(item["looped_conversations"])
            <= len(baseline["looped_conversations"])
            and item["capped_turns"] <= baseline["capped_turns"] + 3
            and item["tok_s"] > 0
        )
        if domain == "prose":
            if label not in (REFERENCE, *noops):
                comparison = prose["comparisons"].get(
                    f"{label}::vs::{REFERENCE}"
                )
                if comparison is None or comparison["task_count"] != 64:
                    raise ValueError("prose quality comparison missing")
                item["prose_win_fraction_vs_reference"] = (
                    comparison["point_win_fraction"]
                )
                eligible = eligible and (
                    comparison["point_win_fraction"] >= 0.5
                )
        else:
            eligible = eligible and (
                item["task_pass"] + 3 >= baseline["task_pass"]
            )
        item["quality_eligible"] = eligible
    fixed = [
        label for label in labels
        if by_label[label]["role"] == "fixed"
        and scores[label]["quality_eligible"]
    ]
    if REFERENCE not in fixed:
        raise ValueError(f"{domain} fixed reference is ineligible")
    comparisons = {}
    for label in labels:
        if by_label[label]["role"] != "learned":
            continue
        noop = by_label[label]["noop_label"]
        comparisons[label] = {
            "vs_own_noop": paired(
                rows, label, noop, 26092601,
            ),
            "vs_each_eligible_fixed": {
                fixed_label: paired(
                    rows, label, fixed_label, 26092601,
                )
                for fixed_label in fixed
            },
        }
    return {
        "arms": scores, "eligible_fixed": fixed,
        "byte_identical_noops": noops,
        "comparisons": comparisons,
        **({
            "judge_receipt_sha256": prose["judge_results_manifest_sha256"],
        } if domain == "prose" else {}),
    }


def score(args):
    if sha(args.workloads / "manifest.json") != WORKLOAD_SHA:
        raise ValueError("mixed validation workload changed")
    arms = json.loads(args.arms.read_text())
    tasks = json.loads(args.tasks.read_text())
    prose = json.loads(args.prose.read_text())
    if (
        arms["schema"] != 1
        or arms["phase"] != "validation"
        or tasks["schema"] != 1
        or tasks["phase"] != "validation"
        or tasks["arms_sha256"] != sha(args.arms)
        or tasks["workloads_sha256"] != WORKLOAD_SHA
        or prose["schema"] != 1
        or prose["phase"] != "validation"
        or prose["arms_sha256"] != sha(args.arms)
        or prose["workloads_sha256"] != WORKLOAD_SHA
        or prose["packets_manifest_sha256"] == ""
    ):
        raise ValueError("mixed validation quality receipt differs")
    quality = hashlib.sha256(
        (sha(args.tasks) + sha(args.prose)).encode()
    ).hexdigest()
    return {
        "schema": 1, "phase": "validation",
        "scope": "one mixed Qwen C/K/D arm menu, complete native request tok/s",
        "arms_sha256": sha(args.arms),
        "model_manifest_sha256": arms["model_manifest_sha256"],
        "source_manifest_sha256": WORKLOAD_SHA,
        "quality_sha256": quality,
        "domains": {
            domain: report(
                args.root, domain, arms, tasks, prose,
            )
            for domain in DOMAINS
        },
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "workloads", "arms", "tasks", "prose", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for name in ("root", "workloads", "arms", "tasks", "prose", "out"):
        setattr(args, name, getattr(args, name).resolve())
    result = score(args)
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({
        "phase": "validation",
        "eligible_fixed": {
            domain: report["eligible_fixed"]
            for domain, report in result["domains"].items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
