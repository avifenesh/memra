"""Decide the single-policy final result across code, prose and math."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


DOMAINS = ("code", "prose", "math")
COUNT = 24
DRAW_COUNT = 20000
FULL_SHA = "b35374d34e2a28b31cca0399e6394fa3db3b0f27e9d593b17032a7856b2a5477"
REFERENCE = "fixed-k20-d3-c0"
LAST_TOKEN = "joint-fresh-last-token"
NO_K_PRIOR = "joint-fresh-window-no-k-prior"
FRESH_FULL = "joint-fresh-only"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def rate(rows, indices):
    tokens = sum(rows[index]["tokens"] for index in indices)
    seconds = sum(rows[index]["seconds"] for index in indices)
    if tokens <= 0 or seconds <= 0:
        raise ValueError("final native request has no rate")
    return {
        "tokens": tokens, "seconds": seconds,
        "tok_s": tokens / seconds,
    }


def available_rate(rows):
    included = [
        index for index, row in enumerate(rows) if not row["loops"]
    ]
    return (
        {**rate(rows, included), "included": included}
        if included else {
            "tokens": 0, "seconds": 0, "tok_s": 0,
            "included": [],
        }
    )


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
    observed = rate(rows[candidate], included)
    reference = rate(rows[control], included)
    rng = random.Random(seed)
    draws = []
    for _ in range(DRAW_COUNT):
        sample = [rng.choice(included) for _ in included]
        draws.append(
            100 * (
                rate(rows[candidate], sample)["tok_s"]
                / rate(rows[control], sample)["tok_s"] - 1
            )
        )
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


def pooled_pair(rows, candidate, control, seed):
    indices = {
        domain: [
            index for index in range(COUNT)
            if not rows[domain][candidate][index]["loops"]
            and not rows[domain][control][index]["loops"]
        ]
        for domain in DOMAINS
    }
    if any(len(value) < COUNT - 2 for value in indices.values()):
        return {
            "status": "too-many-looped-conversations",
            "included": indices,
        }

    def pooled(label, sample):
        tokens = sum(
            rows[domain][label][index]["tokens"]
            for domain in DOMAINS for index in sample[domain]
        )
        seconds = sum(
            rows[domain][label][index]["seconds"]
            for domain in DOMAINS for index in sample[domain]
        )
        if tokens <= 0 or seconds <= 0:
            raise ValueError("pooled final native request has no rate")
        return tokens / seconds, tokens, seconds

    a, a_tokens, a_seconds = pooled(candidate, indices)
    b, b_tokens, b_seconds = pooled(control, indices)
    rng = random.Random(seed)
    draws = []
    for _ in range(DRAW_COUNT):
        sample = {
            domain: [
                rng.choice(indices[domain])
                for _ in indices[domain]
            ]
            for domain in DOMAINS
        }
        a_draw, _, _ = pooled(candidate, sample)
        b_draw, _, _ = pooled(control, sample)
        draws.append(100 * (a_draw / b_draw - 1))
    draws.sort()
    return {
        "status": "paired", "included": indices,
        "candidate_tok_s": a, "control_tok_s": b,
        "delta_percent": 100 * (a / b - 1),
        "bootstrap_95_percent": [
            draws[int(0.025 * DRAW_COUNT)],
            draws[int(0.975 * DRAW_COUNT)],
        ],
        "output_token_ratio": a_tokens / b_tokens,
        "elapsed_ratio": a_seconds / b_seconds,
    }


def native_rows(root, domain, labels, gpu_uuid):
    rows = {}
    for label in labels:
        values = [
            json.loads((root / (
                f"final-{domain}-{index}-{label}.result.json"
            )).read_text())
            for index in range(COUNT)
        ]
        if any(
            row["name"] != f"final-{domain}-{index}-{label}"
            or row["cached_later_turns"] != 7
            or row["seconds"] <= 0
            or row["tokens"] <= 0
            or row["gpu_uuid"] != gpu_uuid
            for index, row in enumerate(values)
        ):
            raise ValueError("final native continuation differs")
        rows[label] = values
    return rows


def noops_identical(root, domain, noops):
    for index in range(COUNT):
        baseline = root / f"final-{domain}-{index}-{REFERENCE}"
        for turn in range(1, 9):
            expected = (baseline / f"turn-{turn}.output.ids").read_bytes()
            for label in noops:
                observed = (
                    root / f"final-{domain}-{index}-{label}"
                    / f"turn-{turn}.output.ids"
                ).read_bytes()
                if observed != expected:
                    raise ValueError("final model-running no-op changed output")


def task_pass(tasks, domain, label):
    sessions = {
        item["session"]: item
        for item in tasks["domains"][domain]["sessions"]
    }
    if len(sessions) != COUNT * len(tasks["labels"]):
        raise ValueError("final hidden task grader inventory differs")
    return sum(
        sessions[f"final-{domain}-{index}-{label}"]["task_pass"]
        for index in range(COUNT)
    )


def behavior(rows):
    k = Counter()
    d = Counter()
    decisions = 0
    stops = 0
    accepted = 0
    drafted = 0
    model_seconds = {"k": 0.0, "cd": 0.0}
    included = {}
    for domain in DOMAINS:
        included[domain] = 0
        for row in rows[domain]:
            if row["loops"]:
                continue
            included[domain] += 1
            k.update({
                int(action): count
                for action, count in row["k_actions"].items()
            })
            d.update({
                int(action): count
                for action, count in row["d_actions"].items()
            })
            decisions += row["c_decisions"]
            stops += row["c_stops"]
            accepted += row["accepted"]
            drafted += row["drafted"]
            model_seconds["k"] += row["k_model_s"]
            model_seconds["cd"] += row["cd_model_s"]
    return {
        "unlooped_conversations": included,
        "k_actions": dict(sorted(k.items())),
        "d_actions": dict(sorted(d.items())),
        "c_decisions": decisions, "c_stops": stops,
        "accepted_diagnostic": accepted,
        "drafted_diagnostic": drafted,
        "acceptance_diagnostic": (
            accepted / drafted if drafted else None
        ),
        "policy_model_seconds": model_seconds,
        "adaptive_k_observed": len(k) > 1,
        "adaptive_d_observed": len(d) > 1,
        "adaptive_c_observed": 0 < stops < decisions,
    }


def qualify_quality(domain, rows, candidate, controls,
                    prose_best, tasks, prose):
    candidate_loops = sum(
        row["loops"] for row in rows[candidate]
    )
    candidate_caps = sum(
        reason == "length"
        for row in rows[candidate] for reason in row["finished"]
    )
    details = {}
    for control in sorted(controls):
        control_loops = sum(row["loops"] for row in rows[control])
        control_caps = sum(
            reason == "length"
            for row in rows[control] for reason in row["finished"]
        )
        eligible = (
            candidate_loops <= control_loops
            and candidate_caps <= control_caps + 9
        )
        result = {
            "candidate_loop_hits": candidate_loops,
            "control_loop_hits": control_loops,
            "candidate_capped_turns": candidate_caps,
            "control_capped_turns": control_caps,
        }
        if domain == "prose":
            comparison = prose["comparisons"].get(
                f"{candidate}::vs::{control}"
            )
            if comparison is None or comparison["task_count"] != 192:
                raise ValueError("final prose checklist pair differs")
            point = comparison["point_win_fraction"]
            lower = comparison["bootstrap_95_win_fraction"][0]
            result.update({
                "point_win_fraction": point,
                "bootstrap_95_win_fraction":
                comparison["bootstrap_95_win_fraction"],
            })
            eligible = eligible and point >= 0.5
            if control == prose_best:
                eligible = eligible and lower >= 0.5
        else:
            a = task_pass(tasks, domain, candidate)
            b = task_pass(tasks, domain, control)
            result.update({
                "candidate_task_pass": a,
                "control_task_pass": b,
            })
            eligible = eligible and a + 9 >= b
        result["eligible"] = eligible
        details[control] = result
    return details


def decide(result):
    if any(
        comparison["status"] != "paired"
        or comparison["bootstrap_95_percent"][0] <= 0
        for comparison in result["pooled_comparisons"].values()
    ):
        return "global-no-go"
    for domain, report in result["domains"].items():
        comparison = report["vs_validation_best_fixed"]
        if (
            comparison["status"] != "paired"
            or comparison["bootstrap_95_percent"][0] < 0
            or not all(
                detail["eligible"]
                for detail in report["quality"].values()
            )
        ):
            return "global-no-go"
    actions = result["observed_behavior"]
    if not all(actions[key] for key in (
        "adaptive_k_observed",
        "adaptive_d_observed",
        "adaptive_c_observed",
    )):
        return "global-no-go"
    return "bounded-one-policy-win"


def score(args):
    if sha(args.workloads / "manifest.json") != FULL_SHA:
        raise ValueError("one-policy final workload changed")
    arms = json.loads(args.arms.read_text())
    selected_path = args.arms.with_name("shared-selected.json")
    selected = json.loads(selected_path.read_text())
    tasks = json.loads(args.tasks.read_text())
    prose = json.loads(args.prose.read_text())
    meta = json.loads(args.run_meta.read_text())
    labels = {item["label"] for item in arms["arms"]}
    candidate = selected["selected_policy"]["label"]
    noop = next(
        item["noop_label"] for item in arms["arms"]
        if item["label"] == candidate
    )
    global_fixed = selected["global_fixed"]
    best = selected["domain_best_fixed_diagnostic"]
    if (
        arms["schema"] != 1
        or arms["phase"] != "final"
        or arms["source_manifest_sha256"] != FULL_SHA
        or arms["model_manifest_sha256"]
        != selected["model_manifest_sha256"]
        or arms["selected_from_validation"] != sha(selected_path)
        or selected["status"] != "selected"
        or selected["scope"]
        != "one immutable C/K/D controller, no domain route"
        or selected["gpu_uuid"] != meta["gpu_uuid"]
        or meta["customer_capture"] is not False
        or set(best) != set(DOMAINS)
        or labels != set(selected["final_arm_labels"])
        or not {candidate, noop, global_fixed, REFERENCE}.issubset(labels)
        or not {
            LAST_TOKEN, NO_K_PRIOR, FRESH_FULL,
        }.issubset(labels)
        or tasks["schema"] != 1
        or tasks["phase"] != "final"
        or tasks["arms_sha256"] != sha(args.arms)
        or tasks["workloads_sha256"] != FULL_SHA
        or prose["schema"] != 1
        or prose["phase"] != "final"
        or prose["arms_sha256"] != sha(args.arms)
        or prose["workloads_sha256"] != FULL_SHA
        or prose["judge_config_sha256"]
        != selected["judge_config_sha256"]
    ):
        raise ValueError("final quality or selected arm lineage differs")
    tasks["labels"] = sorted(labels)
    noops = [
        item["label"] for item in arms["arms"]
        if item["role"] == "noop"
    ]
    native = {
        domain: native_rows(
            args.root, domain, labels, meta["gpu_uuid"],
        )
        for domain in DOMAINS
    }
    domains = {}
    for domain in DOMAINS:
        noops_identical(args.root, domain, noops)
        controls = {global_fixed, best[domain]}
        controls.update(
            label for label in (
                LAST_TOKEN, NO_K_PRIOR, FRESH_FULL,
            )
            if candidate != label
        )
        quality = qualify_quality(
            domain, native[domain], candidate,
            controls, best[domain], tasks, prose,
        )
        domains[domain] = {
            "candidate_rate": available_rate(
                native[domain][candidate],
            ),
            "global_fixed_rate": available_rate(
                native[domain][global_fixed],
            ),
            "last_token_rate": available_rate(
                native[domain][LAST_TOKEN],
            ),
            "window_no_k_prior_rate": available_rate(
                native[domain][NO_K_PRIOR],
            ),
            "fresh_full_rate": available_rate(
                native[domain][FRESH_FULL],
            ),
            "quality": quality,
            "vs_validation_best_fixed": paired(
                native[domain], candidate, best[domain],
                26092641,
            ),
        }
    result = {
        "schema": 1, "phase": "final",
        "scope": "one Qwen C/K/D policy across code, prose and math",
        "selection_sha256": sha(selected_path),
        "gpu_uuid": meta["gpu_uuid"],
        "arms_sha256": sha(args.arms),
        "tasks_quality_sha256": sha(args.tasks),
        "prose_quality_sha256": sha(args.prose),
        "domains": domains,
        "pooled_comparisons": {
            "vs_own_noop": pooled_pair(
                native, candidate, noop, 26092642,
            ),
            "vs_global_fixed": pooled_pair(
                native, candidate, global_fixed, 26092643,
            ),
        },
        "observed_behavior": behavior({
            domain: native[domain][candidate]
            for domain in DOMAINS
        }),
        "history_ablation": {
            label: (
                {"status": "selected-control"}
                if candidate == label else {
                    "status": "paired-diagnostic",
                    "pooled": pooled_pair(
                        native, candidate, label,
                        26092644 + index,
                    ),
                    "domains": {
                        domain: paired(
                            native[domain],
                            candidate, label,
                            26092646 + index,
                        )
                        for domain in DOMAINS
                    },
                }
            )
            for index, label in enumerate(
                (LAST_TOKEN, NO_K_PRIOR, FRESH_FULL)
            )
        },
        "feature_isolation": {
            name: {
                "treatment": treatment,
                "control": control,
                "pooled": pooled_pair(
                    native, treatment, control,
                    26092650 + index,
                ),
                "domains": {
                    domain: paired(
                        native[domain],
                        treatment, control,
                        26092652 + index,
                    )
                    for domain in DOMAINS
                },
            }
            for index, (name, treatment, control) in enumerate((
                (
                    "dc_window_beyond_last_token",
                    NO_K_PRIOR, LAST_TOKEN,
                ),
                (
                    "k_prior_turn_feature",
                    FRESH_FULL, NO_K_PRIOR,
                ),
            ))
        },
    }
    result["status"] = decide(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "root", "workloads", "arms", "tasks", "prose",
        "run-meta", "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "root", "workloads", "arms", "tasks", "prose",
        "run_meta", "out",
    ):
        setattr(args, name, getattr(args, name).resolve())
    result = score(args)
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({
        "status": result["status"],
        "pooled": {
            key: value["delta_percent"]
            for key, value in result["pooled_comparisons"].items()
            if value["status"] == "paired"
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
