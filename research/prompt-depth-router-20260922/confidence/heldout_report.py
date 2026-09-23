"""Paired Qwen K=3/C and K=2 controls on preregistered held-out code."""

import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import re
import statistics

from heldout_pair import choose_c, orders
from fixed_grid import ARMS


LABELS = ("k3off", "selected", "k2off")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def quantile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo = int(index)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (index - lo)


def compare(pairs, bootstrap=True):
    if not pairs:
        return {"pairs": 0}
    rows = {label: [pair[label] for pair in pairs] for label in LABELS}
    rates = {
        label: sum(row["output_tokens"] for row in group)
        / sum(row["elapsed_s"] for row in group)
        for label, group in rows.items()
    }
    result = {"pairs": len(pairs), "arms": {}, "selected_vs": {}}
    for label, group in rows.items():
        result["arms"][label] = {
            "output_tokens": sum(row["output_tokens"] for row in group),
            "complete_request_seconds": sum(row["elapsed_s"] for row in group),
            "tokens_per_second": rates[label],
            "format_covered": sum(row["format"]["requested_format_covered"] for row in group),
            "rounds": sum(row["rounds"] for row in group),
            "confidence_shortened_rounds": sum(
                row["confidence_shortened_rounds"] for row in group
            ),
        }
    for control in ("k3off", "k2off"):
        paired = [
            100 * ((chosen["output_tokens"] / chosen["elapsed_s"])
                   / (base["output_tokens"] / base["elapsed_s"]) - 1)
            for chosen, base in zip(rows["selected"], rows[control])
        ]
        comparison = {
            "pooled_gain_percent": 100 * (rates["selected"] / rates[control] - 1),
            "paired_gain_percent": paired,
            "paired_wins": sum(value > 0 for value in paired),
            "median_paired_gain_percent": statistics.median(paired),
            "output_token_ratio": (
                result["arms"]["selected"]["output_tokens"]
                / result["arms"][control]["output_tokens"]
            ),
            "latency_ratio": (
                result["arms"]["selected"]["complete_request_seconds"]
                / result["arms"][control]["complete_request_seconds"]
            ),
        }
        if bootstrap and len(pairs) >= 3:
            rng = random.Random(20730924)
            draws = []
            for _ in range(5000):
                sampled = [rng.randrange(len(pairs)) for _ in pairs]
                selected_rate = (
                    sum(rows["selected"][i]["output_tokens"] for i in sampled)
                    / sum(rows["selected"][i]["elapsed_s"] for i in sampled)
                )
                control_rate = (
                    sum(rows[control][i]["output_tokens"] for i in sampled)
                    / sum(rows[control][i]["elapsed_s"] for i in sampled)
                )
                draws.append(100 * (selected_rate / control_rate - 1))
            comparison["paired_bootstrap_95_percent"] = [
                quantile(draws, 0.025), quantile(draws, 0.975)
            ]
        else:
            comparison["paired_bootstrap_95_percent"] = None
        result["selected_vs"][control] = comparison
    return result


def histograms(log):
    entries = re.findall(
        r"\[spec-stats\] rounds=(\d+) full_accept=\d+ len_hist=\[([0-9, ]+)\]",
        log.read_text(),
    )
    if len(entries) < 8:
        raise ValueError("held-out run has fewer than eight draft histograms")
    by_turn = []
    for rounds, values in entries[-8:]:
        row = [int(value.strip()) for value in values.split(",")]
        if len(row) != 8 or sum(row) != int(rounds):
            raise ValueError("held-out draft histogram differs from the round count")
        by_turn.append(row)
    return by_turn


def report(root, workloads, development_report):
    status = json.loads((root / "status.json").read_text())
    freeze = json.loads((root / "FREEZE.json").read_text())
    prior = json.loads((root / "PRESELECTION-FREEZE.json").read_text())
    manifest = json.loads((workloads / "manifest.json").read_text())
    development = json.loads(development_report.read_text())
    selected = choose_c(development)
    version = manifest["schema"]
    if version not in (1, 2):
        raise ValueError("unknown held-out qualifier version")
    code_only = version == 2
    generator = "code_only_v2_workloads.py" if code_only else "heldout_workloads.py"
    runner = "heldout_pair_v2.py" if code_only else "heldout_pair.py"
    if (freeze["orders"] != orders()
            or freeze["schema"] != version
            or freeze["workloads_sha256"] != sha(workloads / "manifest.json")
            or freeze["runner_sha256"] != sha(Path(__file__).with_name(runner))
            or freeze["development_report_sha256"] != sha(development_report)
            or freeze["selected_c"] != selected
            or manifest["generator_sha256"] != sha(Path(__file__).with_name(generator))
            or prior["selected_c"] is not None
            or prior["development_report_sha256"] is not None
            or prior["status"] != "registered-before-development-selection"
            or prior["workloads_sha256"] != freeze["workloads_sha256"]
            or prior["runner_sha256"] != freeze["runner_sha256"]):
        raise ValueError("held-out registration or development choice changed")
    if code_only:
        original = workloads.parent / "heldout-workloads"
        original_manifest = json.loads((original / "manifest.json").read_text())
        if (freeze["qualification_scope"] != "code-only-v2"
                or manifest["original_manifest_sha256"] != sha(original / "manifest.json")):
            raise ValueError("code-only v2 changed its versioned qualification scope")
        for index in range(6):
            previous = original_manifest["scenarios"][str(index)]
            current = manifest["scenarios"][str(index)]
            if previous != current or sha(original / previous["file"]) != sha(workloads / current["file"]):
                raise ValueError("code-only v2 changed a preregistered scenario")
    qualifier = json.loads(
        (root / "qwen/qualification/sampled-format-k3.audit.json").read_text()
    )
    relevant = (
        [row for row in qualifier["requests"] if row["kind"] == "code"]
        if code_only else qualifier["requests"]
    )
    if (len(relevant) != (4 if code_only else 8)
            or not all(row["format"]["requested_format_covered"] and not row["loop"]
                       for row in relevant)):
        raise ValueError("held-out format qualification did not pass")
    prose_covered = sum(
        row["format"]["requested_format_covered"]
        for row in qualifier["requests"] if row["kind"] == "prose"
    )
    if status["status"] == "no-positive-development-c":
        if selected is not None or len(status["completed"]) != 1:
            raise ValueError("held-out no-go status differs from the development selection")
        return {
            "status": "no-positive-development-c",
            "selected_c": None,
            "qualification": qualifier["path"],
            "qualification_scope": "code-only-v2" if code_only else "all-formats-v1",
            "qualification_prose_covered": prose_covered,
            "workloads_sha256": freeze["workloads_sha256"],
            "scope": "no held-out speed claim",
        }
    if status["status"] != "completed" or selected is None:
        raise ValueError("held-out comparison is incomplete")
    expected = {
        f"qwen/heldout/{scenario:02}-{label}"
        for scenario in range(6) for label in LABELS
    }
    required_qualification = {
        "qwen/qualification/sampled-format-k3",
        "qwen/qualification/greedy-k2off",
        "qwen/qualification/greedy-k3off",
        "qwen/qualification/greedy-selected",
    }
    if set(status["completed"]) != expected | required_qualification:
        raise ValueError("held-out scored or qualification runs are missing")
    for label in LABELS:
        oracle = root / f"oracle-{label}.log"
        if ("=== SELF-CONSISTENCY PASS ===" not in oracle.read_text()
                or json.loads((root / f"oracle-{label}.exit.json").read_text())["returncode"]):
            raise ValueError("held-out target-only oracle did not pass: " + label)
    gates = [
        root / f"qwen/qualification/greedy-{label}" for label in LABELS
    ]
    for turn in range(1, 9):
        outputs = {(gate / f"turn-{turn}.output.ids").read_bytes() for gate in gates}
        if len(outputs) != 1:
            raise ValueError("held-out greedy driver tapes differ")
    cells = collections.defaultdict(list)
    exclusions = []
    all_code = []
    for scenario in range(6):
        runs = {}
        for label in LABELS:
            stem = f"{scenario:02}-{label}"
            directory = root / "qwen/heldout"
            record = json.loads((directory / f"{stem}.audit.json").read_text())
            command = json.loads((directory / f"{stem}.command.json").read_text())
            exit_row = json.loads((directory / f"{stem}.exit.json").read_text())
            expected_k = 2 if label == "k2off" else 3
            pmin, pmin0 = ARMS[selected] if label == "selected" else (0.0, False)
            if (record["arm"] != f"fixed:{expected_k}"
                    or record["seed"] != manifest["scenarios"][str(scenario)]["seed"]
                    or record["max_new"] != 8192 or record["gate"]
                    or command["pmin"] != pmin or command["pmin0"] != pmin0
                    or command["native_adapt"] or not command["spec_stats"]
                    or exit_row["returncode"] != 0 or exit_row["contamination"]):
                raise ValueError("held-out arm or result differs: " + stem)
            lengths = histograms(directory / f"{stem}.log")
            if label == "k3off" and any(sum(row[:3]) or sum(row[4:]) for row in lengths):
                raise ValueError("K=3 confidence-off control shortened a round")
            if label == "k2off" and any(sum(row[:2]) or sum(row[3:]) for row in lengths):
                raise ValueError("K=2 confidence-off control used another depth")
            if label == "selected" and any(sum(row[4:]) for row in lengths):
                raise ValueError("selected C drafted past its K=3 ceiling")
            runs[label] = (record, directory / stem, lengths)
        for turn in range(1, 9):
            matched = {}
            for label, (record, _, lengths) in runs.items():
                row = record["requests"][turn - 1]
                hist = lengths[turn - 1]
                expected_k = 2 if label == "k2off" else 3
                if row["k"] != expected_k:
                    raise ValueError("held-out control changed fixed K")
                matched[label] = {
                    **row,
                    "rounds": sum(hist),
                    "confidence_shortened_rounds": (
                        sum(hist[:3]) if label == "selected" else 0
                    ),
                }
            key = (matched["k3off"]["kind"], matched["k3off"]["length_target"])
            if (any((row["kind"], row["length_target"]) != key for row in matched.values())
                    or len({(path / f"turn-{turn}.prompt.ids").read_bytes()
                            for _, path, _ in runs.values()}) != 1):
                raise ValueError("held-out paired request identities differ")
            if any(row["loop"] for row in matched.values()):
                exclusions.append({"scenario": scenario, "turn": turn, "kind": key[0],
                                   "length_target": key[1]})
            else:
                cells[key].append(matched)
                if key[0] == "code":
                    all_code.append(matched)
    return {
        "status": "measured-heldout-code-v2" if code_only else "measured-heldout-fixed-c",
        "selected_c": selected,
        "selected_pmin": ARMS[selected][0],
        "selected_pmin0": ARMS[selected][1],
        "qualification_scope": "code-only-v2" if code_only else "all-formats-v1",
        "qualification_code_covered": 4,
        "qualification_prose_covered": prose_covered,
        "workloads_sha256": freeze["workloads_sha256"],
        "development_report_sha256": freeze["development_report_sha256"],
        "matched_loop_exclusions": exclusions,
        "all_code": compare(all_code, bootstrap=False),
        "code_by_prompt_tokens": {
            str(length): compare(cells["code", length])
            for length in (256, 1024, 4096, 16384)
        },
        "all_arms_format_covered_code": {
            str(length): compare([
                pair for pair in cells["code", length]
                if all(row["format"]["requested_format_covered"] for row in pair.values())
            ])
            for length in (256, 1024, 4096, 16384)
        },
        "scope": "six disjoint synthetic scenarios, requested-code native rate; prose diagnostic only under v2; no online C, warm-KV or HTTP claim",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.root, args.workloads, args.development_report)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "selected_c": result["selected_c"],
        "code_pairs": result.get("all_code", {}).get("pairs", 0),
    }))


if __name__ == "__main__":
    main()
