"""Summarize complete native throughput and randomized D costs on development."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


def table(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def score(root):
    summary = json.loads((root / "training-summary.json").read_text())
    if summary["schema"] != 1 or summary["phase"] != "training":
        raise ValueError("not a v4 native training summary")
    fixed = defaultdict(list)
    for record in summary["records"]:
        if record["arm"] != "explore-d":
            fixed[record["variant"]].append(record)
    if any(len(records) != 6 for records in fixed.values()):
        raise ValueError("incomplete matched fixed-arm development grid")
    fixed_score = {}
    for name, records in fixed.items():
        tokens = sum(row["tokens"] for row in records)
        seconds = sum(row["seconds"] for row in records)
        fixed_score[name] = {
            "tokens": tokens,
            "seconds": seconds,
            "tok_s": tokens / seconds,
            "format_pass": sum(row["format"] for row in records),
            "loops": sum(row["loops"] for row in records),
        }
    qualified = {
        name: row for name, row in fixed_score.items()
        if row["format_pass"] == 48 and row["loops"] == 0
    }
    if "k3-c0" not in qualified:
        raise ValueError("K3/C0 development control failed the format/loop gate")
    selected = max(qualified, key=lambda name: qualified[name]["tok_s"])
    lam = qualified[selected]["tok_s"]
    trials = []
    for record in summary["records"]:
        if record["arm"] != "explore-d":
            continue
        session = root / record["name"]
        all_rounds = table(session / "rounds.tsv")
        confidence = {
            (turn, int(row["round"])): row
            for turn in range(1, 9)
            for row in table(session / f"turn-{turn}.confidence.tsv")
        }
        spans = {
            (int(row["turn"]), int(row["round"])): row
            for row in table(session / "spans.tsv")
        }
        if len(confidence) != len(all_rounds) or len(spans) != len(all_rounds):
            raise ValueError("randomized D receipt inventories disagree")
        prior = None
        for row in all_rounds:
            key = (int(row["turn"]), int(row["round"]))
            observed, span = confidence[key], spans[key]
            if (
                observed["eligible"] != row["eligible_for_learning"]
                or int(observed["elapsed_ns"]) != int(row["elapsed_ns"])
                or int(observed["drafted"]) + 1 != int(row["draft_depth"])
                or int(span["k"]) != int(row["draft_depth"])
            ):
                raise ValueError("randomized D receipt fields disagree")
            if row["eligible_for_learning"] == "true":
                trials.append({
                    "session": record["name"],
                    "turn": key[0], "round": key[1],
                    "d": int(observed["drafted"]),
                    "accepted": int(observed["accepted_prefix"]),
                    "emitted": int(row["emitted"]),
                    "elapsed_s": int(row["elapsed_ns"]) / 1e9,
                    "phase": span["context_before"],
                    "prior_accept_band": (
                        "none" if prior is None
                        else "high" if prior >= 0.5 else "low"
                    ),
                })
            prior = int(observed["accepted_prefix"]) / int(observed["drafted"])
    if len({row["session"] for row in trials}) != 6:
        raise ValueError("randomized D lacks all six training conversations")

    def aggregate(rows):
        by_depth = defaultdict(list)
        for row in rows:
            by_depth[row["d"]].append(row)
        result = {}
        for d in (1, 2, 3, 4):
            selected_rows = by_depth[d]
            if not selected_rows:
                raise ValueError(f"D={d} received no eligible randomized rounds")
            tokens = sum(row["emitted"] for row in selected_rows)
            seconds = sum(row["elapsed_s"] for row in selected_rows)
            result[str(d)] = {
                "rounds": len(selected_rows),
                "accepted_per_round": sum(
                    row["accepted"] for row in selected_rows
                ) / len(selected_rows),
                "emitted_per_round": tokens / len(selected_rows),
                "seconds_per_round": seconds / len(selected_rows),
                "round_tok_s": tokens / seconds,
                "utility_per_round": (tokens - lam * seconds) / len(selected_rows),
            }
        return result

    phases = defaultdict(list)
    bands = defaultdict(list)
    for row in trials:
        phases[row["phase"]].append(row)
        bands[row["prior_accept_band"]].append(row)
    return {
        "schema": 1,
        "status": "development randomized assignment; end-to-end fixed-grid and round diagnostics",
        "fixed": fixed_score,
        "strongest_quality_qualified_fixed": selected,
        "fixed_baseline_tok_s": lam,
        "randomized_d": aggregate(trials),
        "by_output_phase": {
            name: aggregate(rows) for name, rows in phases.items()
            if {row["d"] for row in rows} == {1, 2, 3, 4}
        },
        "by_previous_acceptance": {
            name: aggregate(rows) for name, rows in bands.items()
            if {row["d"] for row in rows} == {1, 2, 3, 4}
        },
        "randomized_eligible_rounds": len(trials),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.root)
    with args.out.open("x") as target:
        json.dump(result, target, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps({
        "fixed": result["strongest_quality_qualified_fixed"],
        "randomized_eligible_rounds": result["randomized_eligible_rounds"],
    }))


if __name__ == "__main__":
    main()
