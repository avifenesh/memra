"""Audit and summarize the Qwen fixed-cutoff grid from native run receipts."""

import argparse
import json
from pathlib import Path
import random
import re

from fixed_grid import ARMS, orders


LENGTHS = (256, 1024, 4096, 16384)


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (position - lo)


def summarize(pairs, bootstrap=True):
    if not pairs:
        return {"pairs": 0}
    baseline = [group["off"] for group in pairs]
    base_tokens = sum(row["output_tokens"] for row in baseline)
    base_seconds = sum(row["elapsed_s"] for row in baseline)
    result = {"pairs": len(pairs), "arms": {}}
    for label in ARMS:
        rows = [group[label] for group in pairs]
        tokens = sum(row["output_tokens"] for row in rows)
        seconds = sum(row["elapsed_s"] for row in rows)
        paired = [
            100 * ((row["output_tokens"] / row["elapsed_s"])
                   / (base["output_tokens"] / base["elapsed_s"]) - 1)
            for row, base in zip(rows, baseline)
        ]
        arm = {
            "output_tokens": tokens,
            "complete_request_seconds": seconds,
            "tokens_per_second": tokens / seconds,
            "pooled_gain_percent": 100 * ((tokens / seconds) / (base_tokens / base_seconds) - 1),
            "paired_gain_percent": paired,
            "paired_wins": sum(value > 0 for value in paired),
            "output_token_ratio": tokens / base_tokens,
            "latency_ratio": seconds / base_seconds,
            "format_covered": sum(row["format"]["requested_format_covered"] for row in rows),
            "rounds": sum(row["rounds"] for row in rows),
            "confidence_shortened_rounds": sum(
                row["confidence_shortened_rounds"] for row in rows
            ),
        }
        if bootstrap and label != "off" and len(pairs) >= 3:
            rng = random.Random(20730923)
            draws = []
            for _ in range(5000):
                sampled = [rng.randrange(len(pairs)) for _ in pairs]
                rate = (sum(rows[i]["output_tokens"] for i in sampled)
                        / sum(rows[i]["elapsed_s"] for i in sampled))
                base_rate = (sum(baseline[i]["output_tokens"] for i in sampled)
                             / sum(baseline[i]["elapsed_s"] for i in sampled))
                draws.append(100 * (rate / base_rate - 1))
            arm["paired_bootstrap_95_percent"] = [
                quantile(draws, 0.025), quantile(draws, 0.975)
            ]
        else:
            arm["paired_bootstrap_95_percent"] = None
        result["arms"][label] = arm
    return result


def report(root):
    freeze = json.loads((root / "FREEZE.json").read_text())
    state = json.loads((root / "status.json").read_text())
    if state["status"] != "completed" or freeze["orders"] != orders():
        raise ValueError("fixed confidence grid is incomplete or its order changed")
    if freeze["arms"] != {
        name: {"pmin": pmin, "pmin0": pmin0} for name, (pmin, pmin0) in ARMS.items()
    }:
        raise ValueError("confidence arms changed")
    expected = {
        f"qwen/scored/{scenario:02}-{label}"
        for scenario in range(6) for label in ARMS
    }
    if not expected <= set(state["completed"]) or len(state["completed"]) != len(set(state["completed"])):
        raise ValueError("native run inventory is missing or duplicated")
    cells = {length: [] for length in LENGTHS}
    exclusions = []
    histograms = {}
    for scenario in range(6):
        runs = {}
        for label, (pmin, pmin0) in ARMS.items():
            path = root / "qwen/scored"
            stem = f"{scenario:02}-{label}"
            record = json.loads((path / f"{stem}.audit.json").read_text())
            command = json.loads((path / f"{stem}.command.json").read_text())
            exit_row = json.loads((path / f"{stem}.exit.json").read_text())
            if (record["arm"] != "fixed:3" or record["gate"] or record["max_new"] != 8192
                    or command["pmin"] != pmin or command["pmin0"] != pmin0
                    or command["native_adapt"] or not command["spec_stats"]
                    or exit_row["returncode"] != 0 or exit_row["contamination"]):
                raise ValueError(f"wrong or failed native arm: {stem}")
            text = (path / f"{stem}.log").read_text()
            lengths = re.findall(r"\[spec-stats\] rounds=(\d+) full_accept=\d+ len_hist=\[([0-9, ]+)\]", text)
            if len(lengths) < 8:
                raise ValueError(f"missing draft-length histograms: {stem}")
            histogram = [0] * 8
            by_turn = []
            # The native driver warms several discarded sessions before the eight
            # scored requests. Their spec-stats lines are not part of this clock.
            for rounds, values in lengths[-8:]:
                row = [int(value.strip()) for value in values.split(",")]
                if len(row) != 8 or sum(row) != int(rounds):
                    raise ValueError(f"invalid draft-length histogram: {stem}")
                by_turn.append(row)
                histogram = [left + right for left, right in zip(histogram, row)]
            cuts = sum(histogram[:3])
            if label == "off" and cuts:
                raise ValueError("confidence-off arm drafted fewer than fixed K=3")
            histograms[stem] = {
                "draft_lengths": histogram,
                "by_turn": by_turn,
                "confidence_shortened_rounds": cuts,
            }
            runs[label] = (record, path / stem)
        for turn in (2, 4, 6, 8):
            rows = {}
            for label, (record, _) in runs.items():
                histogram = histograms[f"{scenario:02}-{label}"]["by_turn"][turn - 1]
                rows[label] = {
                    **record["requests"][turn - 1],
                    "rounds": sum(histogram),
                    "confidence_shortened_rounds": sum(histogram[:3]),
                }
            if any(row["kind"] != "code" or row["k"] != 3 for row in rows.values()):
                raise ValueError("a requested code turn did not use fixed K=3")
            length = rows["off"]["length_target"]
            if length not in cells or any(row["length_target"] != length for row in rows.values()):
                raise ValueError("paired code lengths differ")
            prompts = {
                (path / f"turn-{turn}.prompt.ids").read_bytes()
                for _, path in runs.values()
            }
            if len(prompts) != 1:
                raise ValueError("paired code prompt tokens differ")
            if any(row["loop"] for row in rows.values()):
                exclusions.append({"scenario": scenario, "length": length, "turn": turn})
            else:
                cells[length].append(rows)
    return {
        "status": "measured-fixed-cutoff-only",
        "scope": "Qwen3.8-27B native independent requests on one research RTX 5090; no adaptive C, HTTP or concurrency claim",
        "freeze": freeze,
        "draft_length_histograms": histograms,
        "matched_loop_exclusions": exclusions,
        "all_code": summarize(
            [pair for length in LENGTHS for pair in cells[length]], bootstrap=False
        ),
        "code_by_prompt_tokens": {str(length): summarize(cells[length]) for length in LENGTHS},
        "format_covered_code_by_prompt_tokens": {
            str(length): summarize([
                pair for pair in cells[length]
                if all(row["format"]["requested_format_covered"] for row in pair.values())
            ]) for length in LENGTHS
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.root)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "code_pairs": result["all_code"]["pairs"],
        "code_gain_percent": {
            label: round(row["pooled_gain_percent"], 3)
            for label, row in result["all_code"]["arms"].items()
        },
    }))


if __name__ == "__main__":
    main()
