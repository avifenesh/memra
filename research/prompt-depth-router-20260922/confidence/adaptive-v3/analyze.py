"""Replay heldout Qwen native-session receipts into paired, complete-clock rates."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

from learn import read_rounds


ARMS = ("learn-c3", "monitor-c3", "fixed-c3", "fixed:3", "fixed:2")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_tsv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def token_ids(path):
    return [int(value) for value in Path(path).read_text().split()]


def prefix_receipts(path, turns):
    records = []
    for turn in range(2, 9):
        cached = int(turns[turn - 1]["cached_tokens"])
        prior = (
            token_ids(path / f"turn-{turn - 1}.prompt.ids")
            + token_ids(path / f"turn-{turn - 1}.committed.ids")
        )
        current = token_ids(path / f"turn-{turn}.prompt.ids")
        if cached <= 0 or prior[:cached] != current[:cached]:
            raise ValueError(f"native cached prefix differs on turn {turn}")
        digest = hashlib.sha256(
            "\n".join(map(str, current[:cached])).encode()
        ).hexdigest()
        records.append({
            "turn": turn, "cached_tokens": cached,
            "prefix_sha256": digest,
        })
    return records


def quantiles(values):
    ordered = sorted(values)
    return [ordered[int((len(ordered) - 1) * portion)] for portion in (0.025, 0.975)]


def combined(records):
    tokens = sum(row["output_tokens"] for session in records for row in session)
    seconds = sum(row["elapsed_s"] for session in records for row in session)
    drafted = sum(row["drafted"] for session in records for row in session)
    accepted = sum(row["accepted"] for session in records for row in session)
    return {
        "output_tokens": tokens,
        "complete_request_s": seconds,
        "tokens_per_s": tokens / seconds,
        "controller_wall_s": sum(
            row["confidence_policy_ns"] for session in records for row in session
        ) / 1e9,
        "drafted": drafted,
        "accepted": accepted,
        "accepted_per_drafted": accepted / drafted,
        "rounds": sum(row["policy_rounds"] for session in records for row in session),
        "offered_drafts_per_round": drafted / sum(
            row["policy_rounds"] for session in records for row in session
        ),
        "policy_ceiling_histogram": {
            str(depth): sum(
                row["policy_ceiling_histogram"][str(depth)]
                for session in records for row in session
            )
            for depth in (1, 2, 3)
        },
        "offered_draft_histogram": (
            {
                str(depth): sum(
                    row["offered_draft_histogram"][str(depth)]
                    for session in records for row in session
                )
                for depth in (1, 2, 3)
            }
            if all(
                row["offered_draft_histogram"] is not None
                for session in records for row in session
            )
            else None
        ),
        "format_covered": sum(
            row["fenced_parseable_function"] for session in records for row in session
        ),
        "turns": sum(len(session) for session in records),
    }


def replay(root, manifest):
    groups = manifest["groups"]["heldout"]
    sessions, sources, exclusions, prefixes = [], [], [], []
    for index, entry in enumerate(groups):
        arms = {}
        exclude = False
        for arm in ARMS:
            path = root / f"heldout-{index}-{arm}"
            audit = json.loads((path / "audit.json").read_text())
            turns = load_tsv(path / "turns.tsv")
            round_rows = load_tsv(path / "rounds.tsv")
            command = json.loads((path / "command.json").read_text())
            if (
                len(audit) != 8 or len(turns) != 8
                or command["workload_sha256"] != entry["sha256"]
                or command["model_sha256"] !=
                    "1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a"
                or not round_rows
            ):
                raise ValueError("heldout native source/model/workload mismatch")
            hist_by_turn = {
                turn: {str(depth): 0 for depth in (1, 2, 3)}
                for turn in range(1, 9)
            }
            for native_round in round_rows:
                turn, depth = int(native_round["turn"]), int(native_round["draft_depth"])
                if turn not in hist_by_turn or depth not in (1, 2, 3):
                    raise ValueError("heldout round has another turn or draft depth")
                hist_by_turn[turn][str(depth)] += 1
            for row, turn, request in zip(audit, turns, entry["turns"]):
                if row["output_tokens"] != int(turn["output_tokens"]):
                    raise ValueError("audit and native turn disagree")
                row["prompt_tokens"] = int(turn["prompt_tokens"])
                row["reference_padding_chars"] = request["padding_chars"]
                row["confidence_before"] = turn["confidence_before"]
                row["confidence_after"] = turn["confidence_after"]
                row["confidence_policy_ns"] = int(turn["confidence_policy_ns"])
                row["drafted"] = int(turn["drafted"])
                row["accepted"] = int(turn["accepted"])
                row["policy_rounds"] = int(turn["policy_rounds"])
                row["policy_ceiling_histogram"] = hist_by_turn[row["turn"]]
                if sum(hist_by_turn[row["turn"]].values()) != row["policy_rounds"]:
                    raise ValueError("heldout round trace count disagrees with native turn")
                if arm in ("learn-c3", "monitor-c3"):
                    offers = read_rounds(path / f"turn-{row['turn']}.confidence.tsv")
                    if (
                        len(offers) != row["policy_rounds"]
                        or sum(offer["drafted"] for offer in offers) != row["drafted"]
                    ):
                        raise ValueError("sampled offered draft trace disagrees with native turn")
                    row["offered_draft_histogram"] = {
                        str(depth): sum(offer["drafted"] == depth for offer in offers)
                        for depth in (1, 2, 3)
                    }
                    decision = json.loads(
                        (path / f"turn-{row['turn']}.c-decision.json").read_text()
                    )
                    if decision["turn"] != row["turn"]:
                        raise ValueError("live confidence decision has another turn")
                    row["confidence_decision"] = {
                        key: decision[key] for key in (
                            "cutoffs_before", "cutoffs_after", "probe",
                            "eligible_rounds_added", "policy_cpu_ns",
                        )
                    }
                    row["confidence_decision"]["candidate_cutoffs"] = decision.get(
                        "candidate_cutoffs"
                    )
                    if row["turn"] == 1 and row["confidence_before"] != "0.000000000,0.000000000":
                        raise ValueError("live confidence state did not reset for the new conversation")
                    if arm == "monitor-c3" and row["confidence_after"] != "0.000000000,0.000000000":
                        raise ValueError("monitor control applied a learned cutoff")
                else:
                    row["offered_draft_histogram"] = None
            prefixes.append({
                "session": index, "arm": arm,
                "turns": prefix_receipts(path, turns),
            })
            exclude |= any(row["loop"] for row in audit)
            arms[arm] = audit
            sources.append({
                "session": index,
                "arm": arm,
                "turns_sha256": sha(path / "turns.tsv"),
                "audit_sha256": sha(path / "audit.json"),
                "binary_sha256": command["binary_sha256"],
            })
        if exclude:
            exclusions.append({
                "session": index, "reason": "exact output loop in a matched arm"
            })
        else:
            sessions.append({"index": index, "arms": arms})
    if not sessions:
        raise ValueError("all matched heldout conversations looped")
    if len({item["binary_sha256"] for item in sources}) != 1:
        raise ValueError("heldout arms used different native binaries")
    return sessions, sources, exclusions, prefixes


def report(root, manifest):
    sessions, sources, exclusions, prefixes = replay(root, manifest)
    rows = {arm: [session["arms"][arm] for session in sessions] for arm in ARMS}
    metrics = {arm: combined(values) for arm, values in rows.items()}
    control = metrics["fixed:3"]
    for arm in ARMS:
        metrics[arm]["gain_vs_k3c0_percent"] = (
            100 * (metrics[arm]["tokens_per_s"] / control["tokens_per_s"] - 1)
        )
        metrics[arm]["output_token_ratio_vs_k3c0"] = (
            metrics[arm]["output_tokens"] / control["output_tokens"]
        )
        metrics[arm]["complete_time_ratio_vs_k3c0"] = (
            metrics[arm]["complete_request_s"] / control["complete_request_s"]
        )
    rng = random.Random(20773000)
    bands = {}
    for arm in ARMS:
        gains = []
        for _ in range(4096):
            draw = [sessions[rng.randrange(len(sessions))] for _ in sessions]
            picked = combined([item["arms"][arm] for item in draw])
            base = combined([item["arms"]["fixed:3"] for item in draw])
            gains.append(100 * (picked["tokens_per_s"] / base["tokens_per_s"] - 1))
        bands[arm] = quantiles(gains)
    learned_pair_bands = {}
    for comparator in ("fixed-c3", "monitor-c3", "fixed:2"):
        gains = []
        for _ in range(4096):
            draw = [sessions[rng.randrange(len(sessions))] for _ in sessions]
            learned = combined([item["arms"]["learn-c3"] for item in draw])
            other = combined([item["arms"][comparator] for item in draw])
            gains.append(100 * (learned["tokens_per_s"] / other["tokens_per_s"] - 1))
        learned_pair_bands[comparator] = quantiles(gains)
    lengths = {}
    for arm in ARMS:
        lengths[arm] = {}
        for padding in (256, 1024, 4096, 16384):
            picked = [
                [row for row in session if row["reference_padding_chars"] == padding]
                for session in rows[arm]
            ]
            picked = [session for session in picked if session]
            if picked:
                lengths[arm][str(padding)] = {
                    **combined(picked),
                    "realized_prompt_tokens": [
                        row["prompt_tokens"] for session in picked for row in session
                    ],
                }
    trajectories, monitor_trajectories = [], []
    for session in sessions:
        for arm, destination in (
            ("learn-c3", trajectories), ("monitor-c3", monitor_trajectories)
        ):
            for turn in session["arms"][arm]:
                destination.append({
                    "session": session["index"],
                    "turn": turn["turn"],
                    "applied_before": turn["confidence_before"],
                    "applied_after": turn["confidence_after"],
                    "decision": turn["confidence_decision"],
                    "output_tokens": turn["output_tokens"],
                    "complete_request_s": turn["elapsed_s"],
                })
    return {
        "schema": 1,
        "scope": "Qwen code, sampled native eight-turn continuing sessions, complete request clock",
        "included_conversations": [session["index"] for session in sessions],
        "excluded_conversations": exclusions,
        "metrics": metrics,
        "conversation_bootstrap_95_percent_gain_vs_k3c0": bands,
        "learned_vs_comparator_bootstrap_95_percent": learned_pair_bands,
        "requested_reference_padding_cells": lengths,
        "learned_confidence_trajectories": trajectories,
        "monitor_confidence_trajectories": monitor_trajectories,
        "functional_execution_coverage": "not measured by the fenced syntax gate",
        "sources": sources,
        "native_cached_prefixes": prefixes,
        "costs_sha256": sha(root / "costs.json"),
        "oracle_sha256": sha(root / "oracle.json"),
        "workloads_sha256": sha(Path(__file__).with_name("workloads-v3") / "manifest.json"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = report(
        args.root,
        json.loads((args.workloads / "manifest.json").read_text()),
    )
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({
        "metrics": result["metrics"],
        "bootstrap": result["conversation_bootstrap_95_percent_gain_vs_k3c0"],
        "excluded": result["excluded_conversations"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
