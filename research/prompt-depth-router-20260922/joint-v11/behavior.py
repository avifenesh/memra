"""Describe v10 token-history behavior without changing its frozen score."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


from measurement_rows import V10Archive, old_rows


DOMAINS = ("ifeval", "gsm8k")
POLICIES = (
    "fixed-k20-d3-c0", "fixed-k20-c-low", "fixed-k20-c-mid",
    "cd-new-only", "joint-augmented",
)
V10_WORKLOAD_SHA = "dd9fc45646931f66fee8a3b328b404d7da227765c58ce78f57536b3605fcbeb8"
CLASSES = ("word", "number", "space", "line", "symbol", "other")


def custody_check(archive, custody):
    receipt = json.loads(custody.read_text())
    if (
        receipt["status"] != "verified-training-projection"
        or receipt["archive_sha256"] != archive.manifest["archive_sha256"]
        or receipt["parent_archive_sha256"]
        != archive.manifest["parent_archive_sha256"]
        or receipt["model_sha256"] != archive.manifest["model_sha256"]
        or receipt["parent_replay"]["status"]
        != "archive-native-IFEval-GSM8K-noops-and-E2E-match"
        or receipt["parent_replay"]["heldout_arms"] != 320
    ):
        raise ValueError("posthoc source lacks matching replay custody")
    if archive.manifest["workloads_sha256"] != V10_WORKLOAD_SHA:
        raise ValueError("posthoc non-code split differs")


def token_classes(archive, name):
    mapping = {}
    for row in archive.table(f"native/{name}/token-bytes.tsv"):
        token = int(row["id"])
        mapping[token] = old_rows.classify(bytes.fromhex(row["hex"]))
    return mapping


def fresh():
    return {
        "rounds": 0, "accepted": 0, "drafted": 0, "round_seconds": 0.0,
    }


def add_round(group, row):
    depth = int(row["draft_depth"])
    accepted = int(row["emitted"]) - 1
    if not 1 <= depth <= 4 or not 0 <= accepted <= depth:
        raise ValueError("posthoc D receipt differs")
    group["rounds"] += 1
    group["accepted"] += accepted
    group["drafted"] += depth
    group["round_seconds"] += int(row["elapsed_ns"]) / 1e9


def summarize_rounds(groups):
    return {
        key: {
            **value,
            "accepted_per_drafted": (
                value["accepted"] / value["drafted"]
                if value["drafted"] else None
            ),
        }
        for key, value in sorted(groups.items())
    }


def analyze(archive, custody):
    custody_check(archive, custody)
    quality = json.loads(archive.receipt("native/heldout-quality.json"))
    arms = json.loads(archive.receipt("native/arms.json"))
    if (
        quality["phase"] != "heldout"
        or not set(POLICIES).issubset(
            {item["label"] for item in arms["arms"]}
        )
    ):
        raise ValueError("posthoc arm or task-quality source differs")
    report = {
        "schema": 1,
        "scope": "posthoc observed C/K/D behavior; native tok/s remains primary",
        "archive_sha256": archive.manifest["archive_sha256"],
        "domains": {},
    }
    for domain in DOMAINS:
        graded = {
            row["session"]: row
            for row in quality["domains"][domain]["sessions"]
        }
        report["domains"][domain] = {}
        for arm in POLICIES:
            turns = defaultdict(lambda: {
                "turns": 0, "tokens": 0, "seconds": 0.0,
                "task_pass": 0, "c_decisions": 0, "c_stops": 0,
                "capped": 0,
            })
            depths = defaultdict(fresh)
            history = defaultdict(fresh)
            looped = []
            for index in range(16):
                name = f"heldout-{domain}-{index}-{arm}"
                result = archive.json(f"native/{name}.result.json")
                if result["name"] != name or result["loops"]:
                    if result["loops"]:
                        looped.append(index)
                        continue
                    raise ValueError("posthoc native session differs")
                grade = graded[name]["turns"]
                native_turns = archive.table(f"native/{name}/turns.tsv")
                if len(grade) != 8 or len(native_turns) != 8:
                    raise ValueError("posthoc native or quality turn count differs")
                outputs = {
                    turn: archive.ids(
                        f"native/{name}/turn-{turn}.output.ids"
                    )
                    for turn in range(1, 9)
                }
                classes = token_classes(archive, name)
                for number, (native, task) in enumerate(
                    zip(native_turns, grade), 1
                ):
                    if int(native["turn"]) != number or task["turn"] != number:
                        raise ValueError("posthoc turn alignment differs")
                    bucket = "first" if number == 1 else "later"
                    item = turns[bucket]
                    item["turns"] += 1
                    item["tokens"] += int(native["output_tokens"])
                    item["seconds"] += float(native["elapsed_s"])
                    item["task_pass"] += task["pass"]
                    item["c_decisions"] += int(native["confidence_decisions"])
                    item["c_stops"] += int(native["confidence_stops"])
                    item["capped"] += native["finish_reason"] == "length"
                spans = {
                    (int(row["turn"]), int(row["round"])): row
                    for row in archive.table(f"native/{name}/spans.tsv")
                }
                for row in archive.table(f"native/{name}/rounds.tsv"):
                    if row["eligible_for_learning"] != "true":
                        continue
                    turn = int(row["turn"])
                    key = turn, int(row["round"])
                    span = spans[key]
                    offset = int(span["output_start"])
                    ids = outputs[turn]
                    if not 0 <= offset <= len(ids):
                        raise ValueError("posthoc committed history differs")
                    previous = (
                        "start" if offset == 0
                        else CLASSES[classes.get(ids[offset - 1], 5)]
                    )
                    depth = int(row["draft_depth"])
                    add_round(depths[str(depth)], row)
                    add_round(history[f"{previous}:D{depth}"], row)
            turn_summary = {
                bucket: {
                    **item,
                    "tok_s": item["tokens"] / item["seconds"]
                    if item["seconds"] else None,
                }
                for bucket, item in sorted(turns.items())
            }
            report["domains"][domain][arm] = {
                "turns": turn_summary,
                "d_observed": summarize_rounds(depths),
                "previous_committed_class": summarize_rounds(history),
                "looped_conversations_excluded": looped,
            }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--custody", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    archive = V10Archive(args.archive, args.manifest)
    try:
        result = analyze(archive, args.custody)
    finally:
        archive.close()
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({
        domain: {
            arm: {
                "first_tok_s": row["turns"].get("first", {}).get("tok_s"),
                "later_tok_s": row["turns"].get("later", {}).get("tok_s"),
            }
            for arm, row in policies.items()
        }
        for domain, policies in result["domains"].items()
    }, sort_keys=True))


if __name__ == "__main__":
    main()
