"""Check sampled no-op identity and live C engagement before validation."""

import argparse
import json
from pathlib import Path


REFERENCE = "fixed-k20-d3-c0"


def qualify(root, arms_path):
    arms = json.loads(arms_path.read_text())
    if arms["phase"] != "qualification":
        raise ValueError("qualification arm manifest differs")
    labels = {item["label"] for item in arms["arms"]}
    records = {}
    for path in root.glob("qualification-0-*.result.json"):
        result = json.loads(path.read_text())
        if result["variant"] in records:
            raise ValueError("duplicate qualification arm")
        records[result["variant"]] = result
    if set(records) != labels or REFERENCE not in records:
        raise ValueError("qualifier lacks a candidate arm")
    baseline = root / f"qualification-0-{REFERENCE}"
    noops = sorted(
        label for label in labels
        if label.startswith(("k-noop-", "cd-noop-", "joint-noop-"))
    )
    for label in noops:
        session = root / f"qualification-0-{label}"
        for turn in range(1, 9):
            if (
                session / f"turn-{turn}.output.ids"
            ).read_bytes() != (
                baseline / f"turn-{turn}.output.ids"
            ).read_bytes():
                raise ValueError(f"sampled no-op output differs: {label}/{turn}")
    active = {}
    for label in sorted(labels):
        if (
            not label.startswith(("cd-", "joint-"))
            or label.startswith(("cd-noop-", "joint-noop-"))
        ):
            continue
        row = records[label]
        active[label] = {
            "decisions": row["c_decisions"],
            "stops": row["c_stops"],
            "engaged": (
                0 < row["c_stops"] < row["c_decisions"]
            ),
        }
    if not active or not any(value["engaged"] for value in active.values()):
        raise ValueError("none of the C policies made mixed native stop decisions")
    if any(row["cached_later_turns"] != 7 for row in records.values()):
        raise ValueError("qualification arm lacks native KV reuse")
    return {
        "schema": 1,
        "status": "qualified",
        "byte_identical_noops": noops,
        "c_policy_engagement": active,
        "arms": len(records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = qualify(args.root, args.arms)
    with args.out.open("x") as target:
        json.dump(result, target, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
