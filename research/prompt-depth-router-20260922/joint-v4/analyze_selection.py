"""Audit model choices against complete held-back development conversations."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def sessions(path, phase):
    report = json.loads(path.read_text())
    if report["schema"] != 1 or report["phase"] != phase:
        raise ValueError("selection summary has another phase or schema")
    groups = defaultdict(list)
    for record in report["records"]:
        groups[record["variant"]].append(record)
    if any(len(group) != 3 for group in groups.values()):
        raise ValueError("selection arm lacks three matched conversations")
    return groups


def aggregate(groups):
    result = {}
    for name, rows in groups.items():
        tokens = sum(row["tokens"] for row in rows)
        seconds = sum(row["seconds"] for row in rows)
        result[name] = {
            "tokens": tokens,
            "seconds": seconds,
            "tok_s": tokens / seconds,
            "format_pass": sum(row["format"] for row in rows),
            "loops": sum(row["loops"] for row in rows),
        }
    return result


def score(root):
    d = json.loads((root / "selected-model.json").read_text())
    c = json.loads((root / "selected-confidence.json").read_text())
    if d["schema"] != 1 or c["schema"] != 1:
        raise ValueError("selected model schema differs")
    depth = aggregate(sessions(root / "selection-summary.json", "selection"))
    confidence = aggregate(sessions(
        root / "confidence-selection-summary.json", "confidence-selection",
    ))
    if (
        d["fixed_control"] != c["fixed_control"]
        or d["variant"] + "-trained" not in depth
        or "cd-" + c["variant"] not in confidence
        or "cd-noop" not in confidence
        or "c-noop" not in confidence
    ):
        raise ValueError("frozen model decisions differ from measured arms")
    for name, rows in (("depth", depth), ("confidence", confidence)):
        if "k3-c0" not in rows or any(
            row["format_pass"] not in range(25) for row in rows.values()
        ):
            raise ValueError(name + " selection has invalid code-format count")
    return {
        "schema": 1,
        "scope": "old v3 development topics, never a fresh heldout verdict",
        "selected_depth": d["variant"],
        "selected_depth_model_sha256": d["model_sha256"],
        "selected_confidence": c["variant"],
        "selected_confidence_model_sha256": c["model_sha256"],
        "fixed_control": d["fixed_control"],
        "depth_selection": depth,
        "confidence_selection": confidence,
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
        "depth": result["selected_depth"],
        "confidence": result["selected_confidence"],
    }))


if __name__ == "__main__":
    main()
