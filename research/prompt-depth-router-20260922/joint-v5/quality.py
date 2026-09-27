"""Probe fresh generated functions after all native request clocks have stopped."""

import argparse
import json
from pathlib import Path

from run_native import quality, save


def score(root, manifest):
    summary = json.loads((root / "heldout-summary.json").read_text())
    if summary["schema"] != 1 or summary["phase"] != "heldout":
        raise ValueError("fresh native heldout summary differs")
    records = []
    for row in summary["records"]:
        name = row["name"]
        index = int(name.split("-")[1])
        expected = manifest["groups"]["heldout"][index]["turns"]
        tested = [
            {
                "turn": turn,
                "function": item["function"],
                **quality.probe(
                    (root / name / f"turn-{turn}.answer.txt").read_text(),
                    item["function"],
                ),
            }
            for turn, item in enumerate(expected, 1)
        ]
        records.append({
            "session": name, "variant": row["variant"],
            "passed": sum(result["pass"] for result in tested),
            "turns": tested,
        })
    return {
        "schema": 1,
        "scope": "one bounded functional case per fresh Qwen code turn",
        "sessions": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    report = score(args.root, manifest)
    save(args.root / "quality.json", report)
    print(json.dumps({
        variant: sum(row["passed"] for row in report["sessions"]
                     if row["variant"] == variant)
        for variant in sorted({row["variant"] for row in report["sessions"]})
    }))


if __name__ == "__main__":
    main()
