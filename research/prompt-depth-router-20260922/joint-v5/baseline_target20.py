"""Finish target-20/draft-20 native controls after draft-only K clarification."""

import argparse
import json
from pathlib import Path

from run_native import input_manifests, run_one, save


def main():
    parser = argparse.ArgumentParser()
    for name in ("binary", "model", "development", "fresh", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    for name in ("binary", "model", "development", "fresh", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(exist_ok=True)
    old, _ = input_manifests(args)
    topics = old["groups"]["calibration"] + old["groups"]["heldout"][:3]
    records = [
        run_one(
            args, topics[index], args.development,
            f"development-{index}", "topk20-d3-c0", 20, "fixed:3", 3,
        )
        for index in range(6)
    ]
    save(args.out / "target20-draft20-baseline.json", {
        "schema": 1,
        "scope": "target and MTP draft top-k both 20 at temperature 1.0",
        "records": records,
    })
    print(json.dumps({"complete_conversations": len(records)}))


if __name__ == "__main__":
    main()
