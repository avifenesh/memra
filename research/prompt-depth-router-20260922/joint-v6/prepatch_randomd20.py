"""Collect K=20 C/D controls while draft-only prime setup is repaired."""

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
            args, entry, args.development, f"prepatch-{index}",
            "topk20-random-d", 20, "explore-d", 4,
            extra=(f"explore-seed={20786000 + index}",),
        )
        for index, entry in enumerate(topics)
    ]
    save(args.out / "prepatch-k20-randomd.json", {
        "schema": 1,
        "scope": "target and draft top-k=20 randomized D on first v6 source",
        "records": records,
    })
    print(json.dumps({"complete_sessions": len(records)}))


if __name__ == "__main__":
    main()
