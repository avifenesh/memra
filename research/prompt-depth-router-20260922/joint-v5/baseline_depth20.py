"""Measure fixed D controls with target and MTP draft both at top-k=20."""

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
    records = []
    for index, entry in enumerate(topics):
        order = (2, 4) if index % 2 == 0 else (4, 2)
        for d in order:
            records.append(run_one(
                args, entry, args.development, f"depth20-{index}",
                f"target20-draft20-d{d}-c0", 20, f"fixed:{d}", d,
            ))
    save(args.out / "target20-draft20-depth-grid.json", {
        "schema": 1,
        "scope": "temperature 1.0 fixed D2/D4 with target and draft top-k=20",
        "records": records,
    })
    print(json.dumps({"complete_sessions": len(records)}))


if __name__ == "__main__":
    main()
