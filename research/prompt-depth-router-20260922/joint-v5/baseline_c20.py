"""Keep a measured fixed-C reference at target/draft top-k=20."""

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
            args, entry, args.development, f"fixedc20-{index}",
            "target20-draft20-calibrated-c", 20, "fixed-c3", 3,
            extra=("confidence-fixed=0.434978,0.850344",),
        )
        for index, entry in enumerate(topics)
    ]
    save(args.out / "target20-draft20-calibrated-c.json", {
        "schema": 1,
        "scope": "temperature 1.0 fixed C with target and draft top-k=20",
        "cutoffs": [0.434978, 0.850344],
        "records": records,
    })
    print(json.dumps({"complete_sessions": len(records)}))


if __name__ == "__main__":
    main()
