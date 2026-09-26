"""Freeze and run a measured K=20 fixed-C control during draft-prime repair."""

import argparse
import json
from pathlib import Path
import struct

from run_native import input_manifests, run_one, save, v4


def q25(root):
    values = {0: [], 1: []}
    for index in range(6):
        session = root / f"prepatch-{index}-topk20-random-d"
        for turn in range(1, 9):
            for row in v4.table(session / f"turn-{turn}.confidence.tsv"):
                if row["eligible"] != "true":
                    continue
                bits = row["q_bits"].split(",")
                for position in values:
                    if position < len(bits):
                        value = struct.unpack(
                            "<f", struct.pack("<I", int(bits[position]))
                        )[0]
                        if not 0 <= value <= 1:
                            raise ValueError("invalid sampled C quantile")
                        values[position].append(value)
    if any(len(rows) < 150 for rows in values.values()):
        raise ValueError("K=20 fixed C lacks randomized exposure")
    return [
        sorted(values[position])[int(0.25 * (len(values[position]) - 1))]
        for position in (0, 1)
    ]


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
    pair = q25(args.out)
    save(args.out / "prepatch-c20-candidate.json", {
        "schema": 1,
        "scope": "target and draft top-k=20, empirical lower-quartile q",
        "cutoffs": pair,
    })
    records = [
        run_one(
            args, old["groups"]["calibration"][index], args.development,
            f"prepatch-c-{index}", "topk20-d3-c-quarter", 20,
            "fixed-c3", 3,
            extra=(f"confidence-fixed={pair[0]:.9g},{pair[1]:.9g}",),
        )
        for index in range(3)
    ]
    save(args.out / "prepatch-c20-summary.json", {
        "schema": 1, "records": records,
        "candidate": pair,
    })
    print(json.dumps({"complete_sessions": len(records), "cutoffs": pair}))


if __name__ == "__main__":
    main()
