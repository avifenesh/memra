"""Convert native v9 randomized training receipts into K/D/C observations."""

import argparse
import csv
import gzip
import io
import json
from pathlib import Path

from measurement_rows import k_rows, round_rows


class Directory:
    def __init__(self, root):
        self.root = root

    def read(self, name):
        return (self.root / name.removeprefix("native/")).read_bytes()

    def table(self, name):
        return list(csv.DictReader(io.StringIO(self.read(name).decode()), delimiter="\t"))

    def ids(self, name):
        return [int(value) for value in self.read(name).split()]


def build(root, out):
    records = [json.loads(path.read_text()) for path in sorted(root.glob("training-*.result.json"))]
    if len(records) != 144:
        raise ValueError("24 training conversations need six complete arms each")
    by_arm = {}
    for record in records:
        index = int(record["name"].split("-")[1])
        key = index, record["k"], record["variant"].split("-", 1)[1]
        if key in by_arm:
            raise ValueError(f"duplicate native training arm: {key}")
        by_arm[key] = record
    if set(by_arm) != {
        (index, k, label)
        for index in range(24) for k in (3, 10, 20)
        for label in ("fixed-d3", "explore-d")
    }:
        raise ValueError("native training arm inventory differs")
    excluded = sorted({
        int(record["name"].split("-")[1])
        for record in records if record["loops"]
    })
    included = [
        index for index in range(24) if index not in excluded
    ]
    if len(included) < 20:
        raise ValueError("too few unlooped training task groups")
    wrapper = Directory(root)
    fixed = [
        {
            **row, "loops": 0,
        } for row in records
        if row["variant"].endswith("fixed-d3")
        and int(row["name"].split("-")[1]) in included
    ]
    randomized = [
        {
            **row, "loops": 0,
        } for row in records
        if row["variant"].endswith("explore-d")
        and int(row["name"].split("-")[1]) in included
    ]
    out.mkdir(parents=True, exist_ok=False)
    counts = {"k": {}, "d": {}, "c": {}}
    with (
        gzip.open(out / "k.jsonl.gz", "xt") as k_stream,
        gzip.open(out / "d.jsonl.gz", "xt") as d_stream,
        gzip.open(out / "c.jsonl.gz", "xt") as c_stream,
    ):
        for row in k_rows(wrapper, {"records": fixed}, "v9", {
            "k3-fixed-d3", "k10-fixed-d3", "k20-fixed-d3",
        }):
            k_stream.write(json.dumps(row, sort_keys=True) + "\n")
            key = f"K{row['draft_k']}"
            counts["k"][key] = counts["k"].get(key, 0) + 1
        for kind, row in round_rows(
            wrapper, {"records": randomized}, "v9",
            lambda record: record["variant"].endswith("explore-d"),
        ):
            stream = c_stream if kind == "c" else d_stream
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            key = f"K{row['draft_k']}:D{row['draft_depth']}"
            counts[kind][key] = counts[kind].get(key, 0) + 1
    result = {
        "schema": 1,
        "phase": "training",
        "conversation_count": len(included),
        "included_training_topics": included,
        "excluded_looped_training_topics": excluded,
        "counts": counts,
    }
    (out / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root, args.out)
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
