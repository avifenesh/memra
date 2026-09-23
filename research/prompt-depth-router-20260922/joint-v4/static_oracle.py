"""Development-only hindsight over executed, whole native conversations."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile


ARMS = ("fixed:3", "fixed:2", "fixed-c3")
ARCHIVE_SHA = "8631202585850659769481529485f89a53f78381a06fcce67c2d7bdd1f10fb9c"
WORKLOADS_SHA = "00a1b6d97fbae3ecf69b46de7097cf6dd0639ca7aed60ed84585d17a2a8bfdef"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def total(rows):
    tokens = sum(int(row["output_tokens"]) for row in rows)
    seconds = sum(float(row["elapsed_s"]) for row in rows)
    if len(rows) != 8 or tokens <= 0 or seconds <= 0:
        raise ValueError("a fixed control lacks eight complete native turns")
    if any(int(row["turn"]) != index for index, row in enumerate(rows, 1)):
        raise ValueError("fixed control turn order differs")
    return {"tokens": tokens, "seconds": seconds, "tokens_per_s": tokens / seconds}


def report(archive_path, workload_path):
    if sha(archive_path) != ARCHIVE_SHA or sha(workload_path) != WORKLOADS_SHA:
        raise ValueError("static oracle input differs from its pinned development corpus")
    manifest = json.loads(Path(workload_path).read_text())
    conversations = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for index, item in enumerate(manifest["groups"]["heldout"]):
            controls = {}
            for arm in ARMS:
                name = f"native/heldout-{index}-{arm}/turns.tsv"
                member = archive.getmember(name)
                rows = list(csv.DictReader(
                    io.StringIO(archive.extractfile(member).read().decode()),
                    delimiter="\t",
                ))
                controls[arm] = total(rows)
            chosen = max(ARMS, key=lambda arm: (
                controls[arm]["tokens_per_s"], -ARMS.index(arm)
            ))
            conversations.append({
                "index": index, "topic": item["topic"],
                "controls": controls,
                "hindsight_best": chosen,
                "hindsight_gain_vs_k3c0_percent": 100 * (
                    controls[chosen]["tokens_per_s"]
                    / controls["fixed:3"]["tokens_per_s"] - 1
                ),
            })
    def pooled(chosen):
        chosen = list(chosen)
        tokens = sum(row["controls"][arm]["tokens"] for row, arm in chosen)
        seconds = sum(row["controls"][arm]["seconds"] for row, arm in chosen)
        return tokens / seconds
    k3 = pooled((row, "fixed:3") for row in conversations)
    hindsight = pooled((row, row["hindsight_best"]) for row in conversations)
    fixed = {arm: pooled((row, arm) for row in conversations) for arm in ARMS}
    return {
        "schema": 1,
        "status": "development-only whole-conversation hindsight, not a live controller",
        "archive_sha256": ARCHIVE_SHA,
        "workloads_sha256": WORKLOADS_SHA,
        "source_sha256": sha(Path(__file__)),
        "conversations": conversations,
        "pooled_fixed_tok_s": fixed,
        "pooled_hindsight_tok_s": hindsight,
        "hindsight_gain_vs_k3c0_percent": 100 * (hindsight / k3 - 1),
        "winner_counts": {
            arm: sum(row["hindsight_best"] == arm for row in conversations)
            for arm in ARMS
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to replace a prior hindsight report")
    result = report(args.archive, args.workloads)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "fixed_tok_s": result["pooled_fixed_tok_s"],
        "hindsight_gain_percent": result["hindsight_gain_vs_k3c0_percent"],
        "winner_counts": result["winner_counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
