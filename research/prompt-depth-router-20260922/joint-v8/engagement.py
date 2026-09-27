"""Require live after-offer C decisions on the corrected single-head graph."""

import argparse
import json
from pathlib import Path

from run_native import input_manifests, quality, run_one, save, sha, v4


def main():
    parser = argparse.ArgumentParser()
    for name in ("binary", "model", "development", "fresh", "out", "models"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    for name in ("binary", "model", "development", "fresh", "out", "models"):
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(exist_ok=True)
    _, fresh = input_manifests(args)
    router = args.models / "topk" / "topk-prior.tsv"
    record = run_one(
        args, fresh["groups"]["qualification"][0], args.fresh,
        "single-head-engagement-0", "joint-token", 20, "joint-ckd", 4,
        extra=(
            f"topk-model={router}",
            f"joint-model-dir={args.models}",
            "depth-variant=token",
            "confidence-variant=token",
        ),
    )
    turns = v4.table(args.out / record["name"] / "turns.tsv")
    decisions = sum(int(turn["confidence_decisions"]) for turn in turns)
    stops = sum(int(turn["confidence_stops"]) for turn in turns)
    functional = sum(
        quality.probe(
            (args.out / record["name"] / f"turn-{turn}.answer.txt").read_text(),
            expected["function"],
        )["pass"]
        for turn, expected in enumerate(
            fresh["groups"]["qualification"][0]["turns"], 1,
        )
    )
    if (
        decisions == 0 or not 0 < stops < decisions
        or record["format"] != 8 or functional != 8 or record["loops"]
    ):
        raise ValueError("v8 sampled C did not vary or code qualifier failed")
    result = {
        "schema": 1,
        "status": "prime-probability-learned-C-engaged",
        "binary_sha256": args.binary_sha,
        "source_sha256": args.source_sha,
        "router_sha256": sha(router),
        "confidence_decisions": decisions,
        "confidence_stops": stops,
        "format_pass": record["format"],
        "functional_pass": functional,
        "loops": record["loops"],
        "observed_draft_top_k": record["observed_draft_top_k"],
    }
    save(args.out / "engagement-result.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
