"""Freeze a bounded set of fixed and learned validation alternatives."""

import argparse
import gzip
import json
from pathlib import Path


def quantile(values, fraction):
    ordered = sorted(values)
    if len(ordered) < 150:
        raise ValueError("too few new GPU proposal probabilities for fixed C")
    return ordered[int(fraction * (len(ordered) - 1))]


def arm(label, k, action, cap, *extra):
    return {
        "label": label, "k": k, "arm": action,
        "cap": cap, "extra": list(extra),
    }


def build(new, k_models, cd_models, out):
    with gzip.open(new / "c.jsonl.gz", "rt") as source:
        q = [
            item["chosen_probability"]
            for line in source
            if (item := json.loads(line))["draft_k"] == 20
            and item["offer_position"] == 1
        ]
    c_low, c_mid = quantile(q, 0.25), quantile(q, 0.5)
    arms = [
        arm(f"fixed-k{k}-d3-c0", k, "fixed:3", 3)
        for k in (3, 10, 20)
    ]
    arms.append(arm("fixed-k20-d4-c0", 20, "fixed:4", 4))
    arms.extend((
        arm("fixed-k20-d3-c-low-second", 20, "fixed-c3", 3,
            f"confidence-fixed=0,{c_low:.9g}"),
        arm("fixed-k20-d3-c-mid-second", 20, "fixed-c3", 3,
            f"confidence-fixed=0,{c_mid:.9g}"),
    ))
    for source in ("new-only", "augmented"):
        router = (
            k_models / source / "ridge-100" / "topk-prior.tsv"
        ).resolve()
        cd = (cd_models / source).resolve()
        depth = cd / "topk20/depth-history.tsv"
        confidence = cd / "topk20/confidence-history.tsv"
        if not all(path.is_file() for path in (
            router, depth, confidence,
            *(cd / f"topk{k}/{kind}-history.tsv"
              for k in (3, 10, 20)
              for kind in ("depth", "confidence")),
        )):
            raise ValueError(f"missing {source} native policy model")
        arms.extend((
            arm(f"k-{source}", 20, "learn-topk", 3,
                f"topk-model={router}"),
            arm(f"k-noop-{source}", 20, "noop-topk", 3,
                f"topk-model={router}"),
            arm(f"cd-{source}", 20, "joint-cd", 4,
                f"depth-model={depth}", f"confidence-model={confidence}"),
            arm(f"cd-noop-{source}", 20, "noop-cd", 4,
                f"depth-model={depth}", f"confidence-model={confidence}"),
            arm(f"joint-{source}", 20, "joint-ckd", 4,
                f"topk-model={router}", f"joint-model-dir={cd}",
                "depth-variant=history", "confidence-variant=history"),
            arm(f"joint-noop-{source}", 20, "noop-ckd", 4,
                f"topk-model={router}", f"joint-model-dir={cd}",
                "depth-variant=history", "confidence-variant=history"),
        ))
    for variant in ("token", "prior"):
        depth = (cd_models / "new-only" / f"topk20/depth-{variant}.tsv").resolve()
        confidence = (
            cd_models / "new-only" / f"topk20/confidence-{variant}.tsv"
        ).resolve()
        if not depth.is_file() or not confidence.is_file():
            raise ValueError(f"missing new-only C/D {variant} model")
        arms.extend((
            arm(f"cd-{variant}", 20, "joint-cd", 4,
                f"depth-model={depth}", f"confidence-model={confidence}"),
            arm(f"cd-noop-{variant}", 20, "noop-cd", 4,
                f"depth-model={depth}", f"confidence-model={confidence}"),
        ))
    if len(arms) != 22:
        raise ValueError("validation alternative count differs")
    out.mkdir(exist_ok=False)
    for phase, chosen in (
        ("qualification", arms),
        ("validation", arms),
    ):
        (out / f"{phase}-arms.json").write_text(
            json.dumps({
                "schema": 1,
                "phase": phase,
                "fixed_c": {
                    "low_second": c_low,
                    "median_second": c_mid,
                },
                "arms": chosen,
            }, indent=2, sort_keys=True) + "\n"
        )
    return {"arms": len(arms), "low_second": c_low, "median_second": c_mid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--k-models", type=Path, required=True)
    parser.add_argument("--cd-models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.new, args.k_models, args.cd_models, args.out)))


if __name__ == "__main__":
    main()
