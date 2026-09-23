"""Fit separate C/D learners at each quality-eligible sampler top-k."""

import argparse
import json
from pathlib import Path
import sys


V4 = Path(__file__).resolve().parent.parent / "joint-v4"
sys.path.insert(0, str(V4))
import analyze_training
import train_depth
import train_confidence


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def fit(root, models, views):
    summary = json.loads((root / "development-summary.json").read_text())
    qualifier = json.loads((root / "qualification-result.json").read_text())
    if summary["schema"] != 1 or qualifier["schema"] != 1:
        raise ValueError("v5 development and qualifier schemas differ")
    models.mkdir(exist_ok=True)
    views.mkdir(exist_ok=True)
    result = {}
    for k in (3, 10, 20):
        if not qualifier["eligible"][str(k)]:
            result[str(k)] = {"status": "quality-ineligible-on-qualifier"}
            continue
        fixed = [
            {**row, "variant": "k3-c0"}
            for row in summary["records"]
            if row["sampler_top_k"] == k and row["arm"] == "fixed:3"
        ]
        randomized = [
            row for row in summary["records"]
            if row["sampler_top_k"] == k and row["arm"] == "explore-d"
        ]
        if len(fixed) != 6 or len(randomized) != 6:
            raise ValueError(f"top-k={k} needs six matched fixed and random conversations")
        view = views / f"topk{k}"
        view.mkdir()
        for row in fixed + randomized:
            (view / row["name"]).symlink_to((root / row["name"]).resolve())
        save(view / "training-summary.json", {
            "schema": 1, "phase": "training",
            "records": fixed + randomized,
        })
        try:
            analysis = analyze_training.score(view)
        except ValueError as error:
            result[str(k)] = {
                "status": "quality-or-exposure-ineligible",
                "reason": str(error),
            }
            continue
        save(view / "training-analysis.json", analysis)
        save(root / f"topk{k}-training-analysis.json", analysis)
        destination = models / f"topk{k}"
        destination.mkdir()
        depths = [
            train_depth.fit(view, destination / f"depth-{variant}.tsv", variant, 100)
            for variant in train_depth.FEATURES
        ]
        confidence = [
            train_confidence.fit(
                view, destination / f"confidence-{variant}.tsv", variant,
            )
            for variant in train_depth.FEATURES
        ]
        save(destination / "models.json", depths)
        save(destination / "confidence-models.json", confidence)
        result[str(k)] = {
            "status": "trained",
            "fixed_tok_s": analysis["fixed_baseline_tok_s"],
            "randomized_eligible_rounds": analysis["randomized_eligible_rounds"],
            "model_directory": destination.name,
        }
    save(root / "model-training-summary.json", {
        "schema": 1,
        "scope": "temperature 1.0, top-p 0.95, separate sampler top-k models",
        "top_k": result,
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "models", "views"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fit(
        args.root.resolve(), args.models.resolve(), args.views.resolve(),
    )))


if __name__ == "__main__":
    main()
