"""Choose one native K/C/D policy on old development conversations."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from run_native import (
    input_manifests, old_code_probes, run_one, save, sha,
)


FIXED = re.compile(r"topk(3|10|20)-d([234])-c(?:0|-(.+))$")


def fixed_arm(root, name):
    match = FIXED.fullmatch(name)
    if match is None:
        raise ValueError("selected fixed C/K/D name differs")
    k, d, cut = match.groups()
    k, d = int(k), int(d)
    if cut is None:
        return name, k, f"fixed:{d}", d, ()
    candidates = json.loads((root / "fixed-confidence-candidates.json").read_text())
    if candidates["sampler_top_k"] != k or d != 3:
        raise ValueError("fixed confidence control differs from frozen candidate grid")
    pair = candidates["cutoffs"][cut]
    return name, k, "fixed-c3", 3, (
        f"confidence-fixed={pair[0]:.9g},{pair[1]:.9g}",
    )


def scored(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["variant"]].append(row)
    output = {}
    for name, group in grouped.items():
        if len(group) != 3:
            raise ValueError("joint policy selection lacks three conversations")
        tokens = sum(row["tokens"] for row in group)
        seconds = sum(row["seconds"] for row in group)
        looped = sum(row["loops"] for row in group)
        output[name] = {
            "tokens": tokens, "seconds": seconds,
            "tok_s": tokens / seconds if looped == 0 else None,
            "format_pass": sum(row["format"] for row in group),
            "functional_pass": sum(row["functional_pass"] for row in group),
            "loops": looped,
        }
    return output


def select(args, old):
    k_selection = json.loads((args.out / "selected-topk.json").read_text())
    k_training = json.loads((args.out / "topk-training-summary.json").read_text())
    fixed = json.loads((args.out / "selected-fixed.json").read_text())
    models = args.models.resolve()
    router = (
        models / "topk" / f"topk-{k_selection['variant']}.tsv"
        if k_selection["variant"] is not None
        else models / "topk" / "topk-fixed20.tsv"
    )
    expected = (
        k_selection["model_sha256"]
        if k_selection["variant"] is not None
        else k_training["fixed20_model_sha256"]
    )
    if sha(router) != expected:
        raise ValueError("frozen K router or quality fallback differs")
    available = {
        int(line.split("\t")[1])
        for line in router.read_text().splitlines()
        if line.startswith("K\t")
    }
    for k in available:
        for variant in ("token", "history", "prior"):
            for kind in ("depth", "confidence"):
                path = models / f"topk{k}" / f"{kind}-{variant}.tsv"
                if not path.is_file():
                    raise ValueError("joint K action lacks matched C/D model")
    best_fixed = fixed_arm(args.out, fixed["best"])
    topics = old["groups"]["heldout"][3:]
    records = []
    for index, entry in enumerate(topics):
        arms = [
            ("topk20-d3-c0", 20, "fixed:3", 3, ()),
            best_fixed,
        ]
        if best_fixed[0] == "topk20-d3-c0":
            arms.pop()
        if k_selection["variant"] is not None:
            arms.append((
                "topk-only", 20, "learn-topk", 3,
                (f"topk-model={router}",),
            ))
        for variant in ("token", "history", "prior"):
            arms.append((
                f"joint-{variant}", 20, "joint-ckd", 4, (
                    f"topk-model={router}",
                    f"joint-model-dir={models}",
                    f"depth-variant={variant}",
                    f"confidence-variant={variant}",
                ),
            ))
        order = arms[index % len(arms):] + arms[:index % len(arms)]
        if index % 2:
            order.reverse()
        for label, k, arm, cap, extra in order:
            records.append(run_one(
                args, entry, args.development, f"joint-selection-{index}",
                label, k, arm, cap, extra=extra,
            ))
    for row in records:
        index = int(row["name"].split("-")[2])
        row["functional_pass"] = old_code_probes(
            args.out / row["name"], topics[index],
        )
    metrics = scored(records)
    eligible = [
        variant for variant in ("token", "history", "prior")
        if metrics[f"joint-{variant}"]["format_pass"] == 24
        and metrics[f"joint-{variant}"]["functional_pass"] == 24
        and metrics[f"joint-{variant}"]["loops"] == 0
    ]
    chosen = max(
        eligible, key=lambda variant: metrics[f"joint-{variant}"]["tok_s"],
        default=None,
    )
    if chosen is not None:
        for index, entry in enumerate(topics):
            row = run_one(
                args, entry, args.development, f"joint-selection-{index}",
                "joint-noop", 20, "noop-ckd", 4, extra=(
                    f"topk-model={router}",
                    f"joint-model-dir={models}",
                    f"depth-variant={chosen}",
                    f"confidence-variant={chosen}",
                ),
            )
            row["functional_pass"] = old_code_probes(
                args.out / row["name"], entry,
            )
            records.append(row)
        metrics = scored(records)
    save(args.out / "joint-selection-summary.json", {
        "schema": 1, "phase": "joint-selection", "records": records,
    })
    save(args.out / "selected-joint.json", {
        "schema": 1,
        "status": "selected" if chosen else "no-quality-qualified-joint",
        "variant": chosen,
        "router_sha256": sha(router),
        "router_variant": k_selection["variant"],
        "fixed_control": fixed["best"],
        "available_top_k": sorted(available),
        "selection_scores": metrics,
    })
    return chosen


def main():
    parser = argparse.ArgumentParser()
    for name in ("binary", "model", "development", "fresh", "out", "models"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    for name in ("binary", "model", "development", "fresh", "out", "models"):
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(exist_ok=True)
    old, _ = input_manifests(args)
    print(json.dumps({"selected_variant": select(args, old)}))


if __name__ == "__main__":
    main()
