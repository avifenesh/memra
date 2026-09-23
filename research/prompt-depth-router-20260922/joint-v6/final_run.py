"""Run frozen K/C/D policies and fixed controls on fresh Qwen code sessions."""

import argparse
import json
from pathlib import Path

from joint_selection import fixed_arm
from run_native import (
    TOP_K, input_manifests, quality, run_one, save, sha,
)


def frozen(args):
    topk = json.loads((args.out / "selected-topk.json").read_text())
    joint = json.loads((args.out / "selected-joint.json").read_text())
    fixed = json.loads((args.out / "selected-fixed.json").read_text())
    router = (
        args.models / "topk" / f"topk-{topk['variant']}.tsv"
        if topk["variant"] else args.models / "topk" / "topk-fixed20.tsv"
    )
    if (
        joint["schema"] != 1 or topk["schema"] != 1 or fixed["schema"] != 1
        or sha(router) != joint["router_sha256"]
        or joint["fixed_control"] != fixed["best"]
    ):
        raise ValueError("frozen K/C/D selection changed before fresh evaluation")
    if topk["variant"] and sha(router) != topk["model_sha256"]:
        raise ValueError("selected K router changed")
    variant = joint["variant"]
    if variant is not None:
        for k in joint["available_top_k"]:
            for kind in ("depth", "confidence"):
                path = args.models / f"topk{k}" / f"{kind}-{variant}.tsv"
                if not path.is_file():
                    raise ValueError("joint action lacks a fitted C/D model")
    return topk, joint, fixed, router


def joint_args(models, router, variant):
    return (
        f"topk-model={router}",
        f"joint-model-dir={models}",
        f"depth-variant={variant}",
        f"confidence-variant={variant}",
    )


def qualifier(args, fresh, topk, joint, router):
    entry = fresh["groups"]["qualification"][0]
    arms = [("topk20-d3-c0", 20, "fixed:3", 3, ())]
    if topk["variant"] is not None:
        arms.append((
            "topk-noop", 20, "noop-topk", 3, (f"topk-model={router}",),
        ))
    if joint["variant"] is not None:
        arms.append((
            "joint-noop", 20, "noop-ckd", 4,
            joint_args(args.models, router, joint["variant"]),
        ))
    records = [
        run_one(args, entry, args.fresh, "policy-qualification-0",
                label, k, arm, cap, extra=extra)
        for label, k, arm, cap, extra in arms
    ]
    baseline = args.out / records[0]["name"]
    for turn in range(1, 9):
        reference = (baseline / f"turn-{turn}.output.ids").read_bytes()
        for record in records[1:]:
            if (args.out / record["name"] / f"turn-{turn}.output.ids").read_bytes() != reference:
                raise ValueError("sampled no-op K/C/D path changed target output bytes")
    functional = {
        row["variant"]: sum(
            quality.probe(
                (args.out / row["name"] / f"turn-{turn}.answer.txt").read_text(),
                expected["function"],
            )["pass"]
            for turn, expected in enumerate(entry["turns"], 1)
        )
        for row in records
    }
    if any(
        row["format"] != 8 or row["loops"] or functional[row["variant"]] != 8
        for row in records
    ):
        raise ValueError("policy qualifier failed code or loop gate")
    save(args.out / "policy-qualification-result.json", {
        "schema": 1, "status": "sampled-eight-turn-byte-identical",
        "binary_sha256": args.binary_sha,
        "source_sha256": args.source_sha,
        "router_sha256": sha(router),
        "joint_variant": joint["variant"],
        "functional_pass": functional,
    })
    save(args.out / "policy-qualification-summary.json", {
        "schema": 1, "phase": "policy-qualification", "records": records,
    })
    return records


def heldout(args, fresh, topk, joint, fixed, router):
    result = json.loads((args.out / "policy-qualification-result.json").read_text())
    if (
        result["status"] != "sampled-eight-turn-byte-identical"
        or result["binary_sha256"] != args.binary_sha
        or result["source_sha256"] != args.source_sha
        or result["router_sha256"] != sha(router)
        or result["joint_variant"] != joint["variant"]
    ):
        raise ValueError("frozen sampled no-op qualifier does not match heldout binary")
    arms = [
        (f"topk{k}-d3-c0", k, "fixed:3", 3, ())
        for k in TOP_K
    ]
    best = fixed_arm(args.out, fixed["best"])
    if best[0] not in {row[0] for row in arms}:
        arms.append(best)
    if topk["variant"] is not None:
        arms.extend((
            ("topk-learned", 20, "learn-topk", 3, (f"topk-model={router}",)),
            ("topk-noop", 20, "noop-topk", 3, (f"topk-model={router}",)),
        ))
    if joint["variant"] is not None:
        variant = joint["variant"]
        arms.extend((
            ("joint-learned", 20, "joint-ckd", 4,
             joint_args(args.models, router, variant)),
            ("joint-noop", 20, "noop-ckd", 4,
             joint_args(args.models, router, variant)),
        ))
        for k in (20, 3):
            if all(
                (args.models / f"topk{k}" / f"{kind}-{variant}.tsv").is_file()
                for kind in ("depth", "confidence")
            ):
                depth_path = args.models / f"topk{k}" / f"depth-{variant}.tsv"
                confidence_path = args.models / f"topk{k}" / f"confidence-{variant}.tsv"
                arms.append((
                    f"cd-fixedk{k}", k, "joint-cd", 4, (
                        f"depth-model={depth_path}",
                        f"confidence-model={confidence_path}",
                    ),
                ))
    records = []
    for index, entry in enumerate(fresh["groups"]["heldout"]):
        order = arms[index % len(arms):] + arms[:index % len(arms)]
        if index % 2:
            order.reverse()
        for label, k, arm, cap, extra in order:
            records.append(run_one(
                args, entry, args.fresh, f"heldout-{index}",
                label, k, arm, cap, extra=extra,
            ))
    save(args.out / "heldout-summary.json", {
        "schema": 1, "phase": "heldout", "records": records,
    })
    return records


def main():
    parser = argparse.ArgumentParser()
    for name in ("binary", "model", "development", "fresh", "out", "models"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--phase", choices=("qualifier", "heldout"), required=True)
    args = parser.parse_args()
    for name in ("binary", "model", "development", "fresh", "out", "models"):
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(exist_ok=True)
    _, fresh = input_manifests(args)
    topk, joint, fixed, router = frozen(args)
    records = (
        qualifier(args, fresh, topk, joint, router)
        if args.phase == "qualifier"
        else heldout(args, fresh, topk, joint, fixed, router)
    )
    print(json.dumps({"phase": args.phase, "sessions": len(records)}))


if __name__ == "__main__":
    main()
