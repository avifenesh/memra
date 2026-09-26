"""Run frozen v11 non-code candidate arms through native KV conversations."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


V9 = Path(__file__).resolve().parent.parent / "joint-v9"
sys.path.insert(0, str(V9))
import eval as v9_eval


VALIDATION_SHA = "e8e60712c7053d9f64eba42b3f9a7b04ad946971fe095364cabb163b6c73eacd"
FINAL_SHA = "71c538295aeb970856f5feb5d11c888aa2dea4d220f41ab844d6e0587108a3df"
HELDOUT_COMMIT = "ec85aaaf044cca42779c12144b3c6ef1acb96a96155e866e6db6c93c356a1f62"
DOMAINS = ("ifeval", "gsm8k")
REFERENCE = "fixed-k20-d3-c0"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def arm_digest(arms):
    return hashlib.sha256(json.dumps(
        arms, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def save(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def frozen(args):
    expected = FINAL_SHA if args.phase == "heldout" else VALIDATION_SHA
    if (
        sha(args.binary) != v9_eval.BINARY_SHA256
        or sha(args.model) != v9_eval.MODEL_SHA256
        or sha(args.workloads / "manifest.json") != expected
    ):
        raise ValueError("v11 native artifact or workload differs")
    manifest = json.loads((args.workloads / "manifest.json").read_text())
    arms = json.loads(args.arms.read_text())
    if (
        manifest["schema"] != 1
        or args.phase not in ("qualification", "validation", "heldout")
        or arms["schema"] != 1
        or arms["phase"] != args.phase
        or REFERENCE not in {row["label"] for row in arms["arms"]}
    ):
        raise ValueError("v11 phase or fixed control differs")
    if len({row["label"] for row in arms["arms"]}) != len(arms["arms"]):
        raise ValueError("v11 arm labels repeat")
    for spec in arms["arms"]:
        if (
            spec["k"] not in (3, 10, 20)
            or spec["cap"] not in (3, 4)
            or any("=" not in item for item in spec.get("extra", []))
        ):
            raise ValueError("invalid native C/K/D arm")
        v9_eval.model_hashes(spec.get("extra", []))
    for domain in DOMAINS:
        entries = manifest["groups"][args.phase][domain]
        if len(entries) != manifest["splits"][args.phase]:
            raise ValueError("v11 topic conversation count differs")
        for entry in entries:
            if sha(args.workloads / entry["file"]) != entry["sha256"]:
                raise ValueError("v11 frozen topic prompt differs")
    return manifest, arms


def previous_phase(args, arms):
    if args.phase == "qualification":
        return
    receipt = json.loads(
        (args.out.parent / "qualification-result.json").read_text()
    )
    if (
        receipt["status"] != "qualified"
        or receipt["model_sha256"] != v9_eval.MODEL_SHA256
        or receipt["binary_sha256"] != v9_eval.BINARY_SHA256
        or receipt["workload_manifest_sha256"] != VALIDATION_SHA
        or receipt["model_manifest_sha256"]
        != arms["model_manifest_sha256"]
        or receipt["training_manifest_sha256"]
        != arms["training_manifest_sha256"]
    ):
        raise ValueError("matching native qualifier is required")
    if args.phase == "validation":
        if receipt["arm_inventory_sha256"] != arm_digest(arms["arms"]):
            raise ValueError("validation arms changed from qualifier")
    else:
        validation_manifest_path = (
            args.workloads.parent / "validation-workloads/manifest.json"
        )
        validation_manifest = json.loads(validation_manifest_path.read_text())
        final_manifest = json.loads(
            (args.workloads / "manifest.json").read_text()
        )
        if (
            sha(validation_manifest_path) != VALIDATION_SHA
            or sha(args.workloads / "manifest.json") != FINAL_SHA
            or validation_manifest["source_full_manifest_sha256"] != FINAL_SHA
            or validation_manifest["heldout_group_sha256"]
            != HELDOUT_COMMIT
            or any(
                validation_manifest["groups"][phase]
                != final_manifest["groups"][phase]
                for phase in ("qualification", "training", "validation")
            )
        ):
            raise ValueError("final prompt phase lacks frozen validation lineage")
        selected = args.arms.with_name("selected.json")
        choice = json.loads(selected.read_text())
        validation = args.arms.with_name("validation-score.json")
        if (
            sha(selected) != arms["selected_from_validation"]
            or choice["phase"] != "validation-selected"
            or choice["status"] != "selected"
            or sha(validation) != choice["validation_score_sha256"]
            or sorted(choice["final_arms"])
            != sorted(item["label"] for item in arms["arms"])
            or choice["final_arm_inventory_sha256"]
            != arm_digest(arms["arms"])
            or choice["qualification_arm_inventory_sha256"]
            != receipt["arm_inventory_sha256"]
        ):
            raise ValueError("final arms lack frozen validation selection")


def qualifier(args, arms):
    noops = [
        row["label"] for row in arms["arms"]
        if "-noop-" in row["label"]
    ]
    if not noops:
        raise ValueError("no model-running controls in qualifier")
    active = {}
    for domain in DOMAINS:
        prefix = f"qualification-{domain}-0"
        reference = args.out / f"{prefix}-{REFERENCE}"
        for turn in range(1, 9):
            expected = (reference / f"turn-{turn}.output.ids").read_bytes()
            for label in noops:
                session = args.out / f"{prefix}-{label}"
                if (session / f"turn-{turn}.output.ids").read_bytes() != expected:
                    raise ValueError(f"sampled no-op changed output: {domain}/{label}")
        active[domain] = {}
        for spec in arms["arms"]:
            if spec["arm"] not in ("joint-cd", "joint-ckd", "learn-topk"):
                continue
            row = json.loads(
                (args.out / f"{prefix}-{spec['label']}.result.json").read_text()
            )
            if (
                spec["arm"] in ("joint-cd", "joint-ckd")
                and (row["c_decisions"] <= 0 or row["cd_model_s"] <= 0)
            ):
                raise ValueError(f"C model not engaged: {domain}/{spec['label']}")
            if (
                spec["arm"] in ("joint-ckd", "learn-topk")
                and row["k_model_s"] <= 0
            ):
                raise ValueError(f"K model not engaged: {domain}/{spec['label']}")
            active[domain][spec["label"]] = {
                "c_decisions": row["c_decisions"],
                "c_stops": row["c_stops"],
                "cd_model_seconds": row["cd_model_s"],
                "k_model_seconds": row["k_model_s"],
            }
    return {
        "schema": 1,
        "status": "qualified",
        "model_sha256": v9_eval.MODEL_SHA256,
        "binary_sha256": v9_eval.BINARY_SHA256,
        "workload_manifest_sha256": VALIDATION_SHA,
        "model_manifest_sha256": arms["model_manifest_sha256"],
        "training_manifest_sha256": arms["training_manifest_sha256"],
        "arm_inventory_sha256": arm_digest(arms["arms"]),
        "byte_identical_noops": noops,
        "policy_engagement": active,
    }


def run(args):
    manifest, arms = frozen(args)
    previous_phase(args, arms)
    args.out.mkdir(exist_ok=True)
    for domain_index, domain in enumerate(DOMAINS):
        entries = manifest["groups"][args.requested_phase][domain]
        stop = len(entries) if args.stop is None else args.stop
        if not 0 <= args.start < stop <= len(entries):
            raise ValueError("invalid v11 conversation range")
        for index in range(args.start, stop):
            entry = entries[index]
            order = arms["arms"]
            shift = (index + domain_index * 5) % len(order)
            order = order[shift:] + order[:shift]
            if index % 2:
                order.reverse()
            args.phase = f"{args.requested_phase}-{domain}"
            for spec in order:
                row = v9_eval.run_one(args, entry, index, spec)
                target = args.out / f"{row['name']}.result.json"
                if target.exists():
                    if json.loads(target.read_text()) != row:
                        raise ValueError("resumed v11 native result changed")
                else:
                    save(target, row)
                print(json.dumps({
                    "session": row["name"],
                    "tok_s": row["tokens"] / row["seconds"],
                }), flush=True)
            args.phase = args.requested_phase
    if args.requested_phase == "qualification" and args.start == 0 and args.stop is None:
        save(args.out.parent / "qualification-result.json", qualifier(args, arms))


def main():
    parser = argparse.ArgumentParser()
    for name in ("binary", "model", "workloads", "arms", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("qualification", "validation", "heldout"),
        required=True,
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    args = parser.parse_args()
    for name in ("binary", "model", "workloads", "arms", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.requested_phase = args.phase
    run(args)


if __name__ == "__main__":
    main()
