"""Run the same frozen native arm menu on code, prose and math."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


V9 = Path(__file__).resolve().parent.parent / "joint-v9"
sys.path.insert(0, str(V9))
import eval as v9_eval


FULL_SHA = "b35374d34e2a28b31cca0399e6394fa3db3b0f27e9d593b17032a7856b2a5477"
TRAIN_SHA = "2bca403c1edff44a1d707abd60710767f9dbbe2a63721757032b0fbe223c5e00"
VALIDATION_SHA = "bf920b82e0176c4304bcc562ccce34a28384082c888355a280620163a45e5313"
DOMAINS = ("code", "prose", "math")
COUNTS = {"qualification": 1, "validation": 8, "final": 24}
REFERENCE = "fixed-k20-d3-c0"
LABEL = re.compile(r"[a-z0-9][a-z0-9-]*")
FIXED = {
    f"fixed-k{k}-d{depth}-c0": (k, depth)
    for k in (3, 10, 20) for depth in (1, 2, 3, 4)
}
FIXED_C = {
    f"fixed-k{k}-d3-cq{q}"
    for k in (3, 10, 20) for q in (25, 50, 75)
}
EXTRA_KEYS = {
    "topk-model", "depth-model", "confidence-model",
    "joint-model-dir", "depth-variant", "confidence-variant",
    "confidence-fixed",
}


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def manifest_digest(value):
    return hashlib.sha256(
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()


def save(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def current_gpu(meta):
    result = subprocess.check_output(
        [
            "nvidia-smi", "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ], text=True,
    ).strip().splitlines()
    if len(result) != 1 or result[0] != meta["gpu_uuid"]:
        raise ValueError("mixed native phase moved to another physical GPU")
    return result[0]


def freeze(args):
    expected = {
        "qualification": TRAIN_SHA,
        "validation": VALIDATION_SHA,
        "final": FULL_SHA,
    }[args.phase]
    if (
        sha(args.binary) != v9_eval.BINARY_SHA256
        or sha(args.model) != v9_eval.MODEL_SHA256
        or sha(args.workloads / "manifest.json") != expected
        or sha(args.models / "manifest.json")
        != json.loads(args.arms.read_text())["model_manifest_sha256"]
    ):
        raise ValueError("mixed native model, binary or workload pin differs")
    manifest = json.loads((args.workloads / "manifest.json").read_text())
    arms = json.loads(args.arms.read_text())
    meta = json.loads(args.run_meta.read_text())
    if (
        meta["schema"] != 1
        or meta["model_sha256"] != v9_eval.MODEL_SHA256
        or meta["binary_sha256"] != v9_eval.BINARY_SHA256
        or meta["training_workloads_sha256"] != TRAIN_SHA
        or meta["source_full_manifest_sha256"] != FULL_SHA
        or meta["customer_capture"] is not False
        or meta["cuda_allocated"] is not True
        or meta["target_top_k"] != 20
        or meta["temperature"] != 1.0
        or meta["top_p"] != 0.95
    ):
        raise ValueError("mixed native host, sampler or CUDA proof differs")
    gpu_uuid = current_gpu(meta)
    specs = arms["arms"]
    by_label = {item["label"]: item for item in specs}
    if (
        manifest["schema"] != 1
        or set(manifest["groups"]) != {
            "qualification": {"qualification", "training"},
            "validation": {"qualification", "training", "validation"},
            "final": {"qualification", "training", "validation", "final"},
        }[args.phase]
        or set(manifest["groups"][args.phase]) != set(DOMAINS)
        or arms["schema"] != 1
        or arms["phase"] != args.phase
        or arms["source_manifest_sha256"] != FULL_SHA
        or len(by_label) != len(specs)
        or REFERENCE not in by_label
        or by_label[REFERENCE]["role"] != "fixed"
        or by_label[REFERENCE]["arm"] != "fixed:3"
        or by_label[REFERENCE]["k"] != 20
        or by_label[REFERENCE]["cap"] != 3
        or not all(LABEL.fullmatch(item["label"]) for item in specs)
        or len(specs) < 3
    ):
        raise ValueError("mixed native arm or domain inventory differs")
    if args.phase != "final" and (
        manifest["source_full_manifest_sha256"] != FULL_SHA
    ):
        raise ValueError("mixed phase lacks full-source commitment")
    for spec in specs:
        extra = spec.get("extra", [])
        options = dict(item.split("=", 1) for item in extra)
        if (
            spec["role"] not in ("fixed", "learned", "noop")
            or spec["k"] not in (3, 10, 20)
            or spec["cap"] not in (1, 2, 3, 4)
            or any("=" not in item for item in extra)
            or len(options) != len(extra)
            or not set(options).issubset(EXTRA_KEYS)
            or (spec["role"] == "fixed" and (
                (spec["arm"] not in
                 ("fixed:1", "fixed:2", "fixed:3", "fixed:4", "fixed-c3"))
                or (
                    spec["arm"].startswith("fixed:")
                    and bool(extra)
                )
                or (
                    spec["arm"] == "fixed-c3" and (
                        spec["cap"] != 3
                        or not re.fullmatch(
                            rf"fixed-k{spec['k']}-d3-cq(?:25|50|75)",
                            spec["label"],
                        )
                        or set(options) != {"confidence-fixed"}
                    )
                )
            ))
            or (spec["role"] == "learned" and
                spec["arm"] not in ("joint-cd", "joint-ckd", "learn-topk"))
            or (spec["role"] == "noop" and
                spec["arm"] not in ("noop-cd", "noop-ckd", "noop-topk"))
        ):
            raise ValueError("invalid or field-routed mixed C/K/D arm")
        if spec["role"] == "fixed" and spec["label"] in FIXED:
            k, depth = FIXED[spec["label"]]
            if (spec["k"], spec["cap"], spec["arm"]) != (
                k, depth, f"fixed:{depth}"
            ):
                raise ValueError("mixed fixed control does not match label")
        model_hashes = v9_eval.model_hashes(extra)
        if any(
            not Path(path).resolve().is_relative_to(args.models)
            for path in model_hashes
        ):
            raise ValueError("mixed policy weights leave the frozen model root")
        if spec["role"] in ("learned", "noop"):
            if (
                not model_hashes
                or spec["policy_sha256"] != digest(model_hashes)
            ):
                raise ValueError("mixed learned policy weights changed")
    if args.phase != "final":
        if not (set(FIXED) | FIXED_C).issubset(by_label):
            raise ValueError("mixed fixed depth and K controls are missing")
    learned = [
        spec for spec in specs if spec["role"] == "learned"
    ]
    for spec in learned:
        noop = by_label.get(spec["noop_label"])
        if (
            noop is None or noop["role"] != "noop"
            or noop["policy_sha256"] != spec["policy_sha256"]
            or noop["k"] != spec["k"]
            or noop["cap"] != spec["cap"]
            or noop.get("extra", []) != spec.get("extra", [])
            or noop["arm"] != {
                "joint-cd": "noop-cd",
                "joint-ckd": "noop-ckd",
                "learn-topk": "noop-topk",
            }[spec["arm"]]
        ):
            raise ValueError("mixed learned arm lacks exact model-running no-op")
    for domain in DOMAINS:
        entries = manifest["groups"][args.phase][domain]
        if len(entries) != COUNTS[args.phase]:
            raise ValueError(f"mixed {domain} phase count differs")
        for entry in entries:
            if (
                len(entry["turns"]) != 8
                or len(entry["task_ids"]) != 8
                or sha(args.workloads / entry["file"]) != entry["sha256"]
            ):
                raise ValueError(f"mixed {domain} prompt differs")
    return manifest, arms, gpu_uuid


def previous_phase(args, arms):
    if args.phase == "qualification":
        return
    if args.training_manifest is None or (
        sha(args.training_manifest) != TRAIN_SHA
    ):
        raise ValueError("mixed evaluation lacks frozen training projection")
    training = json.loads(args.training_manifest.read_text())
    current = json.loads((args.workloads / "manifest.json").read_text())
    if any(
        training["groups"][phase] != current["groups"][phase]
        for phase in ("qualification", "training")
    ):
        raise ValueError("mixed training prompts changed across phases")
    qualifier_path = args.out.parent / "qualification-result.json"
    receipt = json.loads(qualifier_path.read_text())
    if (
        receipt["status"] != "qualified"
        or receipt["model_sha256"] != v9_eval.MODEL_SHA256
        or receipt["binary_sha256"] != v9_eval.BINARY_SHA256
        or receipt["training_workloads_sha256"] != TRAIN_SHA
        or receipt["model_manifest_sha256"]
        != arms["model_manifest_sha256"]
        or receipt["qualification_arms_sha256"]
        != arms["qualification_arms_sha256"]
        or receipt["gpu_uuid"]
        != json.loads(args.run_meta.read_text())["gpu_uuid"]
    ):
        raise ValueError("mixed validation lacks exact native qualifier")
    if args.phase == "validation" and (
        receipt["qualification_arms_digest"] != digest(arms["arms"])
    ):
        raise ValueError("mixed validation arms changed from qualifier")
    if args.phase != "final":
        return
    if args.validation_manifest is None or (
        sha(args.validation_manifest) != VALIDATION_SHA
    ):
        raise ValueError("mixed final lacks frozen validation projection")
    validation = json.loads(args.validation_manifest.read_text())
    if any(
        validation["groups"][phase] != current["groups"][phase]
        for phase in ("qualification", "training", "validation")
    ):
        raise ValueError("mixed validation prompts changed in final package")
    if manifest_digest(current["groups"]["final"]) != (
        "513beeb27293f98b2e05e71449ab63a15048f4bb9b680900931ef19b02640729"
    ):
        raise ValueError("mixed final prompt commitment differs")
    selected = args.arms.with_name("shared-selected.json")
    choice = json.loads(selected.read_text())
    if (
        arms["selected_from_validation"] != sha(selected)
        or choice["status"] != "selected"
        or choice["scope"] != "one immutable C/K/D controller, no domain route"
        or choice["model_manifest_sha256"]
        != arms["model_manifest_sha256"]
        or choice["source_manifest_sha256"]
        != arms["source_manifest_sha256"]
        or sorted(choice["final_arm_labels"])
        != sorted(item["label"] for item in arms["arms"])
        or choice["selected_policy"]["label"]
        not in {item["label"] for item in arms["arms"]}
    ):
        raise ValueError("mixed final arm differs from one shared selection")


def qualifier(args, arms):
    noops = [
        item for item in arms["arms"] if item["role"] == "noop"
    ]
    active = [
        item for item in arms["arms"] if item["role"] == "learned"
    ]
    if not noops or not active:
        raise ValueError("mixed qualifier lacks a learned/no-op twin")
    engagement = {}
    for domain in DOMAINS:
        prefix = f"qualification-{domain}-0"
        baseline = args.out / f"{prefix}-{REFERENCE}"
        for turn in range(1, 9):
            expected = (
                baseline / f"turn-{turn}.output.ids"
            ).read_bytes()
            for item in noops:
                got = (
                    args.out / f"{prefix}-{item['label']}"
                    / f"turn-{turn}.output.ids"
                ).read_bytes()
                if got != expected:
                    raise ValueError(
                        f"mixed {domain} model-running no-op changed output"
                    )
        engagement[domain] = {}
        for item in active:
            row = json.loads((
                args.out / f"{prefix}-{item['label']}.result.json"
            ).read_text())
            if row["cached_later_turns"] != 7:
                raise ValueError("mixed learned arm lacked native KV reuse")
            if (
                item["arm"] in ("joint-cd", "joint-ckd")
                and (row["c_decisions"] <= 0 or row["cd_model_s"] <= 0)
            ):
                raise ValueError("mixed C/D policy did not engage")
            if (
                item["arm"] in ("learn-topk", "joint-ckd")
                and row["k_model_s"] <= 0
            ):
                raise ValueError("mixed K policy did not engage")
            engagement[domain][item["label"]] = {
                "c_decisions": row["c_decisions"],
                "c_stops": row["c_stops"],
                "k_model_seconds": row["k_model_s"],
                "cd_model_seconds": row["cd_model_s"],
            }
    return {
        "schema": 1,
        "status": "qualified",
        "model_sha256": v9_eval.MODEL_SHA256,
        "binary_sha256": v9_eval.BINARY_SHA256,
        "training_workloads_sha256": TRAIN_SHA,
        "model_manifest_sha256": arms["model_manifest_sha256"],
        "qualification_arms_sha256": sha(args.arms),
        "qualification_arms_digest": digest(arms["arms"]),
        "gpu_uuid":
        json.loads(args.run_meta.read_text())["gpu_uuid"],
        "domains": list(DOMAINS),
        "byte_identical_noops": [
            item["label"] for item in noops
        ],
        "policy_engagement": engagement,
    }


def run(args):
    manifest, arms, gpu_uuid = freeze(args)
    meta = json.loads(args.run_meta.read_text())
    previous_phase(args, arms)
    args.out.mkdir(exist_ok=True)
    specs = arms["arms"]
    for domain_index, domain in enumerate(DOMAINS):
        entries = manifest["groups"][args.requested_phase][domain]
        stop = len(entries) if args.stop is None else args.stop
        if not 0 <= args.start < stop <= len(entries):
            raise ValueError("invalid mixed conversation range")
        for index in range(args.start, stop):
            current_gpu(meta)
            entry = entries[index]
            shift = (index + domain_index * 5) % len(specs)
            order = specs[shift:] + specs[:shift]
            if index % 2:
                order.reverse()
            args.phase = f"{args.requested_phase}-{domain}"
            for spec in order:
                row = v9_eval.run_one(args, entry, index, spec)
                row["gpu_uuid"] = gpu_uuid
                target = args.out / f"{row['name']}.result.json"
                if target.exists():
                    if json.loads(target.read_text()) != row:
                        raise ValueError("resumed mixed result changed")
                else:
                    save(target, row)
                print(json.dumps({
                    "session": row["name"], "status": "native-complete",
                }), flush=True)
            args.phase = args.requested_phase
    if (
        args.requested_phase == "qualification"
        and args.start == 0 and args.stop is None
    ):
        save(
            args.out.parent / "qualification-result.json",
            qualifier(args, arms),
        )


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "binary", "model", "workloads", "arms", "out",
        "run-meta",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=tuple(COUNTS), required=True,
    )
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    args = parser.parse_args()
    for name in (
        "binary", "model", "workloads", "arms", "out",
        "models", "run_meta", "validation_manifest",
        "training_manifest",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    args.requested_phase = args.phase
    run(args)


if __name__ == "__main__":
    main()
