"""Run frozen code-trained C/K/D arms on non-code native conversations."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


V9 = Path(__file__).resolve().parent.parent / "joint-v9"
sys.path.insert(0, str(V9))
import eval as v9_eval


MODEL_SHA = v9_eval.MODEL_SHA256
BINARY_SHA = v9_eval.BINARY_SHA256
DOMAINS = ("ifeval", "gsm8k")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def frozen(args):
    if sha(args.binary) != BINARY_SHA or sha(args.model) != MODEL_SHA:
        raise ValueError("Qwen artifact or native binary differs")
    manifest = json.loads((args.workloads / "manifest.json").read_text())
    if manifest["schema"] != 1:
        raise ValueError("non-code workload schema differs")
    arms = json.loads(args.arms.read_text())
    if arms["schema"] != 1 or len(arms["arms"]) != 10:
        raise ValueError("transfer arm inventory differs")
    labels = [item["label"] for item in arms["arms"]]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate frozen C/K/D arm")
    for domain in DOMAINS:
        for phase, count in (("qualification", 1), ("heldout", 16)):
            entries = manifest["groups"][phase][domain]
            if len(entries) != count:
                raise ValueError("non-code conversation count differs")
            for entry in entries:
                if sha(args.workloads / entry["file"]) != entry["sha256"]:
                    raise ValueError("frozen non-code prompt differs")
    for item in arms["arms"]:
        v9_eval.model_hashes(item.get("extra", []))
    return manifest, arms


def qualifier_result(args, arms):
    baseline = "fixed-k20-d3-c0"
    noops = [
        item["label"] for item in arms["arms"]
        if item["label"].startswith(("cd-noop-", "joint-noop-"))
    ]
    if not noops:
        raise ValueError("transfer qualifier lacks model-running no-ops")
    active = {}
    for domain in DOMAINS:
        prefix = f"qualification-{domain}-0"
        reference = args.out / f"{prefix}-{baseline}"
        for turn in range(1, 9):
            expected = (reference / f"turn-{turn}.output.ids").read_bytes()
            for label in noops:
                session = args.out / f"{prefix}-{label}"
                if (session / f"turn-{turn}.output.ids").read_bytes() != expected:
                    raise ValueError(f"sampled {domain} no-op differs: {label}/{turn}")
        active[domain] = {}
        for label in ("cd-new-only", "joint-augmented"):
            result = json.loads((args.out / f"{prefix}-{label}.result.json").read_text())
            active[domain][label] = {
                "c_decisions": result["c_decisions"],
                "c_stops": result["c_stops"],
                "controller_seconds": result["cd_model_s"],
                "k_controller_seconds": result["k_model_s"],
                "mixed": 0 < result["c_stops"] < result["c_decisions"],
            }
        if any(
            row["c_decisions"] == 0 or row["controller_seconds"] <= 0
            for row in active[domain].values()
        ):
            raise ValueError(f"learned C model did not run on {domain}")
        if active[domain]["joint-augmented"]["k_controller_seconds"] <= 0:
            raise ValueError(f"learned K model did not run on {domain}")
    return {
        "schema": 1,
        "status": "qualified",
        "model_sha256": MODEL_SHA,
        "binary_sha256": BINARY_SHA,
        "workload_manifest_sha256": sha(args.workloads / "manifest.json"),
        "arms_sha256": sha(args.arms),
        "byte_identical_noops": noops,
        "c_engagement": active,
    }


def run(args):
    manifest, arms = frozen(args)
    if args.phase == "heldout":
        result = json.loads((args.out.parent / "qualification-result.json").read_text())
        if (
            result["status"] != "qualified"
            or result["model_sha256"] != MODEL_SHA
            or result["binary_sha256"] != BINARY_SHA
            or result["workload_manifest_sha256"] != sha(args.workloads / "manifest.json")
            or result["arms_sha256"] != sha(args.arms)
        ):
            raise ValueError("transfer heldout lacks matching qualifier")
    args.out.mkdir(exist_ok=True)
    for domain_index, domain in enumerate(DOMAINS):
        entries = manifest["groups"][args.phase][domain]
        stop = len(entries) if args.stop is None else args.stop
        if not 0 <= args.start < stop <= len(entries):
            raise ValueError("invalid frozen conversation range")
        for index in range(args.start, stop):
            entry = entries[index]
            args.phase = f"{args.requested_phase}-{domain}"
            order = arms["arms"]
            shift = (index + domain_index * 5) % len(order)
            order = order[shift:] + order[:shift]
            if index % 2:
                order.reverse()
            for item in order:
                row = v9_eval.run_one(args, entry, index, item)
                path = args.out / f"{row['name']}.result.json"
                if path.exists():
                    if json.loads(path.read_text()) != json.loads(json.dumps(row)):
                        raise ValueError("resumed native transfer result changed")
                else:
                    save(path, row)
                print(json.dumps({
                    "session": row["name"],
                    "tok_s": row["tokens"] / row["seconds"],
                }), flush=True)
            args.phase = args.requested_phase
    if args.requested_phase == "qualification" and args.start == 0 and args.stop is None:
        result_path = args.out.parent / "qualification-result.json"
        save(result_path, qualifier_result(args, arms))
        return result_path
    return None


def main():
    parser = argparse.ArgumentParser()
    for name in ("binary", "model", "workloads", "arms", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("qualification", "heldout"), required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    args = parser.parse_args()
    for name in ("binary", "model", "workloads", "arms", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.requested_phase = args.phase
    result = run(args)
    if result is not None:
        print(json.dumps({"qualified": str(result)}))


if __name__ == "__main__":
    main()
