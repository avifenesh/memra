"""Prove fixed D=1/2 engage on the pinned native binary before training."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "joint-v9"))
import eval as v9_eval
import collect as v9_collect


TRAIN_SHA = "2bca403c1edff44a1d707abd60710767f9dbbe2a63721757032b0fbe223c5e00"
DOMAINS = ("code", "prose", "math")


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


def pilot(args):
    if (
        sha(args.model) != v9_eval.MODEL_SHA256
        or sha(args.binary) != v9_eval.BINARY_SHA256
        or sha(args.workloads / "manifest.json") != TRAIN_SHA
    ):
        raise ValueError("fixed depth pilot source changed")
    meta = json.loads(args.run_meta.read_text())
    gpu = subprocess.check_output(
        [
            "nvidia-smi", "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ], text=True,
    ).strip().splitlines()
    if (
        meta["model_sha256"] != v9_eval.MODEL_SHA256
        or meta["binary_sha256"] != v9_eval.BINARY_SHA256
        or meta["training_workloads_sha256"] != TRAIN_SHA
        or meta["customer_capture"] is not False
        or len(gpu) != 1
        or gpu[0] != meta["gpu_uuid"]
    ):
        raise ValueError("fixed depth pilot moved research GPU")
    manifest = json.loads(
        (args.workloads / "manifest.json").read_text()
    )
    if set(manifest["groups"]) != {
        "qualification", "training",
    }:
        raise ValueError("fixed depth pilot saw reserved prompts")
    args.out.mkdir(exist_ok=True)
    sessions = []
    for domain in DOMAINS:
        entry = manifest["groups"]["qualification"][domain][0]
        for depth in (1, 2):
            spec = {
                "label": f"fixed-k20-d{depth}-c0",
                "k": 20, "cap": depth,
                "arm": f"fixed:{depth}", "extra": [],
            }
            native_args = SimpleNamespace(
                binary=args.binary, model=args.model,
                workloads=args.workloads, out=args.out,
                phase=f"pilot-{domain}",
            )
            row = v9_eval.run_one(native_args, entry, 0, spec)
            rounds = v9_collect.table(
                args.out / row["name"] / "rounds.tsv"
            )
            actual_depths = [
                int(item["draft_depth"])
                for item in rounds
                if item["eligible_for_learning"] == "true"
            ]
            if (
                not actual_depths
                or max(actual_depths) > depth
                or (depth == 1 and set(actual_depths) != {1})
                or (depth == 2 and 2 not in actual_depths)
                or row["cached_later_turns"] != 7
                or set(row["k_actions"]) != {20}
                or row["c_decisions"] != 0
                or row["c_stops"] != 0
            ):
                raise ValueError("native fixed depth did not engage")
            result_path = args.out / f"{row['name']}.result.json"
            if result_path.exists():
                if json.loads(result_path.read_text()) != row:
                    raise ValueError("resumed fixed-depth pilot changed")
            else:
                save(result_path, row)
            sessions.append({
                "domain": domain, "depth": depth,
                "session": row["name"],
                "eligible_rounds": len(actual_depths),
                "actual_depths": {
                    str(d): actual_depths.count(d)
                    for d in sorted(set(actual_depths))
                },
                "loop_hits": row["loops"],
            })
    return {
        "schema": 1,
        "status": "fixed-D1-D2-full-head-and-KV-engaged",
        "model_sha256": v9_eval.MODEL_SHA256,
        "binary_sha256": v9_eval.BINARY_SHA256,
        "training_workloads_sha256": TRAIN_SHA,
        "run_meta_sha256": sha(args.run_meta),
        "gpu_uuid": meta["gpu_uuid"],
        "sessions": sessions,
    }


def replay(root, receipt_path, workloads, run_meta):
    receipt = json.loads(receipt_path.read_text())
    meta = json.loads(run_meta.read_text())
    manifest = json.loads(
        (workloads / "manifest.json").read_text()
    )
    if (
        receipt["status"] != "fixed-D1-D2-full-head-and-KV-engaged"
        or receipt["model_sha256"] != v9_eval.MODEL_SHA256
        or receipt["binary_sha256"] != v9_eval.BINARY_SHA256
        or receipt["training_workloads_sha256"] != TRAIN_SHA
        or receipt["run_meta_sha256"] != sha(run_meta)
        or receipt["gpu_uuid"] != meta["gpu_uuid"]
        or len(receipt["sessions"]) != 6
    ):
        raise ValueError("sealed fixed-depth pilot receipt differs")
    for item in receipt["sessions"]:
        domain = item["domain"]
        depth = item["depth"]
        if domain not in DOMAINS or depth not in (1, 2):
            raise ValueError("sealed pilot domain or depth differs")
        name = f"pilot-{domain}-0-fixed-k20-d{depth}-c0"
        if item["session"] != name:
            raise ValueError("sealed fixed-depth pilot name differs")
        spec = {
            "label": f"fixed-k20-d{depth}-c0",
            "arm": f"fixed:{depth}", "k": 20,
        }
        entry = manifest["groups"]["qualification"][domain][0]
        observed = v9_eval.verify(root / name, entry, spec)
        recorded = json.loads(
            (root / f"{name}.result.json").read_text()
        )
        depths = [
            int(row["draft_depth"])
            for row in v9_collect.table(root / name / "rounds.tsv")
            if row["eligible_for_learning"] == "true"
        ]
        if (
            observed != recorded
            or not depths
            or max(depths) > depth
            or (depth == 1 and set(depths) != {1})
            or (depth == 2 and 2 not in depths)
            or item["eligible_rounds"] != len(depths)
            or item["actual_depths"] != {
                str(d): depths.count(d) for d in sorted(set(depths))
            }
            or item["loop_hits"] != observed["loops"]
        ):
            raise ValueError("sealed D1/D2 native pilot changed")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "binary", "model", "workloads", "run-meta", "out",
        "receipt",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "binary", "model", "workloads", "run_meta", "out",
        "receipt",
    ):
        setattr(args, name, getattr(args, name).resolve())
    result = pilot(args)
    save(args.receipt, result)
    print(json.dumps({
        "status": result["status"],
        "sessions": len(result["sessions"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
