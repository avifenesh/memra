"""Collect randomized C/K/D training receipts on fresh open-prose prompts."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys


BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "joint-v9"))
import collect as v9_collect


def load_v11_collect():
    path = BASE / "joint-v11/collect.py"
    spec = importlib.util.spec_from_file_location("v12_parent_collect", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v11_collect = load_v11_collect()
TRAIN_SHA = "2bca403c1edff44a1d707abd60710767f9dbbe2a63721757032b0fbe223c5e00"
FULL_SHA = "b35374d34e2a28b31cca0399e6394fa3db3b0f27e9d593b17032a7856b2a5477"
ARMS = tuple(
    (k, kind) for k in (3, 10, 20)
    for kind in ("fixed-d3", "explore-d")
) + ((None, "random-k"),)


def freeze(args):
    if (
        v9_collect.sha(args.model) != v9_collect.MODEL_SHA256
        or v9_collect.sha(args.binary) != v9_collect.BINARY_SHA256
        or v9_collect.sha(args.workloads / "manifest.json") != TRAIN_SHA
    ):
        raise ValueError("open-prose randomized training pin differs")
    manifest = json.loads((args.workloads / "manifest.json").read_text())
    if (
        manifest["schema"] != 1
        or manifest["source_full_manifest_sha256"] != FULL_SHA
        or set(manifest["groups"]) != {"qualification", "training"}
        or set(manifest["groups"]["training"]) != {"code", "prose", "math"}
        or len(manifest["groups"]["training"]["prose"]) != 16
    ):
        raise ValueError("open-prose training phase differs")
    entries = manifest["groups"]["training"]["prose"]
    for entry in entries:
        if (
            entry["domain"] != "prose"
            or len(entry["turns"]) != 8
            or len(entry["task_ids"]) != 8
            or v9_collect.sha(args.workloads / entry["file"])
            != entry["sha256"]
        ):
            raise ValueError("open-prose training prompt changed")
    return entries


def collect(args):
    entries = freeze(args)
    if not 0 <= args.start < args.stop <= len(entries):
        raise ValueError("open-prose training range differs")
    args.out.mkdir(exist_ok=True)
    for index in range(args.start, args.stop):
        entry = entries[index]
        shift = index % len(ARMS)
        order = list(ARMS[shift:] + ARMS[:shift])
        if index % 2:
            order.reverse()
        for k, kind in order:
            row = (
                v11_collect.run_random(args, entry, index)
                if kind == "random-k"
                else v9_collect.run_one(args, entry, index, k, kind)
            )
            target = args.out / f"{row['name']}.result.json"
            if target.exists():
                if json.loads(target.read_text()) != row:
                    raise ValueError("resumed open-prose result changed")
            else:
                v9_collect.save(target, row)
            print(json.dumps({
                "session": row["name"],
                "domain": "prose",
                "status": "training-native-complete",
            }), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("binary", "model", "workloads", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=16)
    args = parser.parse_args()
    for name in ("binary", "model", "workloads", "out"):
        setattr(args, name, getattr(args, name).resolve())
    collect(args)


if __name__ == "__main__":
    main()
