"""Collect fresh randomized C/K/D observations on disjoint non-code tasks."""

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


V9 = Path(__file__).resolve().parent.parent / "joint-v9"
sys.path.insert(0, str(V9))
import collect as v9_collect
import eval as v9_eval


WORKLOAD_SHA = "655223c8e4ca61fab179f7a4708b152ec35cf351209ba5210c10956080075a58"
DOMAINS = ("ifeval", "gsm8k")
ARMS = tuple(
    (k, label)
    for k in v9_collect.TOP_K
    for label in ("fixed-d3", "explore-d")
)


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def draft_schedule(index):
    choices = v9_collect.TOP_K
    schedule = list(choices) * 2 + [
        choices[index % 3], choices[(index + 1) % 3],
    ]
    random.Random(25197000 + index).shuffle(schedule)
    if len(schedule) != 8 or set(schedule) != set(choices):
        raise ValueError("random K schedule differs")
    return schedule


def verify_random(root, entry, index):
    schedule = draft_schedule(index)
    observed = v9_eval.verify(
        root, entry,
        {"label": "random-k", "arm": "learn-topk", "k": 20},
    )
    turns = v9_collect.table(root / "turns.tsv")
    if [int(row["draft_top_k"]) for row in turns] != schedule:
        raise ValueError(f"random K assignment differs: {root.name}")
    if observed["c_decisions"] or observed["c_stops"]:
        raise ValueError(f"random K arm unexpectedly applied C: {root.name}")
    return observed


def run_random(args, entry, index):
    schedule = draft_schedule(index)
    root = args.out / f"training-{index}-random-k"
    command = [
        str(args.binary), str(args.model), "embedded",
        str(args.workloads / entry["file"]), str(root), "fixed:3",
        str(entry["seed"]), "4096", "65536", "1.0", "cap=3",
        "sampler-top-k=20", "draft-top-k=20",
        "draft-schedule=" + ",".join(map(str, schedule)),
    ]
    record = {
        "argv": command,
        "model_sha256": v9_collect.MODEL_SHA256,
        "binary_sha256": v9_collect.BINARY_SHA256,
        "workload_sha256": entry["sha256"],
        "target_top_k": 20,
        "draft_schedule": schedule,
        "assignment": "randomized-turn",
    }
    command_path = args.out / f"{root.name}.command.json"
    exit_path = args.out / f"{root.name}.exit.json"
    if root.exists():
        if (
            not command_path.exists()
            or json.loads(command_path.read_text()) != record
            or not exit_path.exists()
            or json.loads(exit_path.read_text())["returncode"] != 0
        ):
            raise ValueError(f"incomplete random K arm: {root.name}")
        return verify_random(root, entry, index)
    v9_collect.save(command_path, record)
    environment = os.environ.copy()
    if environment.get("MEMRA_CAPTURE_DIR"):
        raise ValueError("research instance cannot hold customer capture")
    environment.update({
        "MEMRA_SPEC_ADAPT": "0",
        "MEMRA_SPEC_ADAPT_FLOOR": "1",
        "MEMRA_SPEC_CAPMAX": "7",
        "MEMRA_SPEC_PMIN": "0",
        "MEMRA_SPEC_PMIN0": "0",
        "MEMRA_SPEC_PMIN_INROUND": "0",
        "MEMRA_SPEC_STATS": "1",
    })
    started = time.monotonic()
    with (args.out / f"{root.name}.stdout.log").open("x") as stdout, (
        args.out / f"{root.name}.stderr.log"
    ).open("x") as stderr:
        result = subprocess.run(
            command, env=environment, stdout=stdout, stderr=stderr
        )
    v9_collect.save(exit_path, {
        "returncode": result.returncode,
        "wall_s": time.monotonic() - started,
    })
    if result.returncode:
        raise RuntimeError(f"random K native arm failed: {root.name}")
    engagement = (args.out / f"{root.name}.stderr.log").read_text()
    if (
        "full_vocab=248320 draft_vocab=248320 mtp=embedded"
        not in engagement
        or "target_top_k=20" not in engagement
    ):
        raise ValueError(f"random K full-head engagement missing: {root.name}")
    return verify_random(root, entry, index)


def inventory(workloads):
    path = workloads / "manifest.json"
    if v9_collect.sha(path) != WORKLOAD_SHA:
        raise ValueError("v11 frozen non-code manifest differs")
    manifest = json.loads(path.read_text())
    if manifest["schema"] != 1 or manifest["splits"]["training"] != 16:
        raise ValueError("v11 training split differs")
    for domain in DOMAINS:
        for phase, count in manifest["splits"].items():
            entries = manifest["groups"][phase][domain]
            if len(entries) != count:
                raise ValueError("non-code phase count differs")
            for entry in entries:
                if (
                    entry["domain"] != domain
                    or len(entry["task_ids"]) != 8
                    or v9_collect.sha(workloads / entry["file"]) != entry["sha256"]
                ):
                    raise ValueError("frozen non-code conversation differs")
    return manifest


def collect(args):
    if (
        v9_collect.sha(args.binary) != v9_collect.BINARY_SHA256
        or v9_collect.sha(args.model) != v9_collect.MODEL_SHA256
    ):
        raise ValueError("native binary or Qwen artifact differs")
    manifest = inventory(args.workloads)
    if not 0 <= args.start < args.stop <= 32:
        raise ValueError("invalid v11 training conversation range")
    args.out.mkdir(exist_ok=True)
    for global_index in range(args.start, args.stop):
        domain = DOMAINS[global_index // 16]
        index = global_index % 16
        entry = manifest["groups"]["training"][domain][index]
        order = list(ARMS) + [(None, "random-k")]
        shift = global_index % len(order)
        order = order[shift:] + order[:shift]
        if global_index % 2:
            order.reverse()
        for k, label in order:
            row = (
                run_random(args, entry, global_index)
                if label == "random-k"
                else v9_collect.run_one(args, entry, global_index, k, label)
            )
            target = args.out / f"{row['name']}.result.json"
            if target.exists():
                if json.loads(target.read_text()) != canonical(row):
                    raise ValueError("resumed randomized result changed")
            else:
                v9_collect.save(target, row)
            progress = {
                "domain": domain,
                "conversation": index,
                "session": row["name"],
                "tok_s": row["tokens"] / row["seconds"],
            }
            progress["k_actions" if label == "random-k" else "d_exposure"] = (
                row["k_actions"] if label == "random-k"
                else row["d_exposure"]
            )
            print(json.dumps(progress), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("binary", "model", "workloads", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=32)
    args = parser.parse_args()
    for name in ("binary", "model", "workloads", "out"):
        setattr(args, name, getattr(args, name).resolve())
    collect(args)


if __name__ == "__main__":
    main()
