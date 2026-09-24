"""Run frozen native C/K/D candidates on disjoint validation or final sessions."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import time

from collect import (
    BINARY_SHA256, MODEL_SHA256, loop_candidate, save, sha, table,
)


def model_hashes(extra):
    options = dict(item.split("=", 1) for item in extra)
    files = [
        Path(options[key]) for key in (
            "topk-model", "depth-model", "confidence-model"
        ) if key in options
    ]
    if "joint-model-dir" in options:
        directory = Path(options["joint-model-dir"])
        for k in (3, 10, 20):
            for kind in ("depth", "confidence"):
                variant = options[f"{kind}-variant"]
                files.append(directory / f"topk{k}/{kind}-{variant}.tsv")
    return {str(path): sha(path) for path in files}


def verify(root, entry, spec):
    turns = table(root / "turns.tsv")
    rounds = table(root / "rounds.tsv")
    if len(turns) != 8 or not rounds:
        raise ValueError(f"native evaluation session incomplete: {root.name}")
    loops = 0
    for number, row in enumerate(turns, 1):
        k = int(row["draft_top_k"])
        if (
            int(row["turn"]) != number
            or int(row["sampler_top_k"]) != 20
            or k not in (3, 10, 20)
            or (spec["arm"] not in ("learn-topk", "joint-ckd") and k != spec["k"])
            or float(row["elapsed_s"]) <= 0
            or (number > 1 and (
                row["resumed"] != "true"
                or int(row["cached_tokens"]) <= 0
                or int(row["new_input_tokens"]) <= 0
                or int(row["checkpoint_tokens"]) <= int(row["cached_tokens"])
            ))
        ):
            raise ValueError(f"target/K/KV receipt differs: {root.name}/{number}")
        if sha(root / f"turn-{number}.user.txt") != entry["turns"][number - 1]["user_sha256"]:
            raise ValueError(f"prompt differs: {root.name}/{number}")
        ids = [
            int(value) for value in
            (root / f"turn-{number}.output.ids").read_text().split()
        ]
        if len(ids) != int(row["output_tokens"]):
            raise ValueError(f"output count differs: {root.name}/{number}")
        loops += loop_candidate(ids)
    actions = Counter(int(row["draft_top_k"]) for row in turns)
    depths = Counter(
        int(row["draft_depth"]) for row in rounds
        if row["eligible_for_learning"] == "true"
    )
    return {
        "name": root.name,
        "variant": spec["label"],
        "task_ids": entry["task_ids"],
        "tokens": sum(int(row["output_tokens"]) for row in turns),
        "seconds": sum(float(row["elapsed_s"]) for row in turns),
        "loops": loops,
        "k_actions": dict(sorted(actions.items())),
        "d_actions": dict(sorted(depths.items())),
        "c_decisions": sum(int(row["confidence_decisions"]) for row in turns),
        "c_stops": sum(int(row["confidence_stops"]) for row in turns),
        "k_model_s": sum(int(row["k_model_ns"]) for row in turns) / 1e9,
        "cd_model_s": sum(int(row["depth_policy_ns"]) for row in turns) / 1e9,
        "cached_later_turns": sum(
            int(row["cached_tokens"]) > 0 for row in turns[1:]
        ),
        "finished": [row["finish_reason"] for row in turns],
    }


def run_one(args, entry, index, spec):
    root = args.out / f"{args.phase}-{index}-{spec['label']}"
    extra = spec.get("extra", [])
    command = [
        str(args.binary), str(args.model), "embedded",
        str(args.workloads / entry["file"]), str(root), spec["arm"],
        str(entry["seed"]), "4096", "65536", "1.0",
        f"cap={spec['cap']}", "sampler-top-k=20",
        f"draft-top-k={spec['k']}", *extra,
    ]
    command_record = {
        "argv": command,
        "binary_sha256": BINARY_SHA256,
        "model_sha256": MODEL_SHA256,
        "workload_sha256": entry["sha256"],
        "policy_sha256": model_hashes(extra),
    }
    command_path = args.out / f"{root.name}.command.json"
    exit_path = args.out / f"{root.name}.exit.json"
    if root.exists():
        if (
            not command_path.exists()
            or json.loads(command_path.read_text()) != command_record
            or not exit_path.exists()
            or json.loads(exit_path.read_text())["returncode"] != 0
        ):
            raise ValueError(f"changed or incomplete native arm: {root.name}")
        return verify(root, entry, spec)
    save(command_path, command_record)
    environment = os.environ.copy()
    if environment.get("MEMRA_CAPTURE_DIR"):
        raise ValueError("research instance cannot capture customer content")
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
    save(exit_path, {
        "returncode": result.returncode,
        "wall_s": time.monotonic() - started,
    })
    if result.returncode:
        raise RuntimeError(f"native arm failed: {root.name}")
    engagement = (args.out / f"{root.name}.stderr.log").read_text()
    if (
        "full_vocab=248320 draft_vocab=248320 mtp=embedded" not in engagement
        or "target_top_k=20" not in engagement
    ):
        raise ValueError(f"full-head engagement missing: {root.name}")
    return verify(root, entry, spec)


def main():
    parser = argparse.ArgumentParser()
    for name in ("binary", "model", "workloads", "out", "arms"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("qualification", "validation", "heldout"), required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    args = parser.parse_args()
    for name in ("binary", "model", "workloads", "out", "arms"):
        setattr(args, name, getattr(args, name).resolve())
    if sha(args.binary) != BINARY_SHA256 or sha(args.model) != MODEL_SHA256:
        raise ValueError("binary or Qwen artifact differs")
    manifest = json.loads((args.workloads / "manifest.json").read_text())
    entries = manifest["groups"][args.phase]
    arms = json.loads(args.arms.read_text())
    if arms["schema"] != 1 or arms["phase"] != args.phase:
        raise ValueError("arm manifest belongs to another evaluation phase")
    if args.phase == "heldout":
        selected_path = args.arms.with_name("selected.json")
        selected = json.loads(selected_path.read_text())
        if (
            selected["phase"] != "validation-selected"
            or arms["selected_from_validation"] != sha(selected_path)
            or sorted(spec["label"] for spec in arms["arms"])
            != selected["final_arms"]
        ):
            raise ValueError("final arms differ from frozen validation selection")
    specs = arms["arms"]
    if len({spec["label"] for spec in specs}) != len(specs):
        raise ValueError("duplicate arm label")
    for spec in specs:
        if (
            spec["k"] not in (3, 10, 20)
            or spec["cap"] not in (3, 4)
            or any("=" not in item for item in spec.get("extra", []))
        ):
            raise ValueError("invalid K/C/D arm")
        model_hashes(spec.get("extra", []))
    if args.stop is None:
        args.stop = len(entries)
    if not 0 <= args.start < args.stop <= len(entries):
        raise ValueError("invalid evaluation conversation range")
    args.out.mkdir(exist_ok=True)
    for index in range(args.start, args.stop):
        entry = entries[index]
        if sha(args.workloads / entry["file"]) != entry["sha256"]:
            raise ValueError("frozen evaluation conversation differs")
        order = specs[index % len(specs):] + specs[:index % len(specs)]
        if index % 2:
            order.reverse()
        for spec in order:
            row = run_one(args, entry, index, spec)
            result_path = args.out / f"{row['name']}.result.json"
            if result_path.exists():
                if json.loads(result_path.read_text()) != row:
                    raise ValueError("resumed native evaluation result changed")
            else:
                save(result_path, row)
            print(json.dumps({
                "session": row["name"],
                "tok_s": row["tokens"] / row["seconds"],
                "c_decisions": row["c_decisions"],
                "c_stops": row["c_stops"],
            }), flush=True)


if __name__ == "__main__":
    main()
