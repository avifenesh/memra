"""Collect balanced native K/D/C training sessions on a research GPU."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


MODEL_SHA256 = "1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a"
BINARY_SHA256 = "84b04b6ccccc3b64992377cef0677926cb932c7a98f063aff84d1f1f65224d7f"
TOP_K = (3, 10, 20)


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def table(path):
    with path.open() as source:
        return list(csv.DictReader(source, delimiter="\t"))


def loop_candidate(ids):
    for end in sorted({len(ids), *range(512, len(ids) + 1, 256)}):
        for period in range(1, 129):
            repeated = max(4, (256 + period - 1) // period)
            size = repeated * period
            if end >= size and ids[end - size:end] == ids[end - period:end] * repeated:
                return True
    return False


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def verify_session(root, entry, label, k, phase):
    turns = table(root / "turns.tsv")
    if len(turns) != 8:
        raise ValueError(f"incomplete native conversation: {root}")
    rounds = table(root / "rounds.tsv")
    if not rounds:
        raise ValueError(f"no native draft rounds: {root}")
    loops = 0
    for number, row in enumerate(turns, 1):
        if (
            int(row["turn"]) != number
            or int(row["sampler_top_k"]) != 20
            or int(row["draft_top_k"]) != k
            or float(row["elapsed_s"]) <= 0
            or (number > 1 and (
                row["resumed"] != "true"
                or int(row["cached_tokens"]) <= 0
                or int(row["new_input_tokens"]) <= 0
                or int(row["checkpoint_tokens"]) <= int(row["cached_tokens"])
            ))
        ):
            raise ValueError(f"native target/K/KV receipt differs: {root}/{number}")
        for suffix in ("answer.txt", "output.ids", "user.txt"):
            if not (root / f"turn-{number}.{suffix}").is_file():
                raise ValueError(f"missing native answer: {root}/{number}/{suffix}")
        if sha(root / f"turn-{number}.user.txt") != entry["turns"][number - 1]["user_sha256"]:
            raise ValueError(f"native user prompt differs: {root}/{number}")
        ids = [
            int(value) for value in
            (root / f"turn-{number}.output.ids").read_text().split()
        ]
        if len(ids) != int(row["output_tokens"]):
            raise ValueError(f"native output token count differs: {root}/{number}")
        loops += loop_candidate(ids)
    if phase == "training" and label == "explore-d":
        for number in range(1, 9):
            confidence = table(root / f"turn-{number}.confidence.tsv")
            expected = [
                row for row in rounds if int(row["turn"]) == number
            ]
            if len(confidence) != len(expected):
                raise ValueError(f"C trace and round counts differ: {root}/{number}")
            for offered, round_row in zip(confidence, expected):
                if (
                    int(offered["round"]) != int(round_row["round"])
                    or int(offered["drafted"]) != int(round_row["draft_depth"])
                    or int(offered["elapsed_ns"]) != int(round_row["elapsed_ns"])
                ):
                    raise ValueError(f"C trace and D receipt differ: {root}/{number}")
    return {
        "name": root.name,
        "variant": f"k{k}-{label}",
        "phase": phase,
        "tokens": sum(int(row["output_tokens"]) for row in turns),
        "seconds": sum(float(row["elapsed_s"]) for row in turns),
        "cached_later_turns": sum(
            int(row["cached_tokens"]) > 0 for row in turns[1:]
        ),
        "k": k,
        "d_exposure": {
            str(d): sum(
                int(row["draft_depth"]) == d
                and row["eligible_for_learning"] == "true"
                for row in rounds
            )
            for d in range(1, 5)
        },
        "c_decisions": sum(int(row["confidence_decisions"]) for row in turns),
        "c_stops": sum(int(row["confidence_stops"]) for row in turns),
        "finished": [row["finish_reason"] for row in turns],
        "loops": loops,
        "task_ids": entry["task_ids"],
    }


def run_one(args, entry, index, k, label):
    root = args.out / f"training-{index}-k{k}-{label}"
    arm = "fixed:3" if label == "fixed-d3" else "explore-d"
    extra = [] if label == "fixed-d3" else [
        f"explore-seed={20926000 + index * 31 + k}"
    ]
    command = [
        str(args.binary), str(args.model), "embedded",
        str(args.workloads / entry["file"]), str(root), arm,
        str(entry["seed"]), "4096", "65536", "1.0", "cap=4" if extra else "cap=3",
        "sampler-top-k=20", f"draft-top-k={k}", *extra,
    ]
    command_path = args.out / f"{root.name}.command.json"
    exit_path = args.out / f"{root.name}.exit.json"
    command_record = {
        "argv": command, "model_sha256": MODEL_SHA256,
        "binary_sha256": BINARY_SHA256,
        "workload_sha256": entry["sha256"],
        "target_top_k": 20,
        "draft_top_k": k,
        "temperature": 1.0,
        "top_p": 0.95,
    }
    if root.exists():
        if (
            not command_path.exists()
            or json.loads(command_path.read_text()) != command_record
            or not exit_path.exists()
            or json.loads(exit_path.read_text())["returncode"] != 0
        ):
            raise ValueError(f"changed or incomplete existing run: {root}")
        return verify_session(root, entry, label, k, "training")
    save(command_path, command_record)
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
    save(exit_path, {
        "returncode": result.returncode,
        "wall_s": time.monotonic() - started,
    })
    if result.returncode:
        raise RuntimeError(f"native run failed: {root.name}")
    engagement = (args.out / f"{root.name}.stderr.log").read_text()
    if (
        "full_vocab=248320 draft_vocab=248320 mtp=embedded" not in engagement
        or "target_top_k=20" not in engagement
        or f"draft_top_k={k}" not in engagement
    ):
        raise ValueError(f"native full-head engagement missing: {root.name}")
    return verify_session(root, entry, label, k, "training")


def main():
    parser = argparse.ArgumentParser()
    for name in ("binary", "model", "workloads", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=24)
    args = parser.parse_args()
    for name in ("binary", "model", "workloads", "out"):
        setattr(args, name, getattr(args, name).resolve())
    if sha(args.binary) != BINARY_SHA256 or sha(args.model) != MODEL_SHA256:
        raise ValueError("native binary or Qwen checkpoint differs")
    manifest = json.loads((args.workloads / "manifest.json").read_text())
    entries = manifest["groups"]["training"]
    if not 0 <= args.start < args.stop <= len(entries) or len(entries) != 24:
        raise ValueError("invalid frozen training range")
    for entry in entries:
        if sha(args.workloads / entry["file"]) != entry["sha256"]:
            raise ValueError("frozen MBPP conversation differs")
    args.out.mkdir(exist_ok=True)
    for index in range(args.start, args.stop):
        entry = entries[index]
        arms = [(k, label) for k in TOP_K for label in ("fixed-d3", "explore-d")]
        arms = arms[index % len(arms):] + arms[:index % len(arms)]
        if index % 2:
            arms.reverse()
        for k, label in arms:
            row = run_one(args, entry, index, k, label)
            path = args.out / f"{row['name']}.result.json"
            if path.exists():
                if json.loads(path.read_text()) != row:
                    raise ValueError("resumed native result changed")
            else:
                save(path, row)
            print(json.dumps({
                "session": row["name"], "tok_s": row["tokens"] / row["seconds"],
                "d_exposure": row["d_exposure"],
            }), flush=True)


if __name__ == "__main__":
    main()
