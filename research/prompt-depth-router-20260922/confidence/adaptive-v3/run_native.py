"""Run the frozen Qwen native-session study on one non-production GPU."""

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from costs import cost_table
from learn import loop_candidate, read_costs
from oracle import read_sessions, report


MODEL_SHA = "1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a"
FORMAT = re.compile(r"^```(?:python|py)\n(?P<code>[\s\S]+?)\n```\s*$")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def tsv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def validate(root, entry, label):
    rows = tsv(root / "turns.tsv")
    if len(rows) != 8:
        raise ValueError(f"{label}: incomplete eight-turn native session")
    output = []
    for index, (turn, expected) in enumerate(zip(rows, entry["turns"]), 1):
        if (
            int(turn["turn"]) != index
            or (index > 1 and (
                turn["resumed"] != "true" or int(turn["cached_tokens"]) <= 0
                or int(turn["new_input_tokens"]) <= 0
                or int(turn["checkpoint_tokens"]) <= int(turn["cached_tokens"])
            ))
        ):
            raise ValueError(f"{label}: native KV continuation failed at turn {index}")
        user = root / f"turn-{index}.user.txt"
        if sha(user) != expected["user_sha256"]:
            raise ValueError(f"{label}: user prompt differs at turn {index}")
        ids = [int(value) for value in (
            root / f"turn-{index}.output.ids"
        ).read_text().split()]
        answer = (root / f"turn-{index}.answer.txt").read_text().strip()
        match = FORMAT.fullmatch(answer)
        parsed = False
        if match is not None:
            try:
                tree = ast.parse(match["code"])
                parsed = (
                    len(tree.body) == 1
                    and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef))
                    and tree.body[0].name == expected["function"]
                )
            except SyntaxError:
                pass
        output.append({
            "turn": index, "output_tokens": len(ids),
            "elapsed_s": float(turn["elapsed_s"]),
            "cached_tokens": int(turn["cached_tokens"]),
            "new_input_tokens": int(turn["new_input_tokens"]),
            "checkpoint_tokens": int(turn["checkpoint_tokens"]),
            "finish_reason": turn["finish_reason"],
            "loop": loop_candidate(ids),
            "fenced_parseable_function": parsed,
        })
    return output


def run_one(args, entry, group, index, arm, extra=()):
    label = f"{group}-{index}-{arm}"
    root = args.out / label
    if root.exists():
        raise ValueError(f"{label}: refusing to reuse a native output directory")
    command = [
        str(args.binary), str(args.model), "embedded",
        str(args.workloads / entry["file"]), str(root), arm,
        str(entry["seed"]), "8192", "65536", "0.7", *extra,
    ]
    environment = os.environ.copy()
    environment.update({
        "MEMRA_SPEC_ADAPT": "0",
        "MEMRA_SPEC_ADAPT_FLOOR": "1",
        "MEMRA_SPEC_CAPMAX": "7",
        "MEMRA_SPEC_PMIN": "0",
        "MEMRA_SPEC_PMIN0": "0",
        "MEMRA_SPEC_PMIN_INROUND": "0",
        "MEMRA_SPEC_STATS": "1",
    })
    if environment.get("MEMRA_CAPTURE_DIR"):
        raise ValueError("research pod cannot have capture enabled")
    command_record = {
        "argv": command,
        "memra_settings": {
            key: value for key, value in environment.items()
            if key.startswith("MEMRA_")
        },
        "binary_sha256": args.binary_sha,
        "model_sha256": MODEL_SHA,
        "workload_sha256": entry["sha256"],
    }
    telemetry_command = [
        "nvidia-smi", "--query-gpu=timestamp,uuid,name,memory.used,"
        "utilization.gpu,power.draw,clocks.sm",
        "--format=csv", "-l", "1",
    ]
    started = time.monotonic()
    with (args.out / f"{label}.stdout.log").open("x") as stdout, (
        args.out / f"{label}.stderr.log"
    ).open("x") as stderr, (args.out / f"{label}.gpu.csv").open("x") as gpu:
        telemetry = subprocess.Popen(telemetry_command, stdout=gpu, stderr=subprocess.DEVNULL)
        try:
            result = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr)
        finally:
            telemetry.terminate()
            telemetry.wait(timeout=15)
    if not root.exists():
        root.mkdir()
    (args.out / f"{label}.stdout.log").rename(root / "stdout.log")
    (args.out / f"{label}.stderr.log").rename(root / "stderr.log")
    (args.out / f"{label}.gpu.csv").rename(root / "gpu.csv")
    save(root / "command.json", command_record)
    save(root / "exit.json", {
        "returncode": result.returncode,
        "wall_s": time.monotonic() - started,
    })
    if result.returncode:
        raise RuntimeError(f"{label}: native program failed; inspect stderr.log")
    if "full_vocab=248320 draft_vocab=248320 mtp=embedded" not in (
        root / "stderr.log"
    ).read_text():
        raise ValueError(f"{label}: full MTP head path was not logged")
    save(root / "audit.json", validate(root, entry, label))
    return root


def check_inputs(args):
    if sha(args.model) != MODEL_SHA:
        raise ValueError("Qwen checkpoint differs")
    args.binary_sha = sha(args.binary)
    manifest = json.loads((args.workloads / "manifest.json").read_text())
    if manifest.get("schema") != 1:
        raise ValueError("another frozen workload schema")
    for group in ("qualification", "calibration", "heldout"):
        for entry in manifest["groups"][group]:
            if sha(args.workloads / entry["file"]) != entry["sha256"]:
                raise ValueError("frozen workload text differs")
    return manifest


def qualify(args, manifest):
    entry = manifest["groups"]["qualification"][0]
    root = run_one(args, entry, "qualification", 0, "trace-c3")
    audit = json.loads((root / "audit.json").read_text())
    if any(row["loop"] or not row["fenced_parseable_function"] for row in audit):
        raise ValueError("frozen C=0 code qualifier failed; stop this protocol version")
    return {"status": "qualified", "session": root.name}


def calibrate(args, manifest):
    arms = {1: [], 2: [], 3: []}
    for index, entry in enumerate(manifest["groups"]["calibration"]):
        for k in (list((1, 2, 3))[index:] + list((1, 2, 3))[:index]):
            arms[k].append(run_one(args, entry, "calibration", index, f"trace-c{k}"))
    costs = cost_table(arms, args.source_sha, MODEL_SHA)
    save(args.out / "costs.json", costs)
    sessions, sources, excluded = read_sessions(arms[3])
    oracle = report(sessions, read_costs(args.out / "costs.json"), sources, excluded)
    oracle["costs_sha256"] = sha(args.out / "costs.json")
    save(args.out / "oracle.json", oracle)
    return oracle


def heldout(args, manifest):
    oracle = json.loads((args.out / "oracle.json").read_text())
    cutoffs = oracle["best_calibrated_fixed"]["cutoffs"]
    labels = ("learn-c3", "monitor-c3", "fixed-c3", "fixed:3", "fixed:2")
    for index, entry in enumerate(manifest["groups"]["heldout"]):
        ordered = list(labels[index % len(labels):] + labels[:index % len(labels)])
        if index % 2:
            ordered.reverse()
        for arm in ordered:
            extra = ()
            if arm in ("learn-c3", "monitor-c3"):
                extra = (
                    f"confidence-policy={args.policy}",
                    f"confidence-costs={args.out / 'costs.json'}",
                )
            elif arm == "fixed-c3":
                extra = (f"confidence-fixed={cutoffs[0]},{cutoffs[1]}",)
            run_one(args, entry, "heldout", index, arm, extra)
    return {"status": "heldout-complete", "arms": list(labels)}


def main():
    parser = argparse.ArgumentParser()
    for name in ("binary", "model", "workloads", "out", "policy"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--phase", choices=("qualification", "calibration", "heldout"), required=True)
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.model = args.model.resolve()
    args.workloads = args.workloads.resolve()
    args.out = args.out.resolve()
    args.policy = args.policy.resolve()
    args.out.mkdir(exist_ok=True)
    manifest = check_inputs(args)
    if args.phase == "qualification":
        result = qualify(args, manifest)
    elif args.phase == "calibration":
        result = calibrate(args, manifest)
    else:
        result = heldout(args, manifest)
    save(args.out / f"{args.phase}-result.json", result)
    print(json.dumps({"phase": args.phase, "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
