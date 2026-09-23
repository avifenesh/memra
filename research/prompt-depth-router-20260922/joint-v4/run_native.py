"""Collect pinned Qwen eight-turn E2E and randomized-depth development records."""

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


LANE = Path(__file__).resolve().parent.parent
MODEL_SHA = "1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a"
FORMAT = re.compile(r"^```(?:python|py)\n(?P<code>[\s\S]+?)\n```\s*$")
TRAIN_ARMS = (
    ("k3-c0", "trace-c3", 3, None),
    ("k2-c0", "trace-c2", 2, None),
    ("k4-c0", "fixed:4", 4, None),
    ("k4-d3-c0", "trace-c3", 4, None),
    ("k3-c-cal", "fixed-c3", 3, "0.434978,0.850344"),
    ("k3-c-first", "fixed-c3", 3, "0.434978,0"),
    ("k3-c-second", "fixed-c3", 3, "0,0.850344"),
    ("k4-random-d", "explore-d", 4, None),
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def table(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def loop_candidate(ids):
    for end in sorted({len(ids), *range(512, len(ids) + 1, 256)}):
        for period in range(1, 129):
            repeated = max(4, (256 + period - 1) // period)
            size = repeated * period
            if end >= size and ids[end-size:end] == ids[end-period:end] * repeated:
                return {"end_token": end, "period": period, "repeated_tokens": size}
    return None


def validate(root, entry, arm):
    turns = table(root / "turns.tsv")
    if len(turns) != 8:
        raise ValueError("native session is not eight turns")
    audit = []
    for index, (row, expected) in enumerate(zip(turns, entry["turns"]), 1):
        if int(row["turn"]) != index or (index > 1 and (
            row["resumed"] != "true"
            or int(row["cached_tokens"]) <= 0
            or int(row["new_input_tokens"]) <= 0
            or int(row["checkpoint_tokens"]) <= int(row["cached_tokens"])
        )):
            raise ValueError(f"turn {index} lacks native KV continuation")
        if sha(root / f"turn-{index}.user.txt") != expected["user_sha256"]:
            raise ValueError(f"turn {index} user prompt differs")
        ids = [int(value) for value in
               (root / f"turn-{index}.output.ids").read_text().split()]
        if len(ids) != int(row["output_tokens"]):
            raise ValueError(f"turn {index} output token count differs")
        answer = (root / f"turn-{index}.answer.txt").read_text().strip()
        match = FORMAT.fullmatch(answer)
        parsed = False
        if match:
            try:
                body = ast.parse(match["code"]).body
                parsed = len(body) == 1 and isinstance(
                    body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and body[0].name == expected["function"]
            except SyntaxError:
                pass
        audit.append({
            "turn": index,
            "output_tokens": len(ids),
            "elapsed_s": float(row["elapsed_s"]),
            "cached_tokens": int(row["cached_tokens"]),
            "new_input_tokens": int(row["new_input_tokens"]),
            "checkpoint_tokens": int(row["checkpoint_tokens"]),
            "finish_reason": row["finish_reason"],
            "loop": loop_candidate(ids),
            "fenced_parseable_function": parsed,
        })
    if arm == "explore-d":
        counts = Counter()
        depth = table(root / "rounds.tsv")
        for turn in range(1, 9):
            rounds = table(root / f"turn-{turn}.confidence.tsv")
            current = [row for row in depth if int(row["turn"]) == turn]
            if len(rounds) != len(current):
                raise ValueError("randomized confidence and depth inventory differ")
            for confidence, observed in zip(rounds, current):
                chosen = int(observed["draft_depth"]) - 1
                if (
                    chosen not in (1, 2, 3, 4)
                    or int(confidence["drafted"]) != chosen
                    or int(confidence["round"]) != int(observed["round"])
                    or confidence["eligible"] != observed["eligible_for_learning"]
                ):
                    raise ValueError("randomized chosen D differs from offered D")
                if confidence["eligible"] == "true":
                    counts[chosen] += 1
        if min(counts.values(), default=0) < 25 or len(counts) != 4:
            raise ValueError("randomized D lacks per-action eligible exposure")
        save(root / "exposure.json", dict(sorted(counts.items())))
    return audit


def check_inputs(args):
    if sha(args.model) != MODEL_SHA:
        raise ValueError("checkpoint differs from the pinned full-head Qwen")
    args.binary_sha = sha(args.binary)
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_sha):
        raise ValueError("runtime source digest is not a SHA-256")
    development = json.loads((args.development / "manifest.json").read_text())
    fresh = json.loads((args.fresh / "manifest.json").read_text())
    seeds = json.loads((LANE / "joint-v4/exploration-seeds.json").read_text())
    if (
        development["schema"] != 1 or fresh["schema"] != 1
        or seeds["v3_workloads_sha256"] != sha(args.development / "manifest.json")
        or seeds["v4_fresh_heldout_sha256"] != sha(args.fresh / "manifest.json")
    ):
        raise ValueError("development/fresh manifest or exploration seeds differ")
    for manifest, path in ((development, args.development), (fresh, args.fresh)):
        for group in manifest["groups"].values():
            for item in group:
                if sha(path / item["file"]) != item["sha256"]:
                    raise ValueError("frozen workload text differs")
    return development, fresh, seeds


def run_one(args, item, workloads, label, variant, arm, cap, cutoffs=None,
            explore_seed=None):
    root = args.out / f"{label}-{variant}"
    if root.exists():
        raise ValueError(f"refusing to reuse {root}")
    command = [
        str(args.binary), str(args.model), "embedded",
        str(workloads / item["file"]), str(root), arm,
        str(item["seed"]), "8192", "65536", "0.7", f"cap={cap}",
    ]
    if arm == "explore-d":
        command.append(f"explore-seed={explore_seed}")
    if arm == "fixed-c3":
        command.append(f"confidence-fixed={cutoffs}")
    environment = os.environ.copy()
    environment.update({
        "MEMRA_SPEC_ADAPT": "0", "MEMRA_SPEC_ADAPT_FLOOR": "1",
        "MEMRA_SPEC_CAPMAX": "7", "MEMRA_SPEC_PMIN": "0",
        "MEMRA_SPEC_PMIN0": "0", "MEMRA_SPEC_PMIN_INROUND": "0",
        "MEMRA_SPEC_STATS": "1",
    })
    if environment.get("MEMRA_CAPTURE_DIR"):
        raise ValueError("research pod has capture enabled")
    save(args.out / f"{root.name}.command.json", {
        "argv": command,
        "binary_sha256": args.binary_sha,
        "source_sha256": args.source_sha,
        "model_sha256": MODEL_SHA,
        "workload_sha256": item["sha256"],
        "variant": variant,
        "cap": cap,
        "cutoffs": cutoffs,
        "explore_seed": explore_seed,
        "memra_settings": {
            key: value for key, value in environment.items()
            if key.startswith("MEMRA_")
        },
    })
    started = time.monotonic()
    with (args.out / f"{root.name}.stdout.log").open("x") as stdout, (
        args.out / f"{root.name}.stderr.log"
    ).open("x") as stderr:
        result = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr)
    save(args.out / f"{root.name}.exit.json", {
        "returncode": result.returncode,
        "wall_s": time.monotonic() - started,
    })
    if result.returncode:
        raise RuntimeError(f"{root.name} native run failed; inspect stderr log")
    if "full_vocab=248320 draft_vocab=248320 mtp=embedded" not in (
        args.out / f"{root.name}.stderr.log"
    ).read_text():
        raise ValueError("native run did not log full MTP head engagement")
    audit = validate(root, item, arm)
    save(root / "audit.json", audit)
    return {
        "name": root.name,
        "variant": variant,
        "arm": arm,
        "tokens": sum(row["output_tokens"] for row in audit),
        "seconds": sum(row["elapsed_s"] for row in audit),
        "format": sum(bool(row["fenced_parseable_function"]) for row in audit),
        "loops": sum(bool(row["loop"]) for row in audit),
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("binary", "model", "development", "fresh", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--phase", choices=("qualification", "training"), required=True)
    args = parser.parse_args()
    for name in ("binary", "model", "development", "fresh", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(exist_ok=True)
    development, fresh, seeds = check_inputs(args)
    if args.phase == "qualification":
        item = fresh["groups"]["qualification"][0]
        records = [run_one(
            args, item, args.fresh, "qualification-0", "k3-c0", "trace-c3", 3,
        )]
        if records[0]["format"] != 8 or records[0]["loops"]:
            raise ValueError("fresh full-head Qwen code qualifier failed")
    else:
        records = []
        train = seeds["training"]
        topics = (development["groups"]["calibration"]
                  + development["groups"]["heldout"][:3])
        for index, (schedule, item) in enumerate(zip(train, topics)):
            if (
                schedule["workload_sha256"] != item["sha256"]
                or schedule["sampling_seed"] != item["seed"]
            ):
                raise ValueError("frozen randomized assignment differs from development split")
            order = list(TRAIN_ARMS[index % len(TRAIN_ARMS):]
                         + TRAIN_ARMS[:index % len(TRAIN_ARMS)])
            if index % 2:
                order.reverse()
            for variant, arm, cap, cutoffs in order:
                records.append(run_one(
                    args, item, args.development, f"training-{index}",
                    variant, arm, cap, cutoffs,
                    schedule.get("explore_seed") if arm == "explore-d" else None,
                ))
    save(args.out / f"{args.phase}-summary.json", {
        "schema": 1, "phase": args.phase, "records": records,
    })
    print(json.dumps({"phase": args.phase, "sessions": len(records)}))


if __name__ == "__main__":
    main()
