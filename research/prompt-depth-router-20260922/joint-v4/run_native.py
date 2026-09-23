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
    ("k4-random-d", "explore-d", 4, None),
    ("k3-c0", "fixed:3", 3, None),
    ("k3-trace", "trace-c3", 3, None),
    ("k2-c0", "fixed:2", 2, None),
    ("k4-c0", "fixed:4", 4, None),
    ("k4-d3-c0", "fixed:3", 4, None),
    ("k3-c-cal", "fixed-c3", 3, "0.434978,0.850344"),
    ("k3-c-first", "fixed-c3", 3, "0.434978,0"),
    ("k3-c-second", "fixed-c3", 3, "0,0.850344"),
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
                chosen = int(observed["draft_depth"])
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
            explore_seed=None, depth_model=None):
    root = args.out / f"{label}-{variant}"
    command = [
        str(args.binary), str(args.model), "embedded",
        str(workloads / item["file"]), str(root), arm,
        str(item["seed"]), "8192", "65536", "0.7", f"cap={cap}",
    ]
    if arm == "explore-d":
        command.append(f"explore-seed={explore_seed}")
    if arm == "fixed-c3":
        command.append(f"confidence-fixed={cutoffs}")
    if depth_model is not None:
        command.append(f"depth-model={depth_model}")
    environment = os.environ.copy()
    environment.update({
        "MEMRA_SPEC_ADAPT": "1" if arm == "native" else "0",
        "MEMRA_SPEC_ADAPT_FLOOR": "1",
        "MEMRA_SPEC_CAPMAX": "7", "MEMRA_SPEC_PMIN": "0",
        "MEMRA_SPEC_PMIN0": "0", "MEMRA_SPEC_PMIN_INROUND": "0",
        "MEMRA_SPEC_STATS": "1",
    })
    if environment.get("MEMRA_CAPTURE_DIR"):
        raise ValueError("research pod has capture enabled")
    command_record = {
        "argv": command,
        "binary_sha256": args.binary_sha,
        "source_sha256": args.source_sha,
        "model_sha256": MODEL_SHA,
        "workload_sha256": item["sha256"],
        "variant": variant,
        "cap": cap,
        "cutoffs": cutoffs,
        "depth_model_sha256": sha(depth_model) if depth_model else None,
        "explore_seed": explore_seed,
        "memra_settings": {
            key: value for key, value in environment.items()
            if key.startswith("MEMRA_")
        },
    }
    command_path = args.out / f"{root.name}.command.json"
    exit_path = args.out / f"{root.name}.exit.json"
    if root.exists():
        previous = json.loads(command_path.read_text()) if command_path.exists() else {}
        previous.setdefault("depth_model_sha256", None)
        if (
            not command_path.exists()
            or previous != command_record
            or not exit_path.exists()
            or json.loads(exit_path.read_text())["returncode"] != 0
        ):
            raise ValueError(f"{root.name} is incomplete or has another pinned command")
        audit = validate(root, item, arm)
        audit_path = root / "audit.json"
        if audit_path.exists():
            if json.loads(audit_path.read_text()) != audit:
                raise ValueError(f"{root.name} saved audit differs")
        else:
            save(audit_path, audit)
        return record(root, variant, arm, audit)
    save(command_path, command_record)
    started = time.monotonic()
    with (args.out / f"{root.name}.stdout.log").open("x") as stdout, (
        args.out / f"{root.name}.stderr.log"
    ).open("x") as stderr:
        result = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr)
    save(exit_path, {
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
    return record(root, variant, arm, audit)


def record(root, variant, arm, audit):
    return {
        "name": root.name,
        "variant": variant,
        "arm": arm,
        "tokens": sum(row["output_tokens"] for row in audit),
        "seconds": sum(row["elapsed_s"] for row in audit),
        "format": sum(bool(row["fenced_parseable_function"]) for row in audit),
        "loops": sum(bool(row["loop"]) for row in audit),
    }


def model_paths(path):
    manifest = json.loads((path / "models.json").read_text())
    if len(manifest) != 3:
        raise ValueError("three nested model feature sets were not trained")
    result = {}
    for row in manifest:
        variant = row["variant"]
        model = path / f"depth-{variant}.tsv"
        if variant not in ("token", "history", "prior") or (
            variant in result or sha(model) != row["model_sha256"]
        ):
            raise ValueError("tiny model artifact differs from its frozen manifest")
        result[variant] = model
    return result


def selected_fixed(args):
    analysis = json.loads((args.out / "training-analysis.json").read_text())
    winner = analysis["strongest_quality_qualified_fixed"]
    controls = {name: (arm, cap, cutoffs) for name, arm, cap, cutoffs in TRAIN_ARMS}
    if winner not in controls or winner in ("k4-random-d", "k3-trace"):
        raise ValueError("training selected an unscored static control")
    return winner, controls[winner]


def scored_conversations(args, items, workloads, phase, variants):
    fixed_name, (fixed_arm, fixed_cap, fixed_cutoffs) = selected_fixed(args)
    models = model_paths(args.models)
    if variants is not None and (len(variants) != 1 or variants[0] not in models):
        raise ValueError("heldout needs exactly one frozen selected model")
    evaluated = list(models) if variants is None else list(variants)
    arms = [("k3-c0", "fixed:3", 3, None, None)]
    if fixed_name != "k3-c0":
        arms.append((fixed_name, fixed_arm, fixed_cap, fixed_cutoffs, None))
    arms.append(("native-adapt", "native", 4, None, None))
    for name in evaluated:
        arms.append((f"{name}-trained", "trained-d", 4, None, models[name]))
        arms.append((f"{name}-noop", "noop-d", 4, None, models[name]))
    records = []
    for index, item in enumerate(items):
        order = arms[index % len(arms):] + arms[:index % len(arms)]
        if index % 2:
            order.reverse()
        for label, arm, cap, cutoffs, depth_model in order:
            records.append(run_one(
                args, item, workloads, f"{phase}-{index}", label, arm, cap,
                cutoffs=cutoffs, depth_model=depth_model,
            ))
    return records


def select_model(args, records):
    ranked = []
    for variant in ("token", "history", "prior"):
        chosen = [row for row in records if row["variant"] == f"{variant}-trained"]
        noop = [row for row in records if row["variant"] == f"{variant}-noop"]
        if len(chosen) != 3 or len(noop) != 3:
            raise ValueError("model selection lacks matched conversations")
        tokens = sum(row["tokens"] for row in chosen)
        seconds = sum(row["seconds"] for row in chosen)
        ranked.append({
            "variant": variant,
            "tok_s": tokens / seconds,
            "tokens": tokens,
            "seconds": seconds,
            "format_pass": sum(row["format"] for row in chosen),
            "loops": sum(row["loops"] for row in chosen),
            "noop_tok_s": sum(row["tokens"] for row in noop)
            / sum(row["seconds"] for row in noop),
        })
    qualified = [row for row in ranked if row["format_pass"] == 24 and row["loops"] == 0]
    if not qualified:
        raise ValueError("no learned D variant passes selection format and loop gates")
    chosen = max(qualified, key=lambda row: row["tok_s"])
    model = model_paths(args.models)[chosen["variant"]]
    save(args.out / "selected-model.json", {
        "schema": 1,
        "source": "three held-back v3 development conversations",
        "variant": chosen["variant"],
        "model_sha256": sha(model),
        "selection_scores": ranked,
        "fixed_control": selected_fixed(args)[0],
    })
    return chosen


def main():
    parser = argparse.ArgumentParser()
    for name in ("binary", "model", "development", "fresh", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument(
        "--phase", choices=("qualification", "training", "selection", "heldout"),
        required=True,
    )
    args = parser.parse_args()
    for name in ("binary", "model", "development", "fresh", "out"):
        setattr(args, name, getattr(args, name).resolve())
    if args.models is not None:
        args.models = args.models.resolve()
    args.out.mkdir(exist_ok=True)
    development, fresh, seeds = check_inputs(args)
    if args.phase == "qualification":
        item = fresh["groups"]["qualification"][0]
        records = [run_one(
            args, item, args.fresh, "qualification-0", "k3-c0", "trace-c3", 3,
        )]
        if records[0]["format"] != 8 or records[0]["loops"]:
            raise ValueError("fresh full-head Qwen code qualifier failed")
    elif args.phase == "training":
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
    elif args.phase == "selection":
        if args.models is None:
            raise ValueError("model selection requires frozen trained model files")
        records = scored_conversations(
            args, development["groups"]["heldout"][3:],
            args.development, "selection", None,
        )
        select_model(args, records)
    else:
        if args.models is None:
            raise ValueError("heldout requires a selected frozen model")
        selected = json.loads((args.out / "selected-model.json").read_text())
        models = model_paths(args.models)
        if (
            selected["schema"] != 1
            or selected["fixed_control"] != selected_fixed(args)[0]
            or selected["variant"] not in models
            or sha(models[selected["variant"]]) != selected["model_sha256"]
        ):
            raise ValueError("selected model changed before fresh heldout")
        records = scored_conversations(
            args, fresh["groups"]["heldout"], args.fresh,
            "heldout", [selected["variant"]],
        )
    save(args.out / f"{args.phase}-summary.json", {
        "schema": 1, "phase": args.phase, "records": records,
    })
    print(json.dumps({"phase": args.phase, "sessions": len(records)}))


if __name__ == "__main__":
    main()
