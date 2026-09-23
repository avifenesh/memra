"""Pinned Qwen sampler-top-k qualification and development collector."""

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


LANE = Path(__file__).resolve().parent.parent
MODEL_SHA = "1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a"
TOP_K = (3, 10, 20)


def load_v4(module_name):
    spec = importlib.util.spec_from_file_location(
        "joint_v4_" + module_name, LANE / "joint-v4" / (module_name + ".py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v4 = load_v4("run_native")
quality = load_v4("quality")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def input_manifests(args):
    if sha(args.model) != MODEL_SHA:
        raise ValueError("Qwen full-head checkpoint differs")
    args.binary_sha = sha(args.binary)
    if len(args.source_sha) != 64:
        raise ValueError("missing exact runtime source digest")
    development = json.loads((args.development / "manifest.json").read_text())
    fresh = json.loads((args.fresh / "manifest.json").read_text())
    for manifest, directory in ((development, args.development), (fresh, args.fresh)):
        if manifest["schema"] != 1:
            raise ValueError("another workload manifest schema")
        for entries in manifest["groups"].values():
            for entry in entries:
                if sha(directory / entry["file"]) != entry["sha256"]:
                    raise ValueError("frozen workload text changed")
    return development, fresh


def record(root, label, k, arm, audit, schedule=None):
    observed = [
        int(row["sampler_top_k"]) for row in v4.table(root / "turns.tsv")
    ]
    return {
        "name": root.name, "variant": label,
        "sampler_top_k": (
            "learned" if arm in ("learn-topk", "joint-ckd")
            else k if schedule is None else "scheduled"
        ),
        "sampler_schedule": list(schedule) if schedule is not None else None,
        "observed_top_k": observed,
        "arm": arm, "tokens": sum(row["output_tokens"] for row in audit),
        "seconds": sum(row["elapsed_s"] for row in audit),
        "format": sum(bool(row["fenced_parseable_function"]) for row in audit),
        "loops": sum(bool(row["loop"]) for row in audit),
    }


def validate(root, entry, arm):
    audit = v4.validate(root, entry, "fixed:3" if arm == "explore-d" else arm)
    if arm != "explore-d":
        return audit
    depth = v4.table(root / "rounds.tsv")
    counts = defaultdict(int)
    for turn in range(1, 9):
        confidence = v4.table(root / f"turn-{turn}.confidence.tsv")
        current = [row for row in depth if int(row["turn"]) == turn]
        if len(confidence) != len(current):
            raise ValueError("v5 randomized confidence and depth inventories differ")
        for offered, selected in zip(confidence, current):
            d = int(selected["draft_depth"])
            if (
                d not in (1, 2, 3, 4)
                or int(offered["drafted"]) != d
                or int(offered["round"]) != int(selected["round"])
                or offered["eligible"] != selected["eligible_for_learning"]
            ):
                raise ValueError("v5 sampled D assignment differs from actual offer")
            if offered["eligible"] == "true":
                counts[d] += 1
    exposure = {str(d): counts[d] for d in range(1, 5)}
    path = root / "exposure.json"
    if path.exists():
        if json.loads(path.read_text()) != exposure:
            raise ValueError("v5 randomized D exposure changed on resume")
    else:
        save(path, exposure)
    return audit


def run_one(args, entry, workloads, topic, label, k, arm, cap, *,
            temperature=1.0, extra=(), schedule=None):
    root = args.out / f"{topic}-{label}"
    command = [
        str(args.binary), str(args.model), "embedded",
        str(workloads / entry["file"]), str(root), arm, str(entry["seed"]),
        "8192", "65536", str(temperature), f"cap={cap}",
        f"sampler-top-k={k}", *extra,
    ]
    if schedule is not None:
        if len(schedule) != 8 or any(value not in TOP_K for value in schedule):
            raise ValueError("sampler schedule needs eight allowed top-k actions")
        command.append("sampler-schedule=" + ",".join(map(str, schedule)))
    environment = os.environ.copy()
    environment.update({
        "MEMRA_SPEC_ADAPT": "0", "MEMRA_SPEC_ADAPT_FLOOR": "1",
        "MEMRA_SPEC_CAPMAX": "7", "MEMRA_SPEC_PMIN": "0",
        "MEMRA_SPEC_PMIN0": "0", "MEMRA_SPEC_PMIN_INROUND": "0",
        "MEMRA_SPEC_STATS": "1",
    })
    if environment.get("MEMRA_CAPTURE_DIR"):
        raise ValueError("research pod cannot capture customer content")
    command_record = {
        "argv": command, "binary_sha256": args.binary_sha,
        "source_sha256": args.source_sha, "model_sha256": MODEL_SHA,
        "workload_sha256": entry["sha256"], "sampler_top_k": k,
        "sampler_schedule": list(schedule) if schedule is not None else None,
        "topk_model_sha256": next(
            (
                sha(option.removeprefix("topk-model="))
                for option in extra if option.startswith("topk-model=")
            ),
            None,
        ),
        "joint_models": {
            option.split("=", 1)[0]: option.split("=", 1)[1]
            for option in extra
            if option.startswith((
                "joint-model-dir=", "depth-variant=", "confidence-variant=",
            ))
        },
        "temperature": temperature, "cap": cap,
        "memra_settings": {
            key: value for key, value in environment.items()
            if key.startswith("MEMRA_")
        },
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
            raise ValueError("incomplete or changed native K cell: " + root.name)
        audit = validate(root, entry, arm)
        turns = v4.table(root / "turns.tsv")
        actual = [int(row["sampler_top_k"]) for row in turns]
        if (
            (schedule is not None and actual != list(schedule))
            or (schedule is None and arm not in ("learn-topk", "joint-ckd")
                and actual != [k] * 8)
            or (arm in ("learn-topk", "joint-ckd")
                and any(value not in TOP_K for value in actual))
        ):
            raise ValueError("resumed sampler action differs")
        if arm in ("learn-topk", "noop-topk", "joint-ckd", "noop-ckd") and any(
            int(row["k_model_ns"]) == 0 for row in turns
        ):
            raise ValueError("K model did not run inside request clock")
        audit_path = root / "audit.json"
        if audit_path.exists():
            if json.loads(audit_path.read_text()) != audit:
                raise ValueError("resumed native K audit differs")
        else:
            save(audit_path, audit)
        return record(root, label, k, arm, audit, schedule)
    save(command_path, command_record)
    started = time.monotonic()
    with (args.out / f"{root.name}.stdout.log").open("x") as stdout, (
        args.out / f"{root.name}.stderr.log"
    ).open("x") as stderr:
        result = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr)
    save(exit_path, {"returncode": result.returncode, "wall_s": time.monotonic() - started})
    if result.returncode:
        raise RuntimeError(root.name + " native program failed; inspect stderr")
    engagement = (args.out / f"{root.name}.stderr.log").read_text()
    if (
        "full_vocab=248320 draft_vocab=248320 mtp=embedded" not in engagement
        or f"sampler_top_k={k}" not in engagement
    ):
        raise ValueError("full-head sampler top-k engagement was not logged")
    audit = validate(root, entry, arm)
    turns = v4.table(root / "turns.tsv")
    expected = [k] * 8 if schedule is None else list(schedule)
    actual = [int(row["sampler_top_k"]) for row in turns]
    if (
        (arm not in ("learn-topk", "joint-ckd") and actual != expected)
        or (arm in ("learn-topk", "joint-ckd")
            and any(value not in TOP_K for value in actual))
    ):
        raise ValueError("actual sampled top-k differs from frozen schedule")
    if arm in ("learn-topk", "noop-topk", "joint-ckd", "noop-ckd") and any(
        int(row["k_model_ns"]) == 0 for row in turns
    ):
        raise ValueError("K model did not run inside request clock")
    save(root / "audit.json", audit)
    return record(root, label, k, arm, audit, schedule)


def qualify(args, fresh):
    entry = fresh["groups"]["qualification"][0]
    records = [
        run_one(args, entry, args.fresh, "qualification-0",
                f"topk{k}-d3-c0", k, "fixed:3", 3)
        for k in (20, 10, 3)
    ]
    functional = {}
    for record in records:
        root = args.out / record["name"]
        outcomes = [
            quality.probe(
                (root / f"turn-{turn}.answer.txt").read_text(),
                expected["function"],
            )["pass"]
            for turn, expected in enumerate(entry["turns"], 1)
        ]
        functional[str(record["sampler_top_k"])] = sum(outcomes)
    eligible = {
        str(record["sampler_top_k"]): (
            record["format"] == 8 and record["loops"] == 0
            and functional[str(record["sampler_top_k"])] == 8
        )
        for record in records
    }
    if not eligible["20"]:
        raise ValueError("Qwen recommended top-k=20 qualifier failed")
    save(args.out / "qualification-result.json", {
        "schema": 1, "eligible": eligible, "functional_pass": functional,
        "status": "matched top-k qualifier complete",
    })
    return records


def development(args, old):
    qualifier = json.loads((args.out / "qualification-result.json").read_text())
    frozen = json.loads((Path(__file__).with_name("schedules.json")).read_text())
    if (
        frozen["schema"] != 1
        or frozen["workloads_sha256"] != sha(args.development / "manifest.json")
        or frozen["generator_sha256"] != sha(Path(__file__).with_name("schedules.py"))
    ):
        raise ValueError("frozen K assignments or generator differ")
    learnable = [
        k for k in TOP_K if qualifier["eligible"][str(k)]
    ]
    if 20 not in learnable:
        raise ValueError("top-k=20 code qualifier is required")
    topics = old["groups"]["calibration"] + old["groups"]["heldout"][:3]
    records = []
    for index, entry in enumerate(topics):
        assigned = frozen["training"][index]
        if assigned["training_index"] != index or assigned["workload_sha256"] != entry["sha256"]:
            raise ValueError("frozen sampler schedule differs from this conversation")
        arms = [
            (f"topk{k}-d3-c0", k, "fixed:3", 3, ())
            for k in TOP_K
        ]
        arms.extend(
            (f"topk{k}-random-d", k, "explore-d", 4,
             (f"explore-seed={20785000 + index * 31 + k}",))
            for k in learnable
        )
        order = arms[index % len(arms):] + arms[:index % len(arms)]
        if index % 2:
            order.reverse()
        for label, k, arm, cap, extra in order:
            records.append(run_one(
                args, entry, args.development, f"development-{index}",
                label, k, arm, cap, extra=extra,
            ))
        action_set = set(learnable)
        schedule_key = {
            frozenset((3, 10, 20)): "topk_3_10_20",
            frozenset((3, 20)): "topk_3_20",
            frozenset((10, 20)): "topk_10_20",
        }.get(frozenset(action_set))
        if schedule_key is not None:
            records.append(run_one(
                args, entry, args.development, f"development-{index}",
                "random-topk", 20, "fixed:3", 3,
                schedule=assigned[schedule_key],
            ))
    for k in learnable:
        counts = defaultdict(int)
        for record in records:
            if record["sampler_top_k"] != k or record["arm"] != "explore-d":
                continue
            exposure = json.loads((args.out / record["name"] / "exposure.json").read_text())
            for d, value in exposure.items():
                counts[d] += value
        if min(counts.values(), default=0) < 150 or len(counts) != 4:
            raise ValueError(f"top-k={k} lacks randomized training exposure")
    return records


def old_code_probes(root, entry):
    worker = LANE / "confidence/adaptive-v3/quality.py"
    passed = 0
    for turn, expected in enumerate(entry["turns"], 1):
        payload = {
            "text": (root / f"turn-{turn}.answer.txt").read_text(),
            "function": expected["function"],
        }
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(worker), "--worker"],
                input=json.dumps(payload), text=True, capture_output=True,
                timeout=3,
            )
            passed += (
                result.returncode == 0
                and json.loads(result.stdout)["pass"] is True
            )
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
    return passed


def select_topk(args, old):
    trained = json.loads((args.out / "topk-training-summary.json").read_text())
    manifest = json.loads((args.models / "topk-models.json").read_text())
    if len(manifest) != 3 or trained["schema"] != 1:
        raise ValueError("K router model manifest differs")
    model_paths = {}
    for item in manifest:
        path = args.models / f"topk-{item['variant']}.tsv"
        if (
            item["variant"] not in ("first16", "first32", "prior")
            or item["variant"] in model_paths
            or sha(path) != item["model_sha256"]
        ):
            raise ValueError("K router weights differ from frozen training")
        model_paths[item["variant"]] = path
    records = []
    topics = old["groups"]["heldout"][3:]
    for index, entry in enumerate(topics):
        arms = [
            (f"topk{k}-d3-c0", k, "fixed:3", ())
            for k in TOP_K
        ]
        if len(trained["quality_eligible_top_k"]) > 1:
            for variant, path in model_paths.items():
                arms.append((
                    f"learn-{variant}", 20, "learn-topk",
                    (f"topk-model={path}",),
                ))
                arms.append((
                    f"noop-{variant}", 20, "noop-topk",
                    (f"topk-model={path}",),
                ))
        order = arms[index % len(arms):] + arms[:index % len(arms)]
        if index % 2:
            order.reverse()
        for label, k, arm, extra in order:
            records.append(run_one(
                args, entry, args.development, f"selection-{index}",
                label, k, arm, 3, extra=extra,
            ))
    scores = defaultdict(list)
    for record in records:
        index = int(record["name"].split("-")[1])
        record["functional_pass"] = old_code_probes(
            args.out / record["name"], topics[index],
        )
        scores[record["variant"]].append(record)
    rates = {}
    for variant, rows in scores.items():
        if len(rows) != 3:
            raise ValueError("K selector lacks matched development topics")
        tokens = sum(row["tokens"] for row in rows)
        seconds = sum(row["seconds"] for row in rows)
        rates[variant] = {
            "tokens": tokens, "seconds": seconds,
            "tok_s": tokens / seconds,
            "format_pass": sum(row["format"] for row in rows),
            "functional_pass": sum(row["functional_pass"] for row in rows),
            "loops": sum(row["loops"] for row in rows),
        }
    qualified_fixed = [
        k for k in TOP_K
        if rates[f"topk{k}-d3-c0"]["format_pass"] == 24
        and rates[f"topk{k}-d3-c0"]["functional_pass"] == 24
        and rates[f"topk{k}-d3-c0"]["loops"] == 0
    ]
    if 20 not in qualified_fixed:
        raise ValueError("recommended K=20 fails development-selection quality")
    fixed_best = max(
        qualified_fixed, key=lambda k: rates[f"topk{k}-d3-c0"]["tok_s"],
    )
    qualified_learned = [
        variant for variant in model_paths
        if f"learn-{variant}" in rates
        and rates[f"learn-{variant}"]["format_pass"] == 24
        and rates[f"learn-{variant}"]["functional_pass"] == 24
        and rates[f"learn-{variant}"]["loops"] == 0
    ]
    chosen = max(
        qualified_learned,
        key=lambda variant: rates[f"learn-{variant}"]["tok_s"],
        default=None,
    )
    save(args.out / "selected-topk.json", {
        "schema": 1, "variant": chosen,
        "model_sha256": sha(model_paths[chosen]) if chosen else None,
        "quality_eligible_actions": trained["quality_eligible_top_k"],
        "best_fixed_top_k": fixed_best,
        "selection_scores": rates,
        "status": "selected" if chosen else "no-quality-qualified-K-router",
    })
    return records


def main():
    parser = argparse.ArgumentParser()
    for name in ("binary", "model", "development", "fresh", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument(
        "--phase", choices=("qualification", "development", "selection"),
        required=True,
    )
    args = parser.parse_args()
    for name in ("binary", "model", "development", "fresh", "out"):
        setattr(args, name, getattr(args, name).resolve())
    if args.models is not None:
        args.models = args.models.resolve()
    args.out.mkdir(exist_ok=True)
    old, fresh = input_manifests(args)
    if args.phase == "qualification":
        records = qualify(args, fresh)
    elif args.phase == "development":
        records = development(args, old)
    else:
        if args.models is None:
            raise ValueError("K model selection needs frozen trained weights")
        records = select_topk(args, old)
    save(args.out / f"{args.phase}-summary.json", {
        "schema": 1, "phase": args.phase, "records": records,
    })
    print(json.dumps({"phase": args.phase, "sessions": len(records)}))


if __name__ == "__main__":
    main()
