"""Qualify disjoint Qwen code prompts before opening development C rates."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PREFIX = Path(__file__).resolve().parents[1] / "prefix"
sys.path.insert(0, str(PREFIX))
from audit import same_tapes, save, sha  # noqa: E402
from run import Runner  # noqa: E402
from fixed_grid import ARMS  # noqa: E402


def orders():
    labels = ["k3off", "selected", "k2off"]
    result = []
    for scenario in range(6):
        shift = scenario // 2
        order = labels[shift:] + labels[:shift]
        result.append(order[::-1] if scenario % 2 else order)
    return result


def choose_c(report):
    arms = report["all_code"]["arms"]
    positives = [label for label in ("c015", "c030", "c030zero")
                 if arms[label]["pooled_gain_percent"] > 0]
    return max(positives, key=lambda label: arms[label]["tokens_per_second"]) if positives else None


def target_oracle(out, binaries, models, prompt, label, k, pmin, pmin0, seed):
    settings = {
        "MEMRA_CHAT": "1",
        "MEMRA_PROMPT_FILE": str(prompt),
        "MEMRA_NGEN": "128",
        "MEMRA_SPEC_K": str(k),
        "MEMRA_SPEC_TEMP": "0",
        "MEMRA_TOP_K": "20",
        "MEMRA_TOP_P": "0.95",
        "MEMRA_SEED": str(seed),
        "MEMRA_SPEC_ADAPT": "0",
        "MEMRA_SPEC_PMIN": str(pmin),
        "MEMRA_SPEC_PMIN0": "1" if pmin0 else "0",
        "MEMRA_SPEC_PMIN_INROUND": "0",
    }
    command = [str(binaries / "run-spec"), str(models / "qwen/target.gguf")]
    save(out / f"oracle-{label}.command.json", {
        "argv": command,
        "settings": settings,
        "binary_sha256": sha(binaries / "run-spec"),
    })
    with (out / f"oracle-{label}.log").open("w") as log:
        result = subprocess.run(
            command, env={**os.environ, **settings}, stdout=log,
            stderr=subprocess.STDOUT, timeout=300,
        )
    save(out / f"oracle-{label}.exit.json", {"returncode": result.returncode})
    if (result.returncode != 0
            or "=== SELF-CONSISTENCY PASS ===" not in (out / f"oracle-{label}.log").read_text()):
        raise ValueError("held-out target-only greedy oracle failed: " + label)


def main():
    parser = argparse.ArgumentParser()
    for name in ("repo", "models", "binaries", "source", "workloads", "development_report", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    for name in ("repo", "models", "binaries", "source", "workloads", "development_report", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.workloads / "manifest.json").read_text())
    if manifest["schema"] != 1 or manifest["binary_sha256"] != sha(args.binaries / "qwen-prefix-study"):
        raise ValueError("held-out corpus has another binary or schema")
    if manifest["generator_sha256"] != sha(Path(__file__).with_name("heldout_workloads.py")):
        raise ValueError("held-out task generator changed")
    source = json.loads(args.source.read_text())
    freeze = {
        "schema": 1,
        "status": "registered-before-development-selection",
        "workloads_sha256": sha(args.workloads / "manifest.json"),
        "source_sha256": sha(args.source),
        "runtime_source_sha256": source["runtime_source_sha256"],
        "runner_sha256": sha(Path(__file__)),
        "development_report_sha256": None,
        "selected_c": None,
        "orders": orders(),
        "k3_selected_controls": ["k3off", "selected", "k2off"],
        "sampler": {"temperature": 0.7, "top_k": 20, "top_p": 0.95},
        "max_new": 8192,
        "ctx": 32768,
        "scope": "Qwen code/prose independent native held-out requests on one research RTX 5090",
    }
    save(args.out / "PRESELECTION-FREEZE.json", freeze)
    save(args.out / "FREEZE.json", freeze)
    state = {"status": "qualifying", "completed": []}
    save(args.out / "status.json", state)
    runner = None
    try:
        runner = Runner(
            args.repo, args.models, args.binaries, args.source, args.out, families=("qwen",)
        )
        qualification = manifest["qualification"]
        file = args.workloads / qualification["file"]
        fmt = runner.run(
            "qwen", "qualification", "sampled-format-k3", qualification,
            file, "fixed:3", 8192, pmin=0.0, pmin0=False,
            native_adapt=False, spec_stats=True,
        )
        state["completed"].append(fmt["path"])
        save(args.out / "status.json", state)
        if not all(row["format"]["requested_format_covered"] and not row["loop"]
                   for row in fmt["requests"]):
            raise ValueError("held-out format qualification failed before C selection")

        # The development report stays unopened until the disjoint prompt
        # corpus and its independent format qualification are frozen.
        development = json.loads(args.development_report.read_text())
        development_identity = json.loads(
            (args.development_report.parent / "fixed-grid-v2/identity.json").read_text()
        )
        heldout_identity = json.loads((args.out / "identity.json").read_text())
        if (development["status"] != "measured-fixed-cutoff-only"
                or development["freeze"]["source_sha256"] != sha(args.source)
                or development["freeze"]["runtime_source_sha256"] != source["runtime_source_sha256"]
                or development["freeze"]["k"] != 3
                or development["freeze"]["sampling"] != freeze["sampler"]
                or development_identity["source"] != source
                or development_identity["artifacts"]["qwen"] != json.loads(
                    (args.models / "qwen/artifacts.lock.json").read_text()
                )
                or development_identity["gpu"] != heldout_identity["gpu"]):
            raise ValueError("fixed-C development result differs from the held-out source, model or GPU")
        selected = choose_c(development)
        freeze["development_report_sha256"] = sha(args.development_report)
        freeze["selected_c"] = selected
        save(args.out / "FREEZE.json", freeze)
        if selected is None:
            state["status"] = "no-positive-development-c"
            return

        prompts = file.read_text().split("\n---TURN---\n")
        if len(prompts) != 8:
            raise ValueError("held-out oracle source has another request count")
        oracle_prompt = args.out / "oracle-code-prompt.txt"
        oracle_prompt.write_text(prompts[1])
        for label, k, pmin, pmin0 in (
            ("k2off", 2, 0.0, False),
            ("k3off", 3, 0.0, False),
            ("selected", 3, *ARMS[selected]),
        ):
            target_oracle(
                args.out, args.binaries, args.models, oracle_prompt,
                label, k, pmin, pmin0, qualification["seed"],
            )

        gates = []
        for label, arm, pmin, pmin0 in (
            ("greedy-k2off", "fixed:2", 0.0, False),
            ("greedy-k3off", "fixed:3", 0.0, False),
            ("greedy-selected", "fixed:3", *ARMS[selected]),
        ):
            result = runner.run(
                "qwen", "qualification", label, qualification,
                file, arm, 128, gate=True, pmin=pmin, pmin0=pmin0,
                native_adapt=False, spec_stats=True,
            )
            gates.append(result)
            state["completed"].append(result["path"])
            save(args.out / "status.json", state)
        for other in gates[1:]:
            same_tapes(args.out / gates[0]["path"], args.out / other["path"])

        freeze["status"] = "registered-before-heldout-generation"
        save(args.out / "FREEZE.json", freeze)
        state["status"] = "scoring"
        save(args.out / "status.json", state)
        arms = {
            "k3off": ("fixed:3", 0.0, False),
            "selected": ("fixed:3", *ARMS[selected]),
            "k2off": ("fixed:2", 0.0, False),
        }
        for scenario, order in enumerate(freeze["orders"]):
            entry = manifest["scenarios"][str(scenario)]
            for label in order:
                arm, pmin, pmin0 = arms[label]
                result = runner.run(
                    "qwen", "heldout", f"{scenario:02}-{label}", entry,
                    args.workloads / entry["file"], arm, freeze["max_new"],
                    pmin=pmin, pmin0=pmin0, native_adapt=False,
                    spec_stats=True,
                )
                state["completed"].append(result["path"])
                save(args.out / "status.json", state)
        state["status"] = "completed"
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if runner is not None:
            runner.close()
        save(args.out / "status.json", state)


if __name__ == "__main__":
    main()
