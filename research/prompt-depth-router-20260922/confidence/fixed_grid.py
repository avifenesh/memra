"""Frozen Qwen K=3 cutoff grid on the prefix study's independent requests."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prefix"))
from audit import same_tapes, save, sha  # noqa: E402
from run import Runner  # noqa: E402


ARMS = {
    "off": (0.0, False),
    "c015": (0.15, False),
    "c030": (0.30, False),
    "c030zero": (0.30, True),
}


def orders():
    labels = list(ARMS)
    result = []
    for scenario in range(6):
        shift = scenario // 2
        order = labels[shift:] + labels[:shift]
        result.append(order[::-1] if scenario % 2 else order)
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("repo", "models", "binaries", "source", "workloads", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for name in ("repo", "models", "binaries", "source", "workloads", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.workloads / "manifest.json").read_text())
    inputs = manifest["families"]["qwen"]
    source = json.loads(args.source.read_text())
    freeze = {
        "schema": 1,
        "question": "fixed Qwen code K=3 with confidence stopping",
        "arms": {name: {"pmin": pmin, "pmin0": pmin0} for name, (pmin, pmin0) in ARMS.items()},
        "orders": orders(),
        "workloads_sha256": sha(args.workloads / "manifest.json"),
        "source_sha256": sha(args.source),
        "runtime_source_sha256": source["runtime_source_sha256"],
        "script_sha256": sha(Path(__file__)),
        "sampling": {"temperature": 0.7, "top_k": 20, "top_p": 0.95},
        "k": 3,
        "max_new": 8192,
        "ctx": 32768,
        "scenarios": list(range(6)),
        "primary_turns": [2, 4, 6, 8],
        "scope": "one non-production RTX 5090; independent sampled native requests",
        "status": "registered-before-generation",
    }
    save(args.out / "FREEZE.json", freeze)
    state = {"status": "running", "completed": []}
    save(args.out / "status.json", state)
    runner = None
    try:
        runner = Runner(
            args.repo, args.models, args.binaries, args.source, args.out, families=("qwen",)
        )
        qualification = inputs["qualification"]
        gates = []
        for label in ("off", "c030", "c030zero"):
            pmin, pmin0 = ARMS[label]
            result = runner.run(
                "qwen", "qualification", f"greedy-{label}", qualification,
                args.workloads / qualification["file"], "fixed:3", 128,
                gate=True, pmin=pmin, pmin0=pmin0, native_adapt=False,
                spec_stats=True,
            )
            gates.append(result)
            state["completed"].append(result["path"])
            save(args.out / "status.json", state)
        for gate in gates[1:]:
            same_tapes(args.out / gates[0]["path"], args.out / gate["path"])
        for scenario, order in enumerate(freeze["orders"]):
            entry = inputs["scenarios"][str(scenario)]
            for label in order:
                pmin, pmin0 = ARMS[label]
                result = runner.run(
                    "qwen", "scored", f"{scenario:02}-{label}", entry,
                    args.workloads / entry["file"], "fixed:3", freeze["max_new"],
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
