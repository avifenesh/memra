"""Frozen fixed-three versus bounded-prompt matrix; no fitting on scored outputs."""

import argparse
import json
from pathlib import Path
import select
import sys

from audit import same_tapes, save, sha
from run import Runner

ARMS = {"fixed3": "fixed:3", "prefix64": "prefix:64",
        "prefix128": "prefix:128", "prefix256": "prefix:256"}


def orders():
    result = []
    for index in range(6):
        row = list(ARMS)
        shift = index // 2
        row = row[shift:] + row[:shift]
        result.append(row[::-1] if index % 2 else row)
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("repo", "models", "binaries", "source", "workloads", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--backup-checkpoints", action="store_true")
    args = parser.parse_args()
    for name in ("repo", "models", "binaries", "source", "workloads", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    workloads = json.loads((args.workloads / "manifest.json").read_text())
    freeze = {
        "schema": 1, "workloads_sha256": sha(args.workloads / "manifest.json"),
        "pipeline_sha256": sha(Path(__file__)), "source_sha256": sha(args.source),
        "arms": ARMS, "orders": orders(), "scenarios": list(range(6)),
        "length_targets": [256, 1024, 4096, 16384], "prefix_budgets": [64, 128, 256],
        "ctx": 32768, "temperature": 0.7, "top_k": 20, "top_p": 0.95,
        "budget_candidates": [8192, 12288],
        "budget_rule": "first qualification budget with all eight requested final formats covered and no loops; never inspect scored outputs to select budget",
        "loop_rule": "pinned loop_audit.py; any arm flags a request => exclude that scenario/type/length request from every arm",
        "coverage_rule": "report failures and capped outputs in the primary requested-type table; add an all-arms-format-covered matched subset; no replacement seeds",
        "minimum_pairs_for_promotion": 6,
        "scope": "independent native sampled requests; default thinking; unchanged heads",
        "status": "registered-before-generation",
    }
    save(args.out / "FREEZE.json", freeze)
    state = {"status": "running", "completed": [], "budgets": {}}
    save(args.out / "status.json", state)
    runner = None

    def checkpoint():
        save(args.out / "status.json", state)
        if args.backup_checkpoints:
            sequence = len(state["completed"])
            print(json.dumps({"backup_ready": sequence}), flush=True)
            if not select.select([sys.stdin], [], [], 120)[0]:
                raise RuntimeError("receipt backup acknowledgement timed out")
            if sys.stdin.readline().strip() != f"ACK {sequence}":
                raise RuntimeError("receipt backup was not acknowledged")

    def run(family, phase, label, entry, arm, budget, gate=False, seed=None):
        state.update(family=family, phase=phase, label=label)
        save(args.out / "status.json", state)
        result = runner.run(family, phase, label, entry,
                            args.workloads / entry["file"], arm, budget, gate, seed)
        state["completed"].append(result["path"])
        checkpoint()
        return result

    try:
        runner = Runner(args.repo, args.models, args.binaries, args.source, args.out)
        for family in ("qwen", "gemma"):
            family_inputs = workloads["families"][family]
            probe = family_inputs["qualification"]
            gates = []
            for k in (2, 3, 4):
                gates.append(run(family, "qualification", f"greedy-k{k}",
                                 probe, f"fixed:{k}", 128, gate=True))
            for other in gates[1:]:
                same_tapes(args.out / gates[0]["path"], args.out / other["path"])
            routed = run(family, "qualification", "sampled-prefix128",
                         probe, "prefix:128", 128, seed=probe["seed"] + 10)
            schedule = "schedule:" + ",".join(str(row["k"]) for row in routed["requests"])
            replay = run(family, "qualification", "sampled-replay128",
                         probe, schedule, 128, seed=probe["seed"] + 10)
            same_tapes(args.out / routed["path"], args.out / replay["path"])
            budget = None
            budget_records = []
            for candidate in freeze["budget_candidates"]:
                result = run(family, "qualification", f"coverage-{candidate}",
                             probe, "fixed:3", candidate, seed=probe["seed"] + 20)
                budget_records.append(result)
                if all(row["format"]["requested_format_covered"] and not row["loop"]
                       for row in result["requests"]):
                    budget = candidate
                    break
            save(args.out / f"{family}-qualification.json", {
                "greedy": [row["path"] for row in gates],
                "sampled_identity": [routed["path"], replay["path"]],
                "coverage_runs": [row["path"] for row in budget_records],
                "selected_max_new": budget, "selection_precedes_scoring": True,
            })
            if budget is None:
                raise ValueError(f"{family} did not qualify final prose/code coverage")
            state["budgets"][family] = budget
            for index, order in enumerate(freeze["orders"]):
                entry = family_inputs["scenarios"][str(index)]
                group = []
                for label in order:
                    group.append(run(family, "scored", f"{index:02}-{label}",
                                     entry, ARMS[label], budget))
                excluded = [
                    turn for turn in range(1, 9)
                    if any(row["requests"][turn - 1]["loop"] for row in group)
                ]
                save(args.out / family / "scored" / f"{index:02}-matched.json", {
                    "scenario": index, "order": order, "paths": [row["path"] for row in group],
                    "excluded_turns": excluded, "exclusion_scope": "same request in all four arms",
                })
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
