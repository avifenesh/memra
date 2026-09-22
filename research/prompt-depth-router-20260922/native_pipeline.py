"""Finite native experiment. Runs only on the dedicated research GPU host."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from fit_profile import fit_profile


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def main():
    parser = argparse.ArgumentParser()
    for name in ("repo", "models", "binaries", "workloads", "out", "recipe_commit"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    args = parser.parse_args()
    repo, models, binaries, workloads, out = (
        Path(getattr(args, key)).resolve()
        for key in ("repo", "models", "binaries", "workloads", "out")
    )
    out.mkdir(parents=True, exist_ok=False)
    base = repo / "research/mtp-context-depth-20260921"
    sys.path.insert(0, str(base))
    from calibrate import freeze
    from loop_audit import loop_candidate

    pools = json.loads((workloads / "manifest.json").read_text())
    freeze_record = {
        "pipeline_sha256": sha(Path(__file__)),
        "recipe_commit": args.recipe_commit,
        "runtime_binding_sha256": sha(repo / "REQUEST-ROUTING-SOURCE.json"),
        "workload_manifest_sha256": sha(workloads / "manifest.json"),
        "calibration_pools": ["cal-a", "cal-b", "cal-c"],
        "heldout_pools": [f"held-{c}" for c in "abcdef"],
        "seeds": {
            "qwen": {"calibration": [20311000 + i for i in range(3)],
                     "heldout": [20312000 + i for i in range(6)], "qualification": 20313000},
            "gemma": {"calibration": [20321000 + i for i in range(3)],
                      "heldout": [20322000 + i for i in range(6)], "qualification": 20323000},
        },
        "max_new": 512, "ctx": 49152, "temperature": 0.7,
        "loop_auditor_sha256": sha(base / "loop_audit.py"),
        "loop_rule": "period 1..128; at least four repeats and at least 256 tokens; overlapping end windows",
        "orders": [], "status": "registered_before_generation",
    }
    for i in range(6):
        order = ["native", "calibrated", "context", "routed"]
        shift = i // 2
        order = order[shift:] + order[:shift]
        if i % 2:
            order.reverse()
        freeze_record["orders"].append(order)
    save(out / "FREEZE.json", freeze_record)
    state = {"status": "running", "completed_groups": []}
    save(out / "status.json", state)

    def group(family, phase, index, pool, seed, order, fixed, requests=None, priors=None,
              gate=False, tokens=512):
        maximum = 7 if family == "qwen" else 5
        identity = pools["families"][family][pool]
        workload = workloads / identity["file"]
        if sha(workload) != identity["sha256"]:
            raise ValueError("frozen workload changed")
        parent = out / family / phase
        parent.mkdir(parents=True, exist_ok=True)
        group_name = f"{index:02}-{pool}"
        receipt_dir = parent / group_name
        schedule = parent / f"{group_name}-schedule.json"
        fixed_path = parent / f"{group_name}-fixed.json"
        request_path = parent / f"{group_name}-requests.json"
        save(schedule, [{"cycle": 0, "order": order}])
        save(fixed_path, fixed)
        save(request_path, requests or {})
        binary = binaries / ("mtp-depth-study" if family == "qwen" else "gemma-depth-study")
        cmd = [
            sys.executable, str(base / "run_study.py"), "--family", family,
            "--binary", str(binary), "--target", str(models / family / "target.gguf"),
            "--workload", str(workload), "--out", str(receipt_dir),
            "--lock", "/tmp/memra-gpu.lock", "--max-new", str(tokens), "--ctx", "49152",
            "--seed", str(seed), "--artifact-manifest", str(models / family / "artifacts.lock.json"),
            "--source-commit", args.recipe_commit,
            "--runtime-binding", str(repo / "REQUEST-ROUTING-SOURCE.json"),
            "--schedule", str(schedule), "--fixed-depths", str(fixed_path),
            "--request-depths", str(request_path),
        ]
        if family == "gemma":
            cmd += ["--draft", str(models / family / "assistant.gguf")]
        if priors:
            cmd += ["--priors", str(priors)]
        if gate:
            cmd += ["--gate"]
        state.update(family=family, phase=phase, index=index, pool=pool, order=order)
        save(out / "status.json", state)
        save(parent / f"{group_name}-command.json", cmd)
        started = time.monotonic()
        with (parent / f"{group_name}.log").open("w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=2400)
        save(parent / f"{group_name}-exit.json",
             {"returncode": result.returncode, "process_seconds": time.monotonic() - started})
        if result.returncode:
            raise RuntimeError(f"{family}/{phase}/{group_name} failed; retain and inspect the attempt")
        records = json.loads((receipt_dir / "runs.json").read_text())
        if [r["arm"] for r in records] != order:
            raise ValueError("runner omitted or reordered a requested arm")
        loops = []
        for record in records:
            run = receipt_dir / f'{seed}-{record["arm"]}'
            turns = rows(run / "turns.tsv")
            routing = rows(run / "routing.tsv")
            spans = rows(run / "spans.tsv")
            if len(routing) != 8 or sum(int(t["drafted"]) for t in turns) <= 0:
                raise ValueError("routing coverage or speculative engagement missing")
            for turn, route in enumerate(routing, 1):
                if int(route["turn"]) != turn:
                    raise ValueError("routing turn order changed")
                user = (run / f"turn-{turn}.user.txt").read_bytes()
                if hashlib.sha256(user).hexdigest() != identity["instruction_sha256"][turn - 1]:
                    raise ValueError("router did not consume the frozen user message")
                if route["source"] == "prompt":
                    if route["kind"] != identity["predicted_kinds"][turn - 1]:
                        raise ValueError("native and pre-generation CPU classification differ")
                    k = int(route["k"])
                    if not 1 <= k <= maximum:
                        raise ValueError("routing escaped the native depth cap")
                    full = [s for s in spans if int(s["turn"]) == turn and s["eligible"] == "true"]
                    if any(int(s["k"]) != k for s in full):
                        raise ValueError("a full round changed K inside the request")
                ids = [int(v) for v in (run / f"turn-{turn}.output.ids").read_text().split()]
                flag = loop_candidate(ids)
                if flag:
                    loops.append({"arm": record["arm"], "turn": turn, **flag})
        summary = {
            "receipt_dir": str(receipt_dir.relative_to(out)), "seed": seed,
            "pool": pool, "records": records, "loop_flags": loops,
            "excluded": bool(loops), "exclusion_scope": "complete matched policy set",
        }
        save(parent / f"{group_name}-audit.json", summary)
        state["completed_groups"].append({k: summary[k] for k in ("receipt_dir", "seed", "pool", "excluded")})
        save(out / "status.json", state)
        print(json.dumps({"completed": summary["receipt_dir"], "excluded": bool(loops)}), flush=True)
        return summary

    try:
        for family in ("qwen", "gemma"):
            maximum = 7 if family == "qwen" else 5
            seeds = freeze_record["seeds"][family]
            fixed = {f"k{k}": k for k in range(1, maximum + 1)}
            # Every legal fixed depth gets a target/greedy check before calibration.
            group(family, "qualification", 0, "qualification", seeds["qualification"],
                  ["native", *fixed], fixed, gate=True, tokens=128)
            calibration = []
            for i, pool in enumerate(freeze_record["calibration_pools"]):
                order = list(fixed)
                order = order[i:] + order[:i]
                if i % 2:
                    order.reverse()
                calibration.append(group(
                    family, "calibration", i, pool, seeds["calibration"][i], order, fixed
                ))
            clean = [g for g in calibration if not g["excluded"]]
            if len(clean) < 2:
                raise ValueError("fewer than two clean calibration scenarios")
            global_k, priors, span_fit = freeze(out, family, clean, maximum)

            def rate(group_record, k, kind):
                run = out / group_record["receipt_dir"] / f'{group_record["seed"]}-k{k}'
                turns = rows(run / "turns.tsv")
                predicted = pools["families"][family][group_record["pool"]]["predicted_kinds"]
                chosen = [t for t in turns if predicted[int(t["turn"]) - 1] == kind]
                return sum(int(t["output_tokens"]) for t in chosen), sum(float(t["elapsed_s"]) for t in chosen)

            trials = {
                kind: {k: [rate(g, k, kind) for g in clean] for k in fixed.values()}
                for kind in ("prose", "code", "numeric")
            }
            profile, fitting = fit_profile(global_k, trials, [g["seed"] for g in clean])
            arm = "prompt:" + ",".join(str(profile[k]) for k in ("prose", "code", "numeric", "fallback"))
            fit = {"global_k": global_k, "profile": profile, "prompt_fit": fitting,
                   "span_fit": span_fit, "calibration": calibration, "runtime_arm": arm,
                   "priors_sha256": sha(priors)}
            save(out / f"{family}-FIT.json", fit)
            predicted = pools["families"][family]["qualification"]["predicted_kinds"]
            schedule = [profile.get(kind, global_k) for kind in predicted]
            group(family, "qualification", 1, "qualification", seeds["qualification"] + 1,
                  ["native", "routed"], {}, {"routed": arm}, gate=True, tokens=128)
            pair = group(
                family, "qualification", 2, "qualification", seeds["qualification"] + 2,
                ["routed", "replay"], {},
                {"routed": arm, "replay": "schedule:" + ",".join(map(str, schedule))},
                tokens=128,
            )
            for turn in range(1, 9):
                for tape in ("prompt", "output"):
                    a = out / pair["receipt_dir"] / f'{pair["seed"]}-routed/turn-{turn}.{tape}.ids'
                    b = out / pair["receipt_dir"] / f'{pair["seed"]}-replay/turn-{turn}.{tape}.ids'
                    if a.read_bytes() != b.read_bytes():
                        raise ValueError("sampled prompt/schedule token identity failed")
            heldout = [
                group(family, "heldout", i, pool, seeds["heldout"][i],
                      freeze_record["orders"][i], {"calibrated": global_k},
                      {"routed": arm}, priors=priors)
                for i, pool in enumerate(freeze_record["heldout_pools"])
            ]
            save(out / f"{family}-HELDOUT.json", heldout)
        state["status"] = "completed"
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save(out / "status.json", state)


if __name__ == "__main__":
    main()
