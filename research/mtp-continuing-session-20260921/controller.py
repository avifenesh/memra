"""Freeze fixed-depth calibration, then run ten balanced warm-session comparisons."""
import argparse
import datetime
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

LANE = Path(__file__).resolve().parent
ARMS = ("fixed", "native", "learned", "measured", "calibrated")


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schedule():
    base = (0, 1, 4, 2, 3)
    rows = [tuple(ARMS[(i + offset) % 5] for i in base) for offset in range(5)]
    return rows + [tuple(reversed(row)) for row in rows]


def summarize(groups):
    for group in groups:
        rows = group["records"]
        if {r["arm"] for r in rows} != set(ARMS) or len(rows) != len(ARMS):
            raise ValueError("incomplete matched arm set")
        if any(r["correctness_only"] or r["turns"] != 8 or r["elapsed_s"] <= 0
               or r["tokens"] <= 0 or r["reuse"]["cached_tokens"] <= 0 for r in rows):
            raise ValueError("invalid warm-session measurement")
    pooled = {}
    for arm in ARMS:
        rows = [next(r for r in g["records"] if r["arm"] == arm) for g in groups]
        pooled[arm] = sum(r["tokens"] for r in rows) / sum(r["elapsed_s"] for r in rows)
    comparisons = {}
    for control in (a for a in ARMS if a != "learned"):
        gains = []
        for group in groups:
            rates = {r["arm"]: r["e2e_tok_s"] for r in group["records"]}
            gains.append(100 * (rates["learned"] / rates[control] - 1))
        comparisons[control] = {
            "pooled_change_percent": 100 * (pooled["learned"] / pooled[control] - 1),
            "paired_percent": gains, "wins": sum(g > 0 for g in gains),
            "median_pair_percent": statistics.median(gains),
        }
    return {
        "complete_planned_matrix": len(groups) == 10,
        "sets": len(groups), "runs": len(groups) * len(ARMS),
        "turns": len(groups) * len(ARMS) * 8,
        "pooled_request_e2e_tok_s": pooled, "learned_vs_control": comparisons,
        "cache_policy": "stable-prompt-checkpoint-reuse-required",
        "native_session_not_http": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["models", "binaries", "workloads", "qualification", "out"]:
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--source", required=True)
    a = p.parse_args()
    qualification = json.loads((a.qualification / "status.json").read_text())
    if qualification["state"] != "passed":
        raise ValueError("cold/resumed qualification has not passed")
    expected_checks = {(family, shape) for family in ["qwen", "gemma"] for shape in ["short", "long"]}
    if {(c["family"], c["shape"]) for c in qualification["checks"]} != expected_checks:
        raise ValueError("qualification did not cover both model families and shapes")
    for check in qualification["checks"]:
        family = check["family"]
        binary = a.binaries / ("mtp-depth-study" if family == "qwen" else "gemma-depth-study")
        artifacts = json.loads((a.models / family / "artifacts.lock.json").read_text())
        for mode in ["cold", "warm"]:
            identity = json.loads((a.qualification / check[mode] / "identity.json").read_text())
            if (identity["source_commit"] != a.source
                    or identity["binary_sha256"] != digest(binary)
                    or identity["runner_sha256"] != digest(LANE / "run_study.py")
                    or identity["audit_reuse_sha256"] != digest(LANE / "audit_reuse.py")
                    or identity["artifacts"] != artifacts):
                raise ValueError("qualification belongs to another runtime or artifact")
    a.out.mkdir(parents=True, exist_ok=False)
    save(a.out / "schedule.json", schedule())
    save(a.out / "qualification.json", qualification)
    save(a.out / "source.json", {"source_commit": a.source})
    workload_lock = a.workloads / "workloads.lock.json"
    for name, row in json.loads(workload_lock.read_text()).items():
        if digest(a.workloads / name) != row["sha256"]:
            raise ValueError("frozen workload changed before calibration")
    (a.out / "workloads.lock.json").write_bytes(workload_lock.read_bytes())

    def status(**fields):
        save(a.out / "status.json", {
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields,
        })
        print(json.dumps(fields), flush=True)

    def attempt(family, phase, cycle, order, workload, mapping):
        name = f"{family}-{phase}-{cycle}"
        root = a.out / name
        spec, depths = root.with_suffix(".schedule.json"), root.with_suffix(".depths.json")
        save(spec, [{"cycle": cycle, "order": order}])
        save(depths, mapping)
        model = a.models / family
        seed = (20266000 if family == "qwen" else 20268000) + (1000 if phase == "eval" else 0)
        cmd = [
            sys.executable, str(LANE / "run_study.py"), "--family", family,
            "--binary", str(a.binaries / ("mtp-depth-study" if family == "qwen" else "gemma-depth-study")),
            "--target", str(model / "target.gguf"),
            "--workload", str(a.workloads / f"{family}-{workload}.txt"),
            "--out", str(root), "--lock", "/tmp/memra-gpu.lock",
            "--max-new", "2048", "--ctx", "49152", "--seed", str(seed),
            "--artifact-manifest", str(model / "artifacts.lock.json"),
            "--source-commit", a.source, "--schedule", str(spec), "--fixed-depths", str(depths),
        ]
        if family == "gemma":
            cmd += ["--draft", str(model / "assistant.gguf")]
        status(state="running", family=family, phase=phase, cycle=cycle, receipt_dir=name)
        with root.with_suffix(".driver.log").open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = json.loads((root / "runs.json").read_text())
        if [r["arm"] for r in rows] != list(order):
            raise ValueError("completed order differs from frozen schedule")
        return {"cycle": cycle, "seed": seed + cycle, "receipt_dir": name, "records": rows}

    try:
        for family, maximum in [("qwen", 7), ("gemma", 5)]:
            depths = {f"k{k}": k for k in range(1, maximum + 1)}
            calibration = []
            for cycle, workload in enumerate(["calibration-a", "calibration-b"]):
                order = list(depths) if cycle == 0 else list(reversed(depths))
                calibration.append(attempt(family, "calibration", cycle, order, workload, depths))
            rates = {}
            for arm in depths:
                rows = [next(r for r in g["records"] if r["arm"] == arm) for g in calibration]
                rates[arm] = sum(r["tokens"] for r in rows) / sum(r["elapsed_s"] for r in rows)
            winner = max(depths, key=lambda key: (rates[key], -depths[key]))
            save(a.out / f"{family}-selection.json", {
                "selected_k": depths[winner], "pooled_calibration_tok_s": rates,
                "selected_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "calibration": calibration, "workloads_lock_sha256": digest(workload_lock),
                "rule": "Pooled returned tokens / request E2E seconds; ties choose smaller K.",
            })
            groups = []
            for cycle, order in enumerate(schedule()):
                groups.append(attempt(
                    family, "eval", cycle, order,
                    "heldout-a" if cycle % 2 == 0 else "heldout-b",
                    {"calibrated": depths[winner]},
                ))
                save(a.out / f"{family}-selected-sets.json", groups)
            report = summarize(groups)
            save(a.out / f"{family}-audit.json", report)
            status(state="family-complete", family=family, report=report)
        save(a.out / "DONE.json", {"families": ["qwen", "gemma"], "sets_each": 10})
        status(state="complete")
    except BaseException as exc:
        status(state="failed", error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    main()
