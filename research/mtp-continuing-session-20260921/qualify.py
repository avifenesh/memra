"""Run cold/resumed and depth-policy identity gates before any warm timing study."""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

LANE = Path(__file__).resolve().parent


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def compare(cold, warm, seed):
    for turn in range(1, 9):
        for kind in ["prompt", "output"]:
            name = f"turn-{turn}.{kind}.ids"
            a = (cold / f"{seed}-native" / name).read_bytes()
            b = (warm / f"{seed}-native" / name).read_bytes()
            if a != b:
                raise ValueError(f"cold/resumed {kind} differs at turn {turn}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["models", "binaries", "workloads", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--source", required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    receipts = []

    def status(**fields):
        save(a.out / "status.json", {
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields,
        })
        print(json.dumps(fields), flush=True)

    def run(family, label, workload, maximum, new_tokens, cold=False):
        root = a.out / f"{family}-{label}-{'cold' if cold else 'warm'}"
        schedule = root.with_suffix(".schedule.json")
        depths = root.with_suffix(".depths.json")
        mapping = {f"k{k}": k for k in range(1, maximum + 1)}
        order = ["native"] if cold else ["native", "measured", "fixed", "learned", *mapping]
        save(schedule, [{"cycle": 0, "order": order}])
        save(depths, mapping)
        model = a.models / family
        seed = 20265100 if family == "qwen" else 20265200
        cmd = [
            sys.executable, str(LANE / "run_study.py"), "--family", family,
            "--binary", str(a.binaries / ("mtp-depth-study" if family == "qwen" else "gemma-depth-study")),
            "--target", str(model / "target.gguf"), "--workload", str(workload),
            "--out", str(root), "--lock", "/tmp/memra-gpu.lock",
            "--max-new", str(new_tokens), "--ctx", "49152", "--seed", str(seed),
            "--artifact-manifest", str(model / "artifacts.lock.json"),
            "--source-commit", a.source, "--schedule", str(schedule),
            "--fixed-depths", str(depths), "--gate",
        ]
        if family == "gemma":
            cmd += ["--draft", str(model / "assistant.gguf")]
        if cold:
            cmd += ["--cold-reference"]
        status(state="running", family=family, gate=root.name)
        with root.with_suffix(".driver.log").open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        return root, seed

    try:
        for family, maximum in [("qwen", 7), ("gemma", 5)]:
            for label, workload, tokens in [
                ("short", a.workloads / f"{family}-short.txt", 64),
                ("long", a.workloads / f"{family}-calibration-a.txt", 2048),
            ]:
                cold, seed = run(family, label, workload, maximum, tokens, cold=True)
                warm, _ = run(family, label, workload, maximum, tokens)
                compare(cold, warm, seed)
                receipts.append({"family": family, "shape": label, "cold": cold.name, "warm": warm.name})
                save(a.out / "passed.json", receipts)
        status(state="passed", checks=receipts)
    except BaseException as exc:
        status(state="failed", error_type=type(exc).__name__, error=str(exc), passed=receipts)
        raise


if __name__ == "__main__":
    main()
