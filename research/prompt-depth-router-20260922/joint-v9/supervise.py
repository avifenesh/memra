"""Advance frozen native v9 stages after the training collector exits."""

import json
import os
from pathlib import Path
import select
import subprocess
import sys
import traceback


BASE = Path("/home/ubuntu/mtp-v9")
SCRIPTS = BASE / "joint-v9"
RESULTS = BASE / "results"
MODEL = BASE / "Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
BINARY = BASE / "mtp-v9-binary-20260924"
WORKLOADS = BASE / "workloads"
OLD = BASE / "old-training-v2"
NEW = BASE / "new-training"
K_MODELS = BASE / "k-models"
CD_MODELS = BASE / "cd-models"
ARMS = BASE / "arms"
SELECTED = BASE / "selected"
PYTHON = BASE / "venv/bin/python"


def run(log, name, *command):
    print(json.dumps({"stage": name, "argv": command}), file=log, flush=True)
    result = subprocess.run(command, cwd=BASE, stdout=log, stderr=log)
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    print(json.dumps({"stage": name, "status": "complete"}), file=log, flush=True)


def wait_for_collector():
    pid = int((BASE / "training.pid").read_text())
    proc = Path(f"/proc/{pid}/cmdline")
    if proc.exists():
        args = proc.read_bytes().split(b"\0")
        if b"joint-v9/collect.py" not in args:
            raise ValueError("training PID no longer identifies our collector")
        fd = os.pidfd_open(pid)
        try:
            watcher = select.poll()
            watcher.register(fd, select.POLLIN)
            watcher.poll()
        finally:
            os.close(fd)
    results = list(RESULTS.glob("training-*.result.json"))
    if len(results) != 24 * 6:
        raise ValueError(
            f"training collector ended with {len(results)} of 144 arms"
        )


def pipeline(log):
    wait_for_collector()
    print(json.dumps({"stage": "training", "status": "144-arms-complete"}), file=log, flush=True)
    run(log, "new-rows", sys.executable, str(SCRIPTS / "new_rows.py"),
        "--root", str(RESULTS), "--out", str(NEW))
    run(log, "fit-k", str(PYTHON), str(SCRIPTS / "fit_k_new.py"),
        "--new", str(NEW), "--old", str(OLD), "--out", str(K_MODELS))
    run(log, "fit-cd", str(PYTHON), str(SCRIPTS / "fit_cd.py"),
        "--new", str(NEW), "--old", str(OLD), "--native", str(RESULTS),
        "--out", str(CD_MODELS))
    run(log, "freeze-candidates", sys.executable, str(SCRIPTS / "candidate_arms.py"),
        "--new", str(NEW), "--k-models", str(K_MODELS),
        "--cd-models", str(CD_MODELS), "--out", str(ARMS))
    common = (
        "--binary", str(BINARY), "--model", str(MODEL),
        "--workloads", str(WORKLOADS), "--out", str(RESULTS),
    )
    for phase in ("qualification", "validation"):
        run(log, f"{phase}-native", sys.executable, str(SCRIPTS / "eval.py"),
            *common, "--arms", str(ARMS / f"{phase}-arms.json"),
            "--phase", phase)
        if phase == "qualification":
            run(log, "qualification-noop-and-c", sys.executable,
                str(SCRIPTS / "qualify.py"), "--root", str(RESULTS),
                "--arms", str(ARMS / "qualification-arms.json"),
                "--out", str(BASE / "qualification-result.json"))
        run(log, f"{phase}-quality", sys.executable, str(SCRIPTS / "quality.py"),
            "--root", str(RESULTS), "--manifest", str(WORKLOADS / "manifest.json"),
            "--out", str(BASE / f"{phase}-quality.json"),
            "--phases", phase)
    run(log, "validation-score-and-freeze", sys.executable,
        str(SCRIPTS / "score.py"), "--root", str(RESULTS),
        "--quality", str(BASE / "validation-quality.json"),
        "--arms", str(ARMS / "validation-arms.json"),
        "--phase", "validation", "--out", str(BASE / "validation-score.json"),
        "--selection-out", str(SELECTED))
    run(log, "heldout-native", sys.executable, str(SCRIPTS / "eval.py"),
        *common, "--arms", str(SELECTED / "heldout-arms.json"),
        "--phase", "heldout")
    run(log, "heldout-quality", sys.executable, str(SCRIPTS / "quality.py"),
        "--root", str(RESULTS), "--manifest", str(WORKLOADS / "manifest.json"),
        "--out", str(BASE / "heldout-quality.json"),
        "--phases", "heldout")
    run(log, "heldout-score", sys.executable, str(SCRIPTS / "score.py"),
        "--root", str(RESULTS), "--quality", str(BASE / "heldout-quality.json"),
        "--arms", str(SELECTED / "heldout-arms.json"), "--phase", "heldout",
        "--out", str(BASE / "heldout-score.json"))
    run(log, "seal", sys.executable, str(SCRIPTS / "seal.py"),
        "--base", str(BASE), "--out", str(BASE / "sealed-v9"))
    (BASE / "pipeline-complete.json").write_text(json.dumps({
        "status": "complete",
        "stages": ["training", "fit", "qualification", "validation", "heldout", "seal"],
    }, sort_keys=True) + "\n")


def main():
    if os.environ.get("MEMRA_CAPTURE_DIR"):
        raise ValueError("research VM cannot carry customer capture")
    with (BASE / "supervisor.log").open("x") as log:
        try:
            pipeline(log)
        except Exception as error:
            traceback.print_exc(file=log)
            (BASE / "pipeline-failed.json").write_text(json.dumps({
                "status": "failed", "reason": str(error),
            }, sort_keys=True) + "\n")
            raise


if __name__ == "__main__":
    main()
