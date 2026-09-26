"""Run non-code C/K/D transfer through qualification, scoring and sealing."""

import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


BASE = Path(__file__).resolve().parent.parent
SCRIPTS = BASE / "joint-v10"
RESULTS = BASE / "results"
WORKLOADS = BASE / "workloads"
ARMS = BASE / "arms.json"
MODEL = BASE / "Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
BINARY = BASE / "mtp-depth-study"
IFEVAL_SOURCE = BASE / "third_party/instruction_following_eval"
PYTHON = BASE / "venv/bin/python"


def run(log, name, *argv, python=sys.executable):
    print(json.dumps({"stage": name, "argv": list(map(str, argv))}), file=log, flush=True)
    command = [str(python), str(SCRIPTS / f"{name}.py"), *map(str, argv)]
    result = subprocess.run(command, cwd=BASE, stdout=log, stderr=log)
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    print(json.dumps({"stage": name, "status": "complete"}), file=log, flush=True)


def pipeline(log):
    os.environ["NLTK_DATA"] = str(BASE / "nltk_data")
    common = (
        "--binary", BINARY, "--model", MODEL,
        "--workloads", WORKLOADS, "--arms", ARMS,
        "--out", RESULTS,
    )
    run(log, "run_native", *common, "--phase", "qualification")
    run(log, "quality", "--root", RESULTS,
        "--manifest", WORKLOADS / "manifest.json", "--arms", ARMS,
        "--ifeval-source", IFEVAL_SOURCE, "--phase", "qualification",
        "--out", BASE / "qualification-quality.json", python=PYTHON)
    run(log, "run_native", *common, "--phase", "heldout")
    run(log, "quality", "--root", RESULTS,
        "--manifest", WORKLOADS / "manifest.json", "--arms", ARMS,
        "--ifeval-source", IFEVAL_SOURCE, "--phase", "heldout",
        "--out", BASE / "heldout-quality.json", python=PYTHON)
    run(log, "score", "--root", RESULTS,
        "--quality", BASE / "heldout-quality.json", "--arms", ARMS,
        "--out", BASE / "heldout-score.json")
    run(log, "seal", "--base", BASE, "--out", BASE / "sealed-v10")
    (BASE / "pipeline-complete.json").write_text(
        json.dumps({
            "status": "complete",
            "stages": [
                "qualification", "qualification-quality", "heldout",
                "heldout-quality", "score", "seal",
            ],
        }, sort_keys=True) + "\n"
    )


def main():
    if os.environ.get("MEMRA_CAPTURE_DIR"):
        raise ValueError("research VM cannot hold customer capture")
    with (BASE / "supervisor.log").open("x") as log:
        try:
            pipeline(log)
        except Exception as error:
            traceback.print_exc(file=log)
            (BASE / "pipeline-failed.json").write_text(
                json.dumps({"status": "failed", "reason": str(error)},
                           sort_keys=True) + "\n"
            )
            raise


if __name__ == "__main__":
    main()
