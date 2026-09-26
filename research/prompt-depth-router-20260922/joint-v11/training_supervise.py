"""Run the pinned v11 randomized K/D/C training battery on a research GPU."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import traceback


BASE = Path(__file__).resolve().parent.parent
SCRIPTS = BASE / "joint-v11"
MODEL = BASE / "Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
BINARY = BASE / "mtp-depth-study"
WORKLOADS = BASE / "workloads"
RESULTS = BASE / "training-results"
VALIDATION_SHA = "e8e60712c7053d9f64eba42b3f9a7b04ad946971fe095364cabb163b6c73eacd"
FINAL_SHA = "71c538295aeb970856f5feb5d11c888aa2dea4d220f41ab844d6e0587108a3df"


def stage(log, name, *argv, directory=SCRIPTS):
    print(json.dumps({"stage": name, "argv": list(map(str, argv))}),
          file=log, flush=True)
    command = [sys.executable, str(directory / f"{name}.py"),
               *map(str, argv)]
    result = subprocess.run(command, cwd=BASE, stdout=log, stderr=log)
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    print(json.dumps({"stage": name, "status": "complete"}),
          file=log, flush=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def await_phase(log, name, expected):
    target = BASE / (
        "validation-workloads" if name == "validation"
        else "final-workloads"
    )
    ready = BASE / f"{name}-ready.json"
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    fd = libc.inotify_init1(0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "phase inotify_init1")
    if libc.inotify_add_watch(
        fd, os.fsencode(BASE), 0x8 | 0x80 | 0x100
    ) < 0:
        raise OSError(ctypes.get_errno(), "phase inotify_add_watch")
    events = select.poll()
    events.register(fd, select.POLLIN)
    (BASE / f"needs-{name}.json").write_text(
        json.dumps({"phase": name, "manifest_sha256": expected},
                   sort_keys=True) + "\n"
    )
    print(json.dumps({"stage": f"await-{name}"}), file=log, flush=True)
    deadline = time.monotonic() + 10 * 60
    try:
        while True:
            if ready.exists():
                receipt = json.loads(ready.read_text())
                if (
                    receipt["phase"] != name
                    or receipt["manifest_sha256"] != expected
                    or sha(target / "manifest.json") != expected
                ):
                    raise ValueError(f"{name} release receipt differs")
                print(json.dumps({
                    "stage": f"await-{name}", "status": "complete",
                }), file=log, flush=True)
                return target
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not events.poll(int(remaining * 1000)):
                raise TimeoutError(f"{name} workload release stalled")
            os.read(fd, 4096)
    finally:
        os.close(fd)


def pipeline(log):
    if os.environ.get("MEMRA_CAPTURE_DIR"):
        raise ValueError("training rental cannot hold customer capture")
    os.environ["NLTK_DATA"] = str(BASE / "transfer-inputs/nltk_data")
    stage(log, "training_runmeta", "--base", BASE, directory=BASE / "ops")
    stage(log, "collect", "--binary", BINARY, "--model", MODEL,
          "--workloads", WORKLOADS, "--out", RESULTS,
          "--start", 0, "--stop", 32)
    stage(log, "training_seal", "--base", BASE,
          "--out", BASE / "training-sealed")
    stage(log, "training_replay",
          "--archive", BASE / "training-sealed/native-data.tar.gz",
          "--manifest", BASE / "training-sealed/manifest.json",
          "--out", BASE / "training-replay.json")
    stage(log, "measurement_rows",
          "--v10-archive", BASE / "v10-parent/native-data.tar.gz",
          "--v10-manifest", BASE / "v10-parent/manifest.json",
          "--v10-custody", BASE / "v10-parent/custody.json",
          "--v11-archive", BASE / "training-sealed/native-data.tar.gz",
          "--v11-manifest", BASE / "training-sealed/manifest.json",
          "--v11-replay", BASE / "training-replay.json",
          "--out", BASE / "training-rows")
    stage(log, "feature_audit", "--rows", BASE / "training-rows",
          "--out", BASE / "feature-audit.json")
    stage(log, "preflight_visibility",
          "--archive", BASE / "training-sealed/native-data.tar.gz",
          "--manifest", BASE / "training-sealed/manifest.json",
          "--replay", BASE / "training-replay.json",
          "--out", BASE / "prefix-preflight.json")
    stage(log, "fit_mixed",
          "--v9-new", BASE / "code-training/new",
          "--v9-old", BASE / "code-training/old",
          "--v9-k-manifest", BASE / "code-training/k-models/manifest.json",
          "--v9-table", BASE / "code-training/table.tsv",
          "--v11-rows", BASE / "training-rows",
          "--out", BASE / "policy-models")
    stage(log, "arms", "--models", BASE / "policy-models",
          "--training-rows", BASE / "training-rows",
          "--visibility-preflight", BASE / "prefix-preflight.json",
          "--out", BASE / "policy-arms")
    validation_workloads = await_phase(
        log, "validation", VALIDATION_SHA
    )
    common = (
        "--binary", BINARY, "--model", MODEL,
        "--out", BASE / "eval-results",
    )
    for phase in ("qualification", "validation"):
        arm_path = BASE / f"policy-arms/{phase}-arms.json"
        stage(log, "eval", *common,
              "--workloads", validation_workloads,
              "--arms", arm_path,
              "--phase", phase)
        stage(log, "quality", "--root", BASE / "eval-results",
              "--manifest", validation_workloads / "manifest.json",
              "--arms", arm_path,
              "--ifeval-source",
              BASE / "transfer-inputs/third_party/instruction_following_eval",
              "--phase", phase,
              "--out", BASE / f"{phase}-quality.json")
    stage(log, "score", "--root", BASE / "eval-results",
          "--quality", BASE / "validation-quality.json",
          "--arms", BASE / "policy-arms/validation-arms.json",
          "--out", BASE / "policy-arms/validation-score.json")
    stage(log, "visibility",
          "--training-root", RESULTS,
          "--validation-root", BASE / "eval-results",
          "--workloads", validation_workloads,
          "--out", BASE / "visibility-validation.json")
    stage(log, "select",
          "--score", BASE / "policy-arms/validation-score.json",
          "--arms", BASE / "policy-arms/validation-arms.json",
          "--qualifier", BASE / "qualification-result.json")
    selection = json.loads((BASE / "policy-arms/selected.json").read_text())
    if selection["status"] == "selected":
        final_workloads = await_phase(log, "heldout", FINAL_SHA)
        final_arms = BASE / "policy-arms/heldout-arms.json"
        stage(log, "eval", *common,
              "--workloads", final_workloads,
              "--arms", final_arms,
              "--phase", "heldout")
        stage(log, "quality", "--root", BASE / "eval-results",
              "--manifest", final_workloads / "manifest.json",
              "--arms", final_arms,
              "--ifeval-source",
              BASE / "transfer-inputs/third_party/instruction_following_eval",
              "--phase", "heldout",
              "--out", BASE / "heldout-quality.json")
        stage(log, "score", "--root", BASE / "eval-results",
              "--quality", BASE / "heldout-quality.json",
              "--arms", final_arms,
              "--out", BASE / "heldout-score.json")
    elif selection["status"] != "no-go":
        raise ValueError("v11 validation selection status differs")
    stage(log, "evaluation_seal", "--base", BASE,
          "--out", BASE / "evaluation-sealed")
    stage(log, "evaluation_replay",
          "--archive", BASE / "evaluation-sealed/native-data.tar.gz",
          "--manifest", BASE / "evaluation-sealed/manifest.json",
          "--out", BASE / "evaluation-replay.json")
    (BASE / "pipeline-complete.json").write_text(json.dumps({
        "schema": 1, "status": "complete",
        "stages": ["training_runmeta", "collect",
                   "training_seal", "training_replay", "measurement_rows",
                   "feature_audit", "preflight_visibility", "fit_mixed",
                   "arms", "qualification", "validation", "select",
                   "heldout-if-selected", "evaluation_seal",
                   "evaluation_replay"],
    }, sort_keys=True) + "\n")


def main():
    with (BASE / "supervisor.log").open("x") as log:
        try:
            pipeline(log)
        except Exception as error:
            traceback.print_exc(file=log)
            (BASE / "pipeline-failed.json").write_text(json.dumps({
                "schema": 1, "status": "failed", "reason": str(error),
            }, sort_keys=True) + "\n")
            raise


if __name__ == "__main__":
    main()
