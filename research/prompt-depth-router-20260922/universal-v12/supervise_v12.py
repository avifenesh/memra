"""Run one-policy Qwen C/K/D phases without releasing later prompts early."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import traceback


TRAIN_SHA = "2bca403c1edff44a1d707abd60710767f9dbbe2a63721757032b0fbe223c5e00"
VALIDATION_SHA = "bf920b82e0176c4304bcc562ccce34a28384082c888355a280620163a45e5313"
FULL_SHA = "b35374d34e2a28b31cca0399e6394fa3db3b0f27e9d593b17032a7856b2a5477"
JUDGE_SHA = "624cbb8478326ec7662d6e5aaa959e713cb3bf0330128dd42a7e0dc9b8a05bdd"
TEMPLATE_SHA = "ccd57bd8c4c73f4f83cf8963ef3c2697c1c7b9e907ead91e0d0512cca4ae7a11"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def stage(log, root, script, *options):
    print(json.dumps({
        "stage": script, "args": list(map(str, options)),
    }), file=log, flush=True)
    command = [
        sys.executable, str(root / script), *map(str, options),
    ]
    result = subprocess.run(
        command, cwd=root.parent, stdout=log, stderr=log,
    )
    if result.returncode:
        raise RuntimeError(f"{script} failed with {result.returncode}")
    print(json.dumps({
        "stage": script, "status": "complete",
    }), file=log, flush=True)


def await_phase(base, name, expected, marker):
    target = base / f"phase-{name}"
    request = base / f"needs-{name}.json"
    ready = base / f"{name}-ready.json"
    save(request, marker)
    library = ctypes.CDLL("libc.so.6", use_errno=True)
    fd = library.inotify_init1(0)
    if fd < 0 or library.inotify_add_watch(
        fd, os.fsencode(base), 0x8 | 0x80 | 0x100,
    ) < 0:
        raise OSError(ctypes.get_errno(), "phase inotify")
    events = select.poll()
    events.register(fd, select.POLLIN)
    try:
        while True:
            if ready.exists():
                receipt = json.loads(ready.read_text())
                if (
                    receipt["schema"] != 1
                    or receipt["phase"] != name
                    or receipt["manifest_sha256"] != expected
                    or receipt["request_sha256"] != sha(request)
                    or sha(target / "manifest.json") != expected
                ):
                    raise ValueError("phase release marker differs")
                return target
            if not events.poll(10 * 60 * 1000):
                raise TimeoutError(f"phase {name} release stalled")
            os.read(fd, 4096)
    finally:
        os.close(fd)


def pipeline(base, credential_file, log):
    if os.environ.get("MEMRA_CAPTURE_DIR"):
        raise ValueError("universal research host cannot capture customers")
    model = base / "Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
    binary = base / "mtp-depth-study"
    evaluation = base / "eval-results"
    v11 = base / "joint-v11"
    scripts = base / "universal-v12"
    parents = base / "parents"
    code = base / "code-training"
    v11_rows = base / "v11-rows"
    prose_rows = base / "prose-rows"
    models = base / "policy-models"
    arms = base / "policy-arms"
    judge_config = base / "judge-config.json"
    template = base / "wildbench-pairwise-template.md"
    if (
        sha(judge_config) != JUDGE_SHA
        or sha(template) != TEMPLATE_SHA
        or sha(base / "phase-training/manifest.json") != TRAIN_SHA
    ):
        raise ValueError("universal training judge or phase source changed")
    from judge_bedrock import credential, price_quote, validate_config
    from quality_tasks import sandbox_preflight
    config = json.loads(judge_config.read_text())
    validate_config(config)
    credential(credential_file)
    price_quote(config)
    sandbox_preflight()
    stage(log, base / "ops", "run_meta_v12.py",
          "--base", base)
    stage(log, scripts, "pilot_depth.py",
          "--binary", binary, "--model", model,
          "--workloads", base / "phase-training",
          "--run-meta", base / "run-meta.json",
          "--out", base / "pilot-results",
          "--receipt", base / "pilot-result.json")
    stage(log, v11, "prepare_code_training.py",
          "--archive", parents / "v9/native-data.tar.gz",
          "--manifest", parents / "v9/manifest.json",
          "--out", code)
    stage(log, v11, "measurement_rows.py",
          "--v10-archive", parents / "v10/native-data.tar.gz",
          "--v10-manifest", parents / "v10/manifest.json",
          "--v10-custody", parents / "v10/custody.json",
          "--v11-archive", parents / "v11/native-data.tar.gz",
          "--v11-manifest", parents / "v11/manifest.json",
          "--v11-replay", parents / "v11/training-replay.json",
          "--out", v11_rows)
    stage(log, scripts, "collect_prose.py",
          "--binary", binary, "--model", model,
          "--run-meta", base / "run-meta.json",
          "--workloads", base / "phase-training",
          "--out", base / "training-prose-results")
    stage(log, scripts, "seal_training.py",
          "--base", base, "--out", base / "training-sealed")
    stage(log, scripts, "replay_training.py",
          "--archive", base / "training-sealed/native-data.tar.gz",
          "--manifest", base / "training-sealed/manifest.json",
          "--rows-out", prose_rows,
          "--out", base / "prose-replay.json")
    training = (
        "--v9-new", code / "new",
        "--v9-old", code / "old",
        "--v9-k-manifest", code / "k-models/manifest.json",
        "--v9-table", code / "table.tsv",
        "--v11-rows", v11_rows,
        "--prose-rows", prose_rows,
        "--prose-replay", base / "prose-replay.json",
    )
    stage(log, scripts, "preflight_prefix.py", *training,
          "--out", base / "prefix-preflight.json")
    stage(log, scripts, "fit_shared.py", *training,
          "--out", models)
    stage(log, scripts, "arms_shared.py",
          "--models", models,
          "--v11-rows", v11_rows,
          "--prose-rows", prose_rows,
          "--prose-replay", base / "prose-replay.json",
          "--preflight", base / "prefix-preflight.json",
          "--out", arms)
    native = (
        "--binary", binary, "--model", model,
        "--models", models, "--out", evaluation,
        "--run-meta", base / "run-meta.json",
    )
    stage(log, scripts, "eval.py", *native,
          "--workloads", base / "phase-training",
          "--arms", arms / "qualification-arms.json",
          "--phase", "qualification")
    request = {
        "schema": 1, "phase": "validation",
        "manifest_sha256": VALIDATION_SHA,
        "training_manifest_sha256": TRAIN_SHA,
        "source_full_manifest_sha256": FULL_SHA,
        "model_manifest_sha256": sha(models / "manifest.json"),
        "qualification_sha256":
        sha(base / "qualification-result.json"),
    }
    validation = await_phase(
        base, "validation", VALIDATION_SHA, request,
    )
    stage(log, scripts, "eval.py", *native,
          "--workloads", validation,
          "--training-manifest", base / "phase-training/manifest.json",
          "--arms", arms / "validation-arms.json",
          "--phase", "validation")
    stage(log, scripts, "quality_tasks.py",
          "--root", evaluation, "--workloads", validation,
          "--arms", arms / "validation-arms.json",
          "--phase", "validation",
          "--out", base / "validation-task-quality.json")
    stage(log, scripts, "prose_packets.py",
          "--root", evaluation, "--workloads", validation,
          "--arms", arms / "validation-arms.json",
          "--phase", "validation",
          "--template", template,
          "--judge-config", judge_config,
          "--out", base / "validation-packets")
    stage(log, scripts, "judge_bedrock.py",
          "--packets", base / "validation-packets",
          "--config", judge_config,
          "--credential-file", credential_file,
          "--out", base / "validation-judge")
    stage(log, scripts, "score_prose.py",
          "--packets", base / "validation-packets",
          "--results", base / "validation-judge",
          "--config", judge_config,
          "--out", base / "validation-prose-quality.json")
    stage(log, scripts, "score_validation.py",
          "--root", evaluation, "--workloads", validation,
          "--arms", arms / "validation-arms.json",
          "--tasks", base / "validation-task-quality.json",
          "--prose", base / "validation-prose-quality.json",
          "--run-meta", base / "run-meta.json",
          "--out", arms / "validation-score.json")
    stage(log, scripts, "select_shared.py",
          "--validation", arms / "validation-score.json",
          "--arms", arms / "validation-arms.json")
    selected = json.loads(
        (arms / "shared-selected.json").read_text()
    )
    if selected["status"] == "selected":
        request = {
            "schema": 1, "phase": "final",
            "manifest_sha256": FULL_SHA,
            "training_manifest_sha256": TRAIN_SHA,
            "source_full_manifest_sha256": FULL_SHA,
            "model_manifest_sha256": sha(models / "manifest.json"),
            "selection_sha256": sha(arms / "shared-selected.json"),
        }
        final = await_phase(base, "final", FULL_SHA, request)
        stage(log, scripts, "eval.py", *native,
              "--workloads", final,
              "--training-manifest", base / "phase-training/manifest.json",
              "--validation-manifest", validation / "manifest.json",
              "--arms", arms / "final-arms.json",
              "--phase", "final")
        stage(log, scripts, "quality_tasks.py",
              "--root", evaluation, "--workloads", final,
              "--arms", arms / "final-arms.json",
              "--phase", "final",
              "--out", base / "final-task-quality.json")
        stage(log, scripts, "prose_packets.py",
              "--root", evaluation, "--workloads", final,
              "--arms", arms / "final-arms.json",
              "--phase", "final",
              "--template", template,
              "--judge-config", judge_config,
              "--out", base / "final-packets")
        stage(log, scripts, "judge_bedrock.py",
              "--packets", base / "final-packets",
              "--config", judge_config,
              "--credential-file", credential_file,
              "--prior-manifest",
              base / "validation-judge/manifest.json",
              "--out", base / "final-judge")
        stage(log, scripts, "score_prose.py",
              "--packets", base / "final-packets",
              "--results", base / "final-judge",
              "--config", judge_config,
              "--prior-manifest",
              base / "validation-judge/manifest.json",
              "--out", base / "final-prose-quality.json")
        stage(log, scripts, "score_final.py",
              "--root", evaluation, "--workloads", final,
              "--arms", arms / "final-arms.json",
              "--tasks", base / "final-task-quality.json",
              "--prose", base / "final-prose-quality.json",
              "--run-meta", base / "run-meta.json",
              "--out", base / "final-score.json")
        verdict = json.loads(
            (base / "final-score.json").read_text()
        )["status"]
    elif selected["status"] == "global-no-go":
        verdict = "global-no-go"
    else:
        raise ValueError("mixed validation selection status differs")
    stage(log, scripts, "seal_evaluation.py",
          "--base", base, "--out", base / "evaluation-sealed")
    stage(log, scripts, "replay_evaluation.py",
          "--archive", base / "evaluation-sealed/native-data.tar.gz",
          "--manifest", base / "evaluation-sealed/manifest.json",
          "--out", base / "evaluation-replay.json")
    replay = json.loads(
        (base / "evaluation-replay.json").read_text()
    )
    if (
        replay["status"]
        != "mixed-quality-score-and-selection-replay-match"
        or replay["verdict"] != verdict
    ):
        raise ValueError("one-policy evaluation archive did not replay")
    save(base / "pipeline-complete.json", {
        "schema": 1, "status": "complete",
        "selection_status": selected["status"],
        "verdict": verdict,
        "training_archive_sha256":
        sha(base / "training-sealed/native-data.tar.gz"),
        "evaluation_archive_sha256":
        sha(base / "evaluation-sealed/native-data.tar.gz"),
        "model_manifest_sha256":
        sha(models / "manifest.json"),
        "validation_score_sha256":
        sha(arms / "validation-score.json"),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--credential-file", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    with (base / "supervisor.pid").open("x") as handle:
        handle.write(str(os.getpid()) + "\n")
    with (base / "supervisor.log").open("x") as log:
        try:
            pipeline(base, args.credential_file.resolve(), log)
        except Exception as error:
            traceback.print_exc(file=log)
            save(base / "pipeline-failed.json", {
                "schema": 1, "status": "failed",
                "reason": str(error),
            })
            raise


if __name__ == "__main__":
    main()
