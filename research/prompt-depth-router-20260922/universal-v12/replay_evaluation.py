"""Recompute mixed quality, native rate, and selection from archived bytes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "joint-v11"))
import training_replay as prior_replay
import score_prose
import score_validation
import select_shared
import score_final
import seal_evaluation


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def equal(actual, path, kind):
    if actual != json.loads(path.read_text()):
        raise ValueError(f"archived mixed {kind} score changed on replay")


def replay(archive, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["schema"] != 1
        or manifest["scope"]
        != "one shared C/K/D mixed-domain native and quality result"
        or manifest["model_sha256"] != seal_evaluation.MODEL_SHA
        or manifest["binary_sha256"] != seal_evaluation.BINARY_SHA
        or manifest["training_workloads_sha256"]
        != seal_evaluation.TRAIN_SHA
        or manifest["validation_workloads_sha256"]
        != seal_evaluation.VALIDATION_SHA
        or manifest["source_full_manifest_sha256"]
        != seal_evaluation.FULL_SHA
        or sha(archive) != manifest["archive_sha256"]
    ):
        raise ValueError("mixed evaluation archive manifest differs")
    selected = manifest["selection_status"] == "selected"
    if manifest["selection_status"] not in (
        "selected", "global-no-go",
    ):
        raise ValueError("mixed selection status differs")
    with tempfile.TemporaryDirectory(
        prefix="mtp-v12-evaluation-replay-",
    ) as folder:
        root = Path(folder)
        members = prior_replay.extract(archive, manifest, root)
        source = root / "source/universal-v12"
        for name in (
            "score_prose.py", "score_validation.py", "select_shared.py",
            "score_final.py", "seal_evaluation.py",
            "replay_evaluation.py",
        ):
            if sha(source / name) != sha(BASE / "universal-v12" / name):
                raise ValueError("mixed score replay source differs")
        native = root / "native/eval-results"
        arms = root / "models/policy-arms"
        inputs = root / "inputs"
        quality = root / "quality"
        config = inputs / "judge-config.json"
        validation_prose = score_prose.score(
            quality / "validation-packets",
            quality / "validation-judge",
            config,
        )
        equal(
            validation_prose,
            inputs / "validation-prose-quality.json",
            "validation prose",
        )
        validation = score_validation.score(SimpleNamespace(
            root=native,
            workloads=inputs / "phase-validation",
            arms=arms / "validation-arms.json",
            tasks=inputs / "validation-task-quality.json",
            prose=inputs / "validation-prose-quality.json",
        ))
        equal(
            validation, arms / "validation-score.json",
            "validation native",
        )
        chosen, final_arms = select_shared.choose(
            arms / "validation-score.json",
            arms / "validation-arms.json",
        )
        equal(chosen, arms / "shared-selected.json", "shared selection")
        if (
            sha(arms / "shared-selected.json")
            != manifest["selection_sha256"]
            or sha(arms / "validation-score.json")
            != manifest["validation_score_sha256"]
            or chosen["status"] != manifest["selection_status"]
        ):
            raise ValueError("archived shared selection lineage differs")
        if selected:
            frozen = json.loads(
                (arms / "final-arms.json").read_text()
            )
            if (
                frozen["selected_from_validation"]
                != manifest["selection_sha256"]
                or frozen["arms"] != final_arms
            ):
                raise ValueError("archived final arm choice differs")
            final_prose = score_prose.score(
                quality / "final-packets",
                quality / "final-judge",
                config,
                quality / "validation-judge/manifest.json",
            )
            equal(
                final_prose, inputs / "final-prose-quality.json",
                "final prose",
            )
            final = score_final.score(SimpleNamespace(
                root=native,
                workloads=inputs / "phase-final",
                arms=arms / "final-arms.json",
                tasks=inputs / "final-task-quality.json",
                prose=inputs / "final-prose-quality.json",
            ))
            equal(final, inputs / "final-score.json", "final native")
            if sha(inputs / "final-score.json") != (
                manifest["final_score_sha256"]
            ):
                raise ValueError("archived final verdict changed")
            verdict = final["status"]
        else:
            if (
                final_arms
                or manifest["final_score_sha256"] is not None
                or (inputs / "phase-final").exists()
            ):
                raise ValueError("archived no-go leaked final phase")
            verdict = "global-no-go"
        native_count = len(list(native.glob("*.result.json")))
        if native_count != manifest["expected_native_sessions"]:
            raise ValueError("archived native session count changed")
    return {
        "schema": 1,
        "status": "mixed-quality-score-and-selection-replay-match",
        "archive_sha256": manifest["archive_sha256"],
        "selection_status": manifest["selection_status"],
        "verdict": verdict,
        "native_sessions": native_count,
        "members": members,
        "validation_score_sha256":
        manifest["validation_score_sha256"],
        "final_score_sha256": manifest["final_score_sha256"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = replay(
        args.archive.resolve(), args.manifest.resolve(),
    )
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({
        "status": result["status"], "verdict": result["verdict"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
