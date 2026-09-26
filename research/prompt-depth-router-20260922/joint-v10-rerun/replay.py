"""Verify sealed non-code native, quality and score receipts."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile
from types import SimpleNamespace


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def extract(archive, manifest, target):
    seen = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            name = member.name
            logical = PurePosixPath(name)
            if (
                name not in manifest["members"] or name in seen
                or not member.isfile() or logical.is_absolute()
                or ".." in logical.parts or str(logical) != name
                or member.size != manifest["members"][name]["bytes"]
            ):
                raise ValueError(f"unsafe or changed transfer member: {name}")
            seen.add(name)
            path = target.joinpath(*logical.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            value = hashlib.sha256()
            with source.extractfile(member) as input_file, path.open("xb") as output:
                while chunk := input_file.read(1024 * 1024):
                    value.update(chunk)
                    output.write(chunk)
            if value.hexdigest() != manifest["members"][name]["sha256"]:
                raise ValueError(f"transfer member hash differs: {name}")
    if seen != set(manifest["members"]):
        raise ValueError("sealed transfer archive omits members")
    return len(seen)


def replay(archive, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != 1 or sha(archive) != manifest["archive_sha256"]:
        raise ValueError("sealed transfer archive differs")
    with tempfile.TemporaryDirectory(prefix="mtp-v10-replay-") as temporary:
        root = Path(temporary)
        members = extract(archive, manifest, root)
        inputs = root / "inputs/workloads"
        native = root / "native/results"
        if sha(inputs / "manifest.json") != manifest["workloads_sha256"]:
            raise ValueError("non-code workload manifest differs")
        if sha(root / "inputs/datasets/ifeval.jsonl") != manifest["ifeval_source_sha256"]:
            raise ValueError("IFEval source data differs")
        if sha(root / "inputs/datasets/gsm8k-test.jsonl") != manifest["gsm8k_source_sha256"]:
            raise ValueError("GSM8K source data differs")
        tasks = json.loads((inputs / "manifest.json").read_text())
        arms_path = root / "native/arms.json"
        arms = json.loads(arms_path.read_text())
        sys.path.insert(0, str(root / "source/joint-v9"))
        sys.path.insert(0, str(root / "source/joint-v10-rerun"))
        os.environ["NLTK_DATA"] = str(root / "nltk_data")
        import run_native
        import quality
        import score
        import source_exactness

        source_path = root / "source-proof/runtime-source-joint-v8.tar.gz"
        source_manifest_path = root / "source-proof/manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text())
        source_proof = json.loads(
            (root / "native/source-exactness.json").read_text()
        )
        run_meta = json.loads((root / "native/run-meta.json").read_text())
        if (
            sha(source_path) != source_exactness.SOURCE_SHA
            or sha(source_manifest_path) != source_exactness.SOURCE_MANIFEST_SHA
            or source_manifest["source_archive_sha256"]
            != source_exactness.SOURCE_SHA
            or source_manifest["binary_sha256"] != manifest["binary_sha256"]
            or source_manifest["model_sha256"] != manifest["model_sha256"]
            or run_meta["binary_sha256"] != manifest["binary_sha256"]
            or run_meta["model_sha256"] != manifest["model_sha256"]
            or run_meta["workloads_sha256"] != manifest["workloads_sha256"]
            or run_meta["source_exactness_sha256"]
            != sha(root / "native/source-exactness.json")
            or run_meta["ops_source_sha256"]
            != sha(root / "source/private_ops/make_run_meta.py")
            or source_proof["status"]
            != "exact-source-retains-sampled-pick-before-C-stop"
            or source_proof["binary_sha256"] != manifest["binary_sha256"]
            or source_proof["sampled_branches"]
            != source_exactness.source_order(source_exactness.source_lines(source_path))
            or source_proof["two_token_counterexample"]
            != source_exactness.two_token_example()
        ):
            raise ValueError("sampled C exact-source receipt replay differs")

        for phase, count in (("qualification", 1), ("heldout", 16)):
            for domain in ("ifeval", "gsm8k"):
                entries = tasks["groups"][phase][domain]
                if len(entries) != count:
                    raise ValueError("non-code domain split differs")
                for index, entry in enumerate(entries):
                    if sha(inputs / entry["file"]) != entry["sha256"]:
                        raise ValueError("frozen task text differs")
                    for arm in arms["arms"]:
                        name = f"{phase}-{domain}-{index}-{arm['label']}"
                        row = json.loads((native / f"{name}.result.json").read_text())
                        observed = run_native.v9_eval.verify(
                            native / name, entry, arm
                        )
                        if canonical(observed) != row:
                            raise ValueError(f"native transfer result differs: {name}")
        observed_qualifier = run_native.qualifier_result(
            SimpleNamespace(out=native, workloads=inputs, arms=arms_path),
            arms,
        )
        if canonical(observed_qualifier) != json.loads(
            (root / "native/qualification-result.json").read_text()
        ):
            raise ValueError("transfer qualifier replay differs")
        evaluator = quality.ifeval_grader(
            root / "third_party/instruction_following_eval"
        )
        for phase in ("qualification", "heldout"):
            observed = quality.score(native, tasks, arms, phase, evaluator)
            saved = json.loads((root / f"native/{phase}-quality.json").read_text())
            if canonical(observed) != saved:
                raise ValueError(f"{phase} task-quality replay differs")
            if phase == "heldout":
                scored = score.score(native, observed, arms)
                if canonical(scored) != json.loads(
                    (root / "native/heldout-score.json").read_text()
                ):
                    raise ValueError("non-code paired score replay differs")
    return {
        "status": "archive-native-IFEval-GSM8K-noops-and-E2E-match",
        "members": members,
        "qualification_arms": 20,
        "heldout_arms": 320,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.archive, args.manifest)
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
