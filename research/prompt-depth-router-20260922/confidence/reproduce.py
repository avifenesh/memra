"""Re-audit every fixed-C native request from sealed public science records."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile

from verify_source import verify as verify_source
from render_fixed import render as render_fixed


ARCHIVES = {
    "native-data.tar.gz",
    "harness-source.tar.gz",
    "runtime-source-confidence.tar.gz",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unpack(archive, expected, out):
    seen = set()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            logical = PurePosixPath(member.name)
            if (not member.isfile() or logical.is_absolute() or ".." in logical.parts
                    or str(logical) != member.name or member.name in seen):
                raise ValueError("unsafe or duplicate scientific archive member")
            seen.add(member.name)
            if member.name not in expected or member.size != expected[member.name]["bytes"]:
                raise ValueError("scientific archive inventory changed")
            path = out.joinpath(*logical.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, path.open("xb") as target:
                shutil.copyfileobj(source, target)
            if sha(path) != expected[member.name]["sha256"]:
                raise ValueError("scientific archive member digest differs")
    if seen != set(expected):
        raise ValueError("scientific archive is incomplete")


def reproduce(candidate, manifest_sha256, base, out):
    if sha(candidate / "manifest.json") != manifest_sha256:
        raise ValueError("scientific manifest differs from its external pin")
    manifest = json.loads((candidate / "manifest.json").read_text())
    if manifest["schema"] != 1 or set(manifest["files"]) != ARCHIVES:
        raise ValueError("scientific archive set changed")
    for name, record in manifest["files"].items():
        path = candidate / name
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("scientific archive bytes differ: " + name)
    out.mkdir(parents=True, exist_ok=False)
    data = out / "data"
    data.mkdir()
    unpack(candidate / "native-data.tar.gz", manifest["native_members"], data)
    unpack(candidate / "harness-source.tar.gz", manifest["harness_members"], data)

    source = json.loads((data / "source-confidence.json").read_text())
    if (source["runtime_source_sha256"] != manifest["runtime_source_sha256"]
            or source["source_patcher_sha256"] != manifest["source_patch_sha256"]
            or source["base_source_recipe_commit"] != manifest["source_recipe_commit"]):
        raise ValueError("source identity differs from the scientific manifest")
    if sha(base) != source["base_runtime_source_sha256"]:
        raise ValueError("base source differs from its measured version")
    source_receipt = verify_source(
        base,
        candidate / "runtime-source-confidence.tar.gz",
        data / "source-patch.json",
        data / "source-confidence.json",
    )
    for name, expected in (
        ("harness/prefix/run.py", source["runner_sha256"]),
        ("harness/confidence/fixed_grid.py", source["grid_sha256"]),
        ("harness/confidence/report_fixed.py", source["reporter_sha256"]),
        ("harness/confidence/patch_source.py", source["source_patcher_sha256"]),
    ):
        if sha(data / name) != expected:
            raise ValueError("measured harness source differs: " + name)
    for name in manifest["harness_members"]:
        local = Path(__file__).resolve().parents[1] / name.removeprefix("harness/")
        if sha(local) != sha(data / name):
            raise ValueError("published harness differs from the measured source: " + name)

    repo = out / "repo"
    repo.mkdir()
    with tarfile.open(candidate / "runtime-source-confidence.tar.gz", "r:gz") as archive:
        archive.extractall(repo, filter="data")
    locked = json.loads((repo / "research/mtp-context-depth-20260921/qwen-artifacts.lock.json").read_text())
    if json.loads((data / "models/qwen/artifacts.lock.json").read_text()) != locked:
        raise ValueError("Qwen artifact lock differs from the measured source")

    context_dir = repo / "research/mtp-context-depth-20260921"
    sys.path.insert(0, str(context_dir))
    sys.path.insert(0, str(data / "harness/prefix"))
    sys.path.insert(0, str(data / "harness/confidence"))
    from audit import audit_run  # noqa: PLC0415
    from audit_context import audit_run as audit_context  # noqa: PLC0415
    from loop_audit import loop_candidate  # noqa: PLC0415
    from run_study import metrics_from_log  # noqa: PLC0415
    from report_fixed import report  # noqa: PLC0415

    root = data / "fixed-grid-v2"
    status = json.loads((root / "status.json").read_text())
    workloads = json.loads((data / "workloads/manifest.json").read_text())
    qwen = workloads["families"]["qwen"]
    checked = 0
    for relative in status["completed"]:
        family, phase, label = relative.split("/")
        if family != "qwen" or phase not in ("qualification", "scored"):
            raise ValueError("unexpected native record path")
        entry = (qwen["qualification"] if phase == "qualification"
                 else qwen["scenarios"][str(int(label.split("-")[0]))])
        run = root / relative
        saved = json.loads(run.with_name(run.name + ".audit.json").read_text())
        text = run.with_name(run.name + ".log").read_text()
        metrics = metrics_from_log(text)["fixed:3"]
        independent = audit_run(
            run, entry, "fixed:3", entry["seed"], metrics, audit_context, loop_candidate
        )
        if (saved["requests"] != independent["requests"]
                or saved["context_accounting"] != independent["context_accounting"]
                or saved["metrics"] != metrics):
            raise ValueError("native request audit differs: " + relative)
        checked += 1
    for name in ("oracle-v2-off.log", "oracle-v2-c030.log", "oracle-v2-c030zero.log"):
        if "=== SELF-CONSISTENCY PASS ===" not in (data / name).read_text():
            raise ValueError("target oracle failed: " + name)
    result = report(root)
    recorded = json.loads((data / "fixed-grid-v2-report.json").read_text())
    if result != recorded:
        raise ValueError("fixed-C report differs from independently audited records")
    (out / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    identity = json.loads((root / "identity.json").read_text())
    (out / "RESULTS.md").write_text(render_fixed(result, identity))
    receipt = {
        "status": "replayed-without-GPU",
        "native_runs": checked,
        "code_pairs": result["all_code"]["pairs"],
        "source": source_receipt,
        "results_sha256": sha(out / "RESULTS.json"),
        "markdown_sha256": sha(out / "RESULTS.md"),
    }
    (out / "REPLAY.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reproduce(args.candidate, args.manifest_sha256, args.base, args.out)))


if __name__ == "__main__":
    main()
