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
    from offline_adaptive import replay as offline_replay  # noqa: PLC0415
    from heldout_report import report as heldout_report  # noqa: PLC0415
    from fixed_grid import ARMS as confidence_arms  # noqa: PLC0415

    root = data / "fixed-grid-v2"
    status = json.loads((root / "status.json").read_text())
    freeze = json.loads((root / "FREEZE.json").read_text())
    workloads = json.loads((data / "workloads/manifest.json").read_text())
    if sha(data / "workloads/manifest.json") != freeze["workloads_sha256"]:
        raise ValueError("registered workload manifest differs")
    qwen = workloads["families"]["qwen"]
    for entry in (qwen["qualification"], *qwen["scenarios"].values()):
        if sha(data / "workloads" / entry["file"]) != entry["sha256"]:
            raise ValueError("registered workload text differs: " + entry["file"])
    checked = 0
    for relative in status["completed"]:
        family, phase, label = relative.split("/")
        if family != "qwen" or phase not in ("qualification", "scored"):
            raise ValueError("unexpected native record path")
        entry = (qwen["qualification"] if phase == "qualification"
                 else qwen["scenarios"][str(int(label.split("-")[0]))])
        run = root / relative
        saved = json.loads(run.with_name(run.name + ".audit.json").read_text())
        command = json.loads(run.with_name(run.name + ".command.json").read_text())
        exit_row = json.loads(run.with_name(run.name + ".exit.json").read_text())
        settings = command["settings"]
        cutoff = freeze["arms"][label.split("-")[-1]]
        if (float(settings["MEMRA_SPEC_PMIN"]) != cutoff["pmin"]
                or settings["MEMRA_SPEC_PMIN0"] != ("1" if cutoff["pmin0"] else "0")
                or settings["MEMRA_SPEC_ADAPT"] != "0"
                or settings["MEMRA_SPEC_STATS"] != "1"
                or settings["MEMRA_SPEC_PMIN_INROUND"] != "0"
                or command["binary_sha256"] != source["binaries"]["qwen-prefix-study"]
                or command["workload_sha256"] != entry["sha256"]
                or command["seed"] != entry["seed"]
                or exit_row["returncode"] != 0 or exit_row["contamination"]
                or run.with_name(run.name + ".gpu.csv").stat().st_size == 0):
            raise ValueError("native arm settings or execution differ: " + relative)
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
    for name in (
        "oracle-v2-off.log", "oracle-v2-c015.log",
        "oracle-v2-c030.log", "oracle-v2-c030zero.log",
    ):
        if "=== SELF-CONSISTENCY PASS ===" not in (data / name).read_text():
            raise ValueError("target oracle failed: " + name)
    if not (data / "sampled-gate.exit").read_text().startswith("exit=0 "):
        raise ValueError("sampled cutoff gate did not complete")
    for label in ("off", "c015", "c030", "c030zero"):
        name = f"sampled-oracle-{label}.log"
        text = (data / name).read_text()
        if ("=== SELF-CONSISTENCY PASS ===" not in text
                or "PASS (seeded rerun identical)" not in text):
            raise ValueError("sampled cutoff reproducibility failed: " + name)
    result = report(root)
    recorded = json.loads((data / "fixed-grid-v2-report.json").read_text())
    if result != recorded:
        raise ValueError("fixed-C report differs from independently audited records")
    (out / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    identity = json.loads((root / "identity.json").read_text())
    (out / "RESULTS.md").write_text(render_fixed(result, identity))
    offline = offline_replay(root)
    if json.loads(json.dumps(offline)) != json.loads((data / "offline-adaptive.json").read_text()):
        raise ValueError("offline adaptive-C replay differs from its recorded outcome")
    (out / "OFFLINE-RESULTS.json").write_text(json.dumps(offline, indent=2) + "\n")

    if not (data / "postscore-blind.exit").read_text().startswith("exit=0 "):
        raise ValueError("blinded qualification did not close")
    heldout_root = data / "heldout-pair"
    heldout_workloads = data / "heldout-workloads"
    heldout_manifest = json.loads((heldout_workloads / "manifest.json").read_text())
    heldout_status = json.loads((heldout_root / "status.json").read_text())
    heldout_freeze = json.loads((heldout_root / "FREEZE.json").read_text())
    if (sha(heldout_workloads / "manifest.json") != heldout_freeze["workloads_sha256"]
            or heldout_freeze["source_sha256"] != sha(data / "source-confidence.json")
            or heldout_freeze["runner_sha256"] != sha(data / "harness/confidence/heldout_pair.py")
            or heldout_manifest["generator_sha256"]
            != sha(data / "harness/confidence/heldout_workloads.py")
            or heldout_manifest["prefix_helper_sha256"]
            != sha(data / "harness/prefix/workloads.py")
            or heldout_manifest["prompt_helper_sha256"]
            != sha(data / "harness/prefix/workloads_simple.py")
            or heldout_manifest["binary_sha256"] != source["binaries"]["qwen-prefix-study"]):
        raise ValueError("held-out workload or source identity differs")
    for entry in (
        heldout_manifest["qualification"], *heldout_manifest["scenarios"].values()
    ):
        if sha(heldout_workloads / entry["file"]) != entry["sha256"]:
            raise ValueError("held-out prompt bytes differ: " + entry["file"])
    heldout_identity = json.loads((heldout_root / "identity.json").read_text())
    if (heldout_identity["source"] != source
            or heldout_identity["artifacts"]["qwen"] != locked
            or heldout_identity["gpu"] != identity["gpu"]):
        raise ValueError("held-out model or GPU differs from development")
    heldout_checked = 0
    for relative in heldout_status["completed"]:
        family, phase, label = relative.split("/")
        if family != "qwen" or phase not in ("qualification", "heldout"):
            raise ValueError("unexpected held-out native record")
        entry = (
            heldout_manifest["qualification"] if phase == "qualification"
            else heldout_manifest["scenarios"][str(int(label.split("-")[0]))]
        )
        kind = label.split("-")[-1]
        expected_arm = "fixed:2" if kind == "k2off" else "fixed:3"
        pmin, pmin0 = (
            confidence_arms[heldout_freeze["selected_c"]]
            if kind == "selected" else (0.0, False)
        )
        run = heldout_root / relative
        saved = json.loads(run.with_name(run.name + ".audit.json").read_text())
        command = json.loads(run.with_name(run.name + ".command.json").read_text())
        exit_row = json.loads(run.with_name(run.name + ".exit.json").read_text())
        settings = command["settings"]
        if (saved["arm"] != expected_arm or saved["seed"] != entry["seed"]
                or float(settings["MEMRA_SPEC_PMIN"]) != pmin
                or settings["MEMRA_SPEC_PMIN0"] != ("1" if pmin0 else "0")
                or settings["MEMRA_SPEC_ADAPT"] != "0"
                or settings["MEMRA_SPEC_STATS"] != "1"
                or settings["MEMRA_SPEC_PMIN_INROUND"] != "0"
                or command["binary_sha256"] != source["binaries"]["qwen-prefix-study"]
                or command["workload_sha256"] != entry["sha256"]
                or command["seed"] != entry["seed"]
                or exit_row["returncode"] != 0 or exit_row["contamination"]
                or run.with_name(run.name + ".gpu.csv").stat().st_size == 0):
            raise ValueError("held-out native command or result differs: " + relative)
        metrics = metrics_from_log(
            run.with_name(run.name + ".log").read_text()
        )[expected_arm]
        independent = audit_run(
            run, entry, expected_arm, entry["seed"], metrics, audit_context, loop_candidate
        )
        if (saved["requests"] != independent["requests"]
                or saved["context_accounting"] != independent["context_accounting"]
                or saved["metrics"] != metrics):
            raise ValueError("held-out request audit differs: " + relative)
        heldout_checked += 1
    heldout = heldout_report(
        heldout_root, heldout_workloads, data / "fixed-grid-v2-report.json"
    )
    if json.loads(json.dumps(heldout)) != json.loads((data / "heldout-report.json").read_text()):
        raise ValueError("held-out report differs from independently audited records")
    (out / "HELDOUT-RESULTS.json").write_text(json.dumps(heldout, indent=2) + "\n")

    receipt = {
        "status": "replayed-without-GPU",
        "native_runs": checked,
        "heldout_native_runs": heldout_checked,
        "code_pairs": result["all_code"]["pairs"],
        "source": source_receipt,
        "results_sha256": sha(out / "RESULTS.json"),
        "markdown_sha256": sha(out / "RESULTS.md"),
        "offline_results_sha256": sha(out / "OFFLINE-RESULTS.json"),
        "heldout_results_sha256": sha(out / "HELDOUT-RESULTS.json"),
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
