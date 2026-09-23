"""Project the complete fixed-C science record from the private operator bundle."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from archive_io import write_archive  # noqa: E402


HARNESS = {
    "harness/prefix/audit.py",
    "harness/prefix/run.py",
    "harness/confidence/fixed_grid.py",
    "harness/confidence/report_fixed.py",
    "harness/confidence/patch_source.py",
    "harness/confidence/prepare_patched.py",
    "harness/confidence/remote_job_v2.sh",
    "harness/confidence/offline_adaptive.py",
    "harness/confidence/sampled_gate.sh",
    "harness/confidence/postscore_blind.sh",
    "harness/confidence/heldout_workloads.py",
    "harness/confidence/heldout_pair.py",
}
SCIENCE = {
    "fixed-grid-v2-report.json",
    "oracle-code-prompt.txt",
    "oracle-v2-off.log",
    "oracle-v2-c030.log",
    "oracle-v2-c030zero.log",
    "oracle-v2-c015.log",
    "source-confidence.json",
    "source-patch.json",
    "models/qwen/artifacts.lock.json",
    "workloads/manifest.json",
    "offline-adaptive.json",
    "sampled-gate.exit",
    "sampled-oracle-off.log",
    "sampled-oracle-c015.log",
    "sampled-oracle-c030.log",
    "sampled-oracle-c030zero.log",
    "postscore-blind.exit",
    "heldout-workloads/manifest.json",
    "heldout-pair/PRESELECTION-FREEZE.json",
    "heldout-pair/FREEZE.json",
    "heldout-pair/status.json",
}
SOURCE = "runtime-source-confidence.tar.gz"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_extract(archive, out):
    seen = set()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            logical = PurePosixPath(member.name.rstrip("/"))
            if logical.is_absolute() or ".." in logical.parts or str(logical) != member.name.rstrip("/"):
                raise ValueError("unsafe operator archive path")
            if member.isdir():
                continue
            if not member.isfile() or member.name in seen:
                raise ValueError("non-file or duplicate operator member")
            seen.add(member.name)
            path = out.joinpath(*logical.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, path.open("xb") as target:
                shutil.copyfileobj(source, target)
    return seen


def inventory(files):
    return {
        name: {"bytes": path.stat().st_size, "sha256": sha(path)}
        for name, path in sorted(files.items())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="confidence-projection-") as temporary:
        root = Path(temporary)
        seen = safe_extract(args.operator, root)
        if not HARNESS <= seen or not SCIENCE <= seen or SOURCE not in seen or "job-v2.exit" not in seen:
            raise ValueError("operator bundle lacks source, records or reporting code")
        if not (root / "job-v2.exit").read_text().startswith("exit=0 "):
            raise ValueError("operator study did not close successfully")
        if not (root / "sampled-gate.exit").read_text().startswith("exit=0 "):
            raise ValueError("sampled cutoff gate did not close successfully")
        if not (root / "postscore-blind.exit").read_text().startswith("exit=0 "):
            raise ValueError("blinded held-out preparation did not close successfully")
        native = {
            name: root / name for name in seen
            if name.startswith((
                "fixed-grid-v2/", "workloads/",
                "heldout-workloads/", "heldout-pair/",
            )) or name in SCIENCE
        }
        source = json.loads((root / "source-confidence.json").read_text())
        freeze = json.loads((root / "fixed-grid-v2/FREEZE.json").read_text())
        status = json.loads((root / "fixed-grid-v2/status.json").read_text())
        heldout_status = json.loads((root / "heldout-pair/status.json").read_text())
        identity = json.loads((root / "fixed-grid-v2/identity.json").read_text())
        if (status["status"] != "completed"
                or freeze["source_sha256"] != sha(root / "source-confidence.json")
                or freeze["script_sha256"] != sha(root / "harness/confidence/fixed_grid.py")
                or freeze["workloads_sha256"] != sha(root / "workloads/manifest.json")
                or identity["source"] != source
                or identity["artifacts"]["qwen"] != json.loads(
                    (root / "models/qwen/artifacts.lock.json").read_text()
                )
                or source["runtime_source_sha256"] != sha(root / SOURCE)
                or source["runner_sha256"] != sha(root / "harness/prefix/run.py")
                or source["grid_sha256"] != sha(root / "harness/confidence/fixed_grid.py")
                or source["reporter_sha256"] != sha(root / "harness/confidence/report_fixed.py")
                or source["source_patcher_sha256"] != sha(root / "harness/confidence/patch_source.py")
                or source["source_patch_receipt_sha256"] != sha(root / "source-patch.json")):
            raise ValueError("operator result is incomplete or source bindings changed")
        if heldout_status["status"] not in ("no-positive-development-c", "completed"):
            raise ValueError("held-out qualification or selected C study is incomplete")
        write_archive(out / "native-data.tar.gz", native)
        write_archive(
            out / "harness-source.tar.gz",
            {name: root / name for name in HARNESS},
        )
        shutil.copyfile(root / SOURCE, out / SOURCE)
        archives = {
            name: out / name
            for name in ("native-data.tar.gz", "harness-source.tar.gz", SOURCE)
        }
        manifest = {
            "schema": 1,
            "scope": "Qwen fixed K=3 confidence-cutoff science; no serving qualification",
            "operator_archive_sha256": sha(args.operator),
            "source_recipe_commit": source["base_source_recipe_commit"],
            "source_patch_sha256": source["source_patcher_sha256"],
            "runtime_source_sha256": source["runtime_source_sha256"],
            "native_members": inventory(native),
            "harness_members": inventory({name: root / name for name in HARNESS}),
            "files": inventory(archives),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (out / "manifest.sha256").write_text(sha(out / "manifest.json") + "  manifest.json\n")
        print(json.dumps({
            "native_members": len(native),
            "harness_members": len(HARNESS),
            "manifest_sha256": sha(out / "manifest.json"),
        }))


if __name__ == "__main__":
    main()
