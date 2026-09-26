"""Verify sealed v8 code probes, C decisions and fresh E2E on hosted CPU."""

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

from analyze_final import score
from quality import score as quality_score


FRESH_MANIFEST = (
    "research/prompt-depth-router-20260922/joint-v4/"
    "workloads-v4/manifest.json"
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract(archive, manifest, target):
    expected = manifest["members"]
    seen = set()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            name = member.name
            logical = PurePosixPath(name)
            if (
                name not in expected or name in seen or not member.isfile()
                or logical.is_absolute() or ".." in logical.parts
                or str(logical) != name
                or member.size != expected[name]["bytes"]
            ):
                raise ValueError("unsafe or changed v8 archive member " + name)
            seen.add(name)
            destination = (
                target.joinpath(*logical.parts)
                if logical.parts[0] in ("native", "models")
                else target / "inputs.tar.gz"
                if name == "source/inputs.tar.gz"
                else None
            )
            if destination is not None:
                destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with stream.extractfile(member) as source:
                sink = destination.open("xb") if destination is not None else None
                try:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                        if sink is not None:
                            sink.write(chunk)
                finally:
                    if sink is not None:
                        sink.close()
            if digest.hexdigest() != expected[name]["sha256"]:
                raise ValueError("v8 evidence member hash differs " + name)
    if seen != set(expected):
        raise ValueError("v8 archive omits a sealed member")
    return len(seen)


def check_models(root):
    topk = root / "models/topk"
    manifest = json.loads((topk / "topk-models.json").read_text())
    for row in manifest:
        path = topk / f"topk-{row['variant']}.tsv"
        if sha(path) != row["model_sha256"]:
            raise ValueError("trained K router hash differs")
    selected = json.loads((root / "native/selected-topk.json").read_text())
    joint = json.loads((root / "native/selected-joint.json").read_text())
    if selected["variant"] is not None:
        path = topk / f"topk-{selected['variant']}.tsv"
        if sha(path) != selected["model_sha256"]:
            raise ValueError("selected K router hash differs")
    if joint["variant"] is not None:
        for k in joint["available_top_k"]:
            folder = root / "models" / f"topk{k}"
            for kind, manifest_name in (
                ("depth", "models.json"),
                ("confidence", "confidence-models.json"),
            ):
                rows = json.loads((folder / manifest_name).read_text())
                for row in rows:
                    path = folder / f"{kind}-{row['variant']}.tsv"
                    if sha(path) != row["model_sha256"]:
                        raise ValueError("selected C/D model family hash differs")


def check_parent(archive, manifest_path, root, lineage):
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["schema"] != 1
        or sha(archive) != manifest["archive_sha256"]
        or manifest["source_archive_sha256"] != lineage["parent_source_sha256"]
        or manifest["binary_sha256"] != lineage["parent_binary_sha256"]
        or manifest["lineage_sha256"] != sha(root / "native/training-parent.json")
    ):
        raise ValueError("v6 training parent seal differs from v8 lineage")
    parent = root / "parent"
    members = extract(archive, manifest, parent)
    for name, digest in lineage["control_sha256"].items():
        if sha(parent / "native" / name) != digest:
            raise ValueError("v6 training control differs: " + name)
    for name, digest in lineage["model_sha256"].items():
        if sha(parent / "models" / name) != digest:
            raise ValueError("v6 fitted model differs: " + name)
    if (
        (parent / "native/heldout-summary.json").exists()
        or (parent / "native/continuation.exit").read_text().strip() != "1"
    ):
        raise ValueError("v6 parent was not stopped before fresh heldout")
    selected = json.loads(
        (parent / "native/joint-selection-summary.json").read_text()
    )
    decisions = 0
    for record in selected["records"]:
        if not record["variant"].startswith("joint-"):
            continue
        with (parent / "native" / record["name"] / "turns.tsv").open() as stream:
            decisions += sum(
                int(row["confidence_decisions"])
                for row in csv.DictReader(stream, delimiter="\t")
            )
    if decisions:
        raise ValueError("v6 parent unexpectedly engaged learned C")
    return members


def replay(archive, manifest_path, parent_archive=None, parent_manifest=None):
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != 1 or sha(archive) != manifest["archive_sha256"]:
        raise ValueError("v8 archive differs from sealed manifest")
    if (parent_archive is None) != (parent_manifest is None):
        raise ValueError("v6 parent archive and manifest must be paired")
    with tempfile.TemporaryDirectory(prefix="joint-v8-replay-") as tmp:
        root = Path(tmp)
        members = extract(archive, manifest, root)
        check_models(root)
        native = root / "native"
        parent_members = None
        if parent_archive is not None:
            lineage = json.loads((native / "training-parent.json").read_text())
            parent_members = check_parent(
                parent_archive, parent_manifest, root, lineage,
            )
        with tarfile.open(root / "inputs.tar.gz", "r:gz") as inputs:
            frozen = inputs.getmember(FRESH_MANIFEST)
            if not frozen.isfile() or frozen.size > 1024 * 1024:
                raise ValueError("fresh workload manifest differs")
            with inputs.extractfile(frozen) as source:
                workload = json.load(source)
        archived_quality = json.loads((native / "quality.json").read_text())
        if quality_score(native, workload) != archived_quality:
            raise ValueError("independent fresh code probe differs")
        engagement = json.loads((native / "engagement-result.json").read_text())
        qualification = json.loads(
            (native / "policy-qualification-result.json").read_text()
        )
        if (
            not 0 < engagement["confidence_stops"] < engagement["confidence_decisions"]
            or not 0 < qualification["confidence_stops"].get("joint-learned", 0)
            < qualification["confidence_decisions"].get("joint-learned", 0)
            or engagement["source_sha256"] != manifest["source_archive_sha256"]
            or engagement["binary_sha256"] != manifest["binary_sha256"]
        ):
            raise ValueError("sealed single-head C engagement differs")
        frozen = json.loads((native / "final-analysis.json").read_text())
        if score(native) != frozen:
            raise ValueError("independent fresh E2E score differs")
    return {
        "status": "all-v8-hashes-C-decisions-code-probes-and-E2E-score-match",
        "members": members,
        "parent_members": parent_members,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent-archive", type=Path)
    parser.add_argument("--parent-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(
        args.archive, args.manifest,
        args.parent_archive, args.parent_manifest,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
