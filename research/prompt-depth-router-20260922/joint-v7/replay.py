"""Verify sealed v7 code probes, C engagement and fresh E2E on hosted CPU."""

import argparse
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
                raise ValueError("unsafe or changed v7 archive member " + name)
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
                raise ValueError("v7 evidence member hash differs " + name)
    if seen != set(expected):
        raise ValueError("v7 archive omits a sealed member")
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


def replay(archive, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != 1 or sha(archive) != manifest["archive_sha256"]:
        raise ValueError("v7 archive differs from sealed manifest")
    with tempfile.TemporaryDirectory(prefix="joint-v6-replay-") as tmp:
        root = Path(tmp)
        members = extract(archive, manifest, root)
        check_models(root)
        native = root / "native"
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
            engagement["confidence_decisions"] <= 0
            or qualification["confidence_decisions"].get("joint-learned", 0) <= 0
            or engagement["source_sha256"] != manifest["source_archive_sha256"]
            or engagement["binary_sha256"] != manifest["binary_sha256"]
        ):
            raise ValueError("sealed single-head C engagement differs")
        frozen = json.loads((native / "final-analysis.json").read_text())
        if score(native) != frozen:
            raise ValueError("independent fresh E2E score differs")
    return {
        "status": "all-v7-hashes-C-engagement-code-probes-and-E2E-score-match",
        "members": members,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.archive, args.manifest), sort_keys=True))


if __name__ == "__main__":
    main()
