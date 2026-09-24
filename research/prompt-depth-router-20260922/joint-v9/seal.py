"""Seal complete v9 native evidence without copying Qwen checkpoint weights."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


MODEL_SHA = "1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a"
BINARY_SHA = "84b04b6ccccc3b64992377cef0677926cb932c7a98f063aff84d1f1f65224d7f"
WORKLOAD_SHA = "3067e98e6c1b2af0e7de1010e62239182b3264c5a1902e01712c50727bc83079"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def inventory(base):
    groups = {
        "source/joint-v9": base / "joint-v9",
        "source/joint-v4": base / "joint-v4",
        "inputs/workloads": base / "workloads",
        "training/old": base / "old-training-v2",
        "training/new": base / "new-training",
        "training/k-models": base / "k-models",
        "training/cd-models": base / "cd-models",
        "selection/arms": base / "arms",
        "selection/final": base / "selected",
        "native/results": base / "results",
        "native/qualifier": base / "qualifier-v3-k20",
    }
    named = {}
    for prefix, root in groups.items():
        if not root.is_dir():
            raise ValueError(f"missing evidence directory: {root}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"evidence symlink: {path}")
            if not path.is_file() or path.suffix == ".pyc":
                continue
            relative = f"{prefix}/{path.relative_to(root).as_posix()}"
            named[relative] = path
    for name in (
        "training.log", "qualifier-v3-k20.stdout.log",
        "qualifier-v3-k20.stderr.log", "qualification-result.json",
        "qualification-quality.json", "validation-quality.json",
        "validation-score.json", "heldout-quality.json",
        "heldout-score.json",
    ):
        path = base / name
        if not path.is_file():
            raise ValueError(f"missing evidence file: {path}")
        named[f"native/{name}"] = path
    if len(list((base / "results").glob("training-*.result.json"))) != 144:
        raise ValueError("training evidence is incomplete")
    result = json.loads((base / "heldout-score.json").read_text())
    if result["phase"] != "heldout":
        raise ValueError("final native score is missing")
    if (base / "pipeline-failed.json").exists():
        raise ValueError("cannot seal a failed native pipeline")
    return dict(sorted(named.items()))


def seal(base, out):
    if sha(base / "Qwen3.8-27B-NVFP4-Q5K-mtp.gguf") != MODEL_SHA:
        raise ValueError("Qwen artifact differs")
    if sha(base / "mtp-v9-binary-20260924") != BINARY_SHA:
        raise ValueError("native executable differs")
    if sha(base / "workloads/manifest.json") != WORKLOAD_SHA:
        raise ValueError("frozen MBPP workload differs")
    files = inventory(base)
    if out.exists():
        raise ValueError("seal destination already exists")
    out.mkdir()
    members = {
        name: {"bytes": path.stat().st_size, "sha256": sha(path)}
        for name, path in files.items()
    }
    archive = out / "native-data.tar.gz"
    with archive.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as tar:
                for name, path in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = path.stat().st_size
                    info.mtime = info.uid = info.gid = 0
                    info.mode = 0o644
                    info.uname = info.gname = ""
                    with path.open("rb") as source:
                        tar.addfile(info, source)
    manifest = {
        "schema": 1,
        "scope": "Qwen draft-only C/K/D v9 on one Nebius RTX PRO 6000",
        "model_sha256": MODEL_SHA,
        "binary_sha256": BINARY_SHA,
        "workloads_sha256": WORKLOAD_SHA,
        "archive_sha256": sha(archive),
        "members": members,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return {
        "archive_sha256": manifest["archive_sha256"],
        "members": len(members),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(seal(args.base, args.out), sort_keys=True))


if __name__ == "__main__":
    main()
