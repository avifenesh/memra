"""Seal native non-code transfer receipts without checkpoint weights."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


MODEL_SHA = "1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a"
BINARY_SHA = "84b04b6ccccc3b64992377cef0677926cb932c7a98f063aff84d1f1f65224d7f"
WORKLOAD_SHA = "2142c62394fbb3f90c684a942e09a227f24f8a32d0011f550dd7b6841ed32dc1"
IFEVAL_SHA = "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49"
GSM8K_SHA = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def inventory(base):
    groups = {
        "source/joint-v10": base / "joint-v10",
        "source/joint-v9": base / "joint-v9",
        "third_party/instruction_following_eval": (
            base / "third_party/instruction_following_eval"
        ),
        "nltk_data": base / "nltk_data",
        "inputs/workloads": base / "workloads",
        "inputs/datasets": base / "datasets",
        "policy/models": base / "models",
        "native/results": base / "results",
    }
    files = {}
    for prefix, root in groups.items():
        if not root.is_dir():
            raise ValueError(f"missing sealed evidence group: {root}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"evidence contains a symlink: {path}")
            if not path.is_file() or path.suffix == ".pyc":
                continue
            files[f"{prefix}/{path.relative_to(root).as_posix()}"] = path
    for name in (
        "arms.json", "run-meta.json", "qualification-result.json",
        "qualification-quality.json", "heldout-quality.json",
        "heldout-score.json",
    ):
        path = base / name
        if not path.is_file():
            raise ValueError(f"missing native transfer receipt: {name}")
        files[f"native/{name}"] = path
    if len(list((base / "results").glob("qualification-*.result.json"))) != 20:
        raise ValueError("non-code qualification arm inventory differs")
    if len(list((base / "results").glob("heldout-*.result.json"))) != 320:
        raise ValueError("non-code final arm inventory differs")
    if (base / "pipeline-failed.json").exists():
        raise ValueError("cannot seal a failed transfer pipeline")
    if set(json.loads((base / "heldout-score.json").read_text())["domains"]) != {
        "ifeval", "gsm8k",
    }:
        raise ValueError("both non-code final strata must be scored")
    return dict(sorted(files.items()))


def seal(base, out):
    pins = {
        base / "Qwen3.8-27B-NVFP4-Q5K-mtp.gguf": MODEL_SHA,
        base / "mtp-depth-study": BINARY_SHA,
        base / "workloads/manifest.json": WORKLOAD_SHA,
        base / "datasets/ifeval.jsonl": IFEVAL_SHA,
        base / "datasets/gsm8k-test.jsonl": GSM8K_SHA,
    }
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError(f"pinned transfer input differs: {path.name}")
    files = inventory(base)
    if out.exists():
        raise ValueError("sealed transfer destination already exists")
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
        "scope": "code-trained Qwen C/K/D non-code transfer on IFEval and GSM8K",
        "model_sha256": MODEL_SHA,
        "binary_sha256": BINARY_SHA,
        "workloads_sha256": WORKLOAD_SHA,
        "ifeval_source_sha256": IFEVAL_SHA,
        "gsm8k_source_sha256": GSM8K_SHA,
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
