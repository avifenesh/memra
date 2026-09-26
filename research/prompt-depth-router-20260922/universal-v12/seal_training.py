"""Seal native prose training sessions and training-only prompt bytes."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


MODEL_SHA = "1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a"
BINARY_SHA = "84b04b6ccccc3b64992377cef0677926cb932c7a98f063aff84d1f1f65224d7f"
TRAIN_SHA = "2bca403c1edff44a1d707abd60710767f9dbbe2a63721757032b0fbe223c5e00"
FULL_SHA = "b35374d34e2a28b31cca0399e6394fa3db3b0f27e9d593b17032a7856b2a5477"
DOMAINS = ("code", "prose", "math")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def expected_workloads(root):
    manifest_path = root / "manifest.json"
    if sha(manifest_path) != TRAIN_SHA:
        raise ValueError("training-only prompt projection changed")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["schema"] != 1
        or manifest["source_full_manifest_sha256"] != FULL_SHA
        or set(manifest["groups"]) != {"qualification", "training"}
    ):
        raise ValueError("training-only phase inventory changed")
    expected = {"manifest.json"}
    for phase in ("qualification", "training"):
        if set(manifest["groups"][phase]) != set(DOMAINS):
            raise ValueError("mixed training domain inventory differs")
        for domain in DOMAINS:
            for entry in manifest["groups"][phase][domain]:
                name = entry["file"]
                if Path(name).name != name or sha(root / name) != entry["sha256"]:
                    raise ValueError("training prompt bytes changed")
                expected.add(name)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != expected:
        raise ValueError("reserved validation/final prompt entered training host")


def inventory(base):
    groups = {
        "inputs/phase-training": base / "phase-training",
        "source/universal-v12": base / "universal-v12",
        "source/joint-v9": base / "joint-v9",
        "source/joint-v11": base / "joint-v11",
        "source/private_ops": base / "ops",
        "diagnostic/pilot-results": base / "pilot-results",
        "diagnostic/judge-preflight":
        base / "judge-preflight",
        "native/training-prose-results":
        base / "training-prose-results",
    }
    files = {}
    for prefix, root in groups.items():
        if not root.is_dir():
            raise ValueError(f"training seal group missing: {prefix}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("training seal contains symlink")
            if path.is_file() and path.suffix != ".pyc":
                files[f"{prefix}/{path.relative_to(root).as_posix()}"] = path
    for name in (
        "rental.json", "cuda-accept.json", "run-meta.json",
        "pilot-result.json",
    ):
        path = base / name
        if not path.is_file():
            raise ValueError(f"training run metadata missing: {name}")
        files[f"native/{name}"] = path
    results = list((base / "training-prose-results").glob(
        "training-*.result.json"
    ))
    pilot_results = list((base / "pilot-results").glob(
        "pilot-*.result.json"
    ))
    if len(results) != 112:
        raise ValueError("randomized prose training session count differs")
    if len(pilot_results) != 6:
        raise ValueError("fixed D1/D2 native pilot session count differs")
    for path in results:
        row = json.loads(path.read_text())
        if row["name"] != path.name.removesuffix(".result.json"):
            raise ValueError("randomized prose result name differs")
    return dict(sorted(files.items()))


def seal(base, out):
    if (
        sha(base / "Qwen3.8-27B-NVFP4-Q5K-mtp.gguf") != MODEL_SHA
        or sha(base / "mtp-depth-study") != BINARY_SHA
    ):
        raise ValueError("sealed prose native artifact changed")
    expected_workloads(base / "phase-training")
    metadata = json.loads((base / "run-meta.json").read_text())
    if (
        metadata["model_sha256"] != MODEL_SHA
        or metadata["binary_sha256"] != BINARY_SHA
        or metadata["training_workloads_sha256"] != TRAIN_SHA
        or metadata["source_full_manifest_sha256"] != FULL_SHA
        or metadata["customer_capture"] is not False
        or metadata["cuda_allocated"] is not True
        or metadata["rental_sha256"] != sha(base / "rental.json")
        or metadata["cuda_accept_sha256"]
        != sha(base / "cuda-accept.json")
        or metadata["ops_source_sha256"]
        != sha(base / "ops/run_meta_v12.py")
        or metadata["judge_preflight_sha256"]
        != sha(base / "judge-preflight/manifest.json")
    ):
        raise ValueError("research host identity or CUDA proof differs")
    for name, expected in metadata["source_files_sha256"].items():
        if (
            not name.startswith(
                ("universal-v12/", "joint-v9/", "joint-v11/")
            )
            or sha(base / name) != expected
        ):
            raise ValueError("research training source changed after run metadata")
    pilot = json.loads((base / "pilot-result.json").read_text())
    if (
        pilot["status"] != "fixed-D1-D2-full-head-and-KV-engaged"
        or pilot["model_sha256"] != MODEL_SHA
        or pilot["binary_sha256"] != BINARY_SHA
        or pilot["training_workloads_sha256"] != TRAIN_SHA
        or pilot["run_meta_sha256"] != sha(base / "run-meta.json")
        or pilot["gpu_uuid"] != metadata["gpu_uuid"]
    ):
        raise ValueError("fixed D1/D2 pilot does not match research host")
    files = inventory(base)
    if out.exists():
        raise ValueError("sealed prose training destination exists")
    out.mkdir()
    members = {
        name: {"bytes": path.stat().st_size, "sha256": sha(path)}
        for name, path in files.items()
    }
    archive = out / "native-data.tar.gz"
    with archive.open("xb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0,
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as tar:
                for name, path in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = path.stat().st_size
                    info.mtime = info.uid = info.gid = 0
                    info.mode = 0o644
                    with path.open("rb") as source:
                        tar.addfile(info, source)
    manifest = {
        "schema": 1,
        "scope": "fresh mixed-prose training-only native data",
        "model_sha256": MODEL_SHA,
        "binary_sha256": BINARY_SHA,
        "training_workloads_sha256": TRAIN_SHA,
        "source_full_manifest_sha256": FULL_SHA,
        "pilot_sha256": sha(base / "pilot-result.json"),
        "judge_preflight_sha256":
        sha(base / "judge-preflight/manifest.json"),
        "archive_sha256": sha(archive),
        "members": members,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return {
        "archive_sha256": manifest["archive_sha256"],
        "members": len(members), "sessions": 112,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        seal(args.base.resolve(), args.out.resolve()), sort_keys=True,
    ))


if __name__ == "__main__":
    main()
