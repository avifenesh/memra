"""Seal v11 randomized native training receipts without checkpoint weights."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


MODEL_SHA = "1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a"
BINARY_SHA = "84b04b6ccccc3b64992377cef0677926cb932c7a98f063aff84d1f1f65224d7f"
WORKLOAD_SHA = "655223c8e4ca61fab179f7a4708b152ec35cf351209ba5210c10956080075a58"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def inventory(base):
    groups = {
        "source/joint-v11": base / "joint-v11",
        "source/joint-v9": base / "joint-v9",
        "source/joint-v4": base / "joint-v4",
        "source/private_ops": base / "ops",
        "inputs/workloads": base / "workloads",
        "diagnostic/pilot-results": base / "pilot-results",
        "native/training-results": base / "training-results",
    }
    files = {}
    for prefix, root in groups.items():
        if not root.is_dir():
            raise ValueError(f"missing training evidence group: {root}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"training evidence contains symlink: {path}")
            if not path.is_file() or path.suffix == ".pyc":
                continue
            files[f"{prefix}/{path.relative_to(root).as_posix()}"] = path
    metadata = base / "run-meta.json"
    if not metadata.is_file():
        raise ValueError("v11 training lacks GPU run metadata")
    files["native/run-meta.json"] = metadata
    tasks = json.loads((base / "workloads/manifest.json").read_text())
    workload_root = base / "workloads"
    expected = {"manifest.json"}
    for phase in ("qualification", "training"):
        for domain in ("ifeval", "gsm8k"):
            for entry in tasks["groups"][phase][domain]:
                name = entry["file"]
                if Path(name).name != name:
                    raise ValueError("training prompt path leaves workload root")
                if sha(workload_root / name) != entry["sha256"]:
                    raise ValueError("training prompt bytes differ")
                expected.add(name)
    actual = {
        path.relative_to(workload_root).as_posix()
        for path in workload_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if (
        set(tasks["groups"]) != {"qualification", "training"}
        or actual != expected
        or any(path.is_symlink() for path in workload_root.rglob("*"))
    ):
        raise ValueError("training archive would contain reserved prompts")
    pilot = base / "pilot-result.json"
    if not pilot.is_file():
        raise ValueError("v11 randomized training lacks native pilot")
    files["diagnostic/pilot-result.json"] = pilot
    behavior = base / "behavior-v10.json"
    if not behavior.is_file():
        raise ValueError("v11 training lacks v10 behavior diagnostic")
    files["diagnostic/behavior-v10.json"] = behavior
    for name in ("source-manifest.json", "rental.json"):
        path = base / name
        if not path.is_file():
            raise ValueError(f"v11 training lacks frozen {name}")
        files[f"inputs/{name}"] = path
    results = list((base / "training-results").glob("training-*.result.json"))
    if len(results) != 224:
        raise ValueError("v11 fixed-K, randomized-K and randomized-D session count differs")
    for path in results:
        row = json.loads(path.read_text())
        if row["name"] != path.name.removesuffix(".result.json"):
            raise ValueError("v11 training result name differs")
    if (base / "pipeline-failed.json").exists():
        raise ValueError("cannot seal failed v11 training")
    return dict(sorted(files.items()))


def seal(base, out):
    model = base / "Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
    binary = base / "mtp-depth-study"
    manifest = base / "workloads/manifest.json"
    for path, expected in (
        (model, MODEL_SHA), (binary, BINARY_SHA),
        (manifest, WORKLOAD_SHA),
    ):
        if sha(path) != expected:
            raise ValueError(f"v11 pinned training input changed: {path.name}")
    files = inventory(base)
    if out.exists():
        raise ValueError("v11 training archive destination exists")
    out.mkdir()
    members = {
        name: {"bytes": path.stat().st_size, "sha256": sha(path)}
        for name, path in files.items()
    }
    archive = out / "native-data.tar.gz"
    with archive.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                           mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as tar:
                for name, path in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = path.stat().st_size
                    info.mtime = info.uid = info.gid = 0
                    info.mode = 0o644
                    info.uname = info.gname = ""
                    with path.open("rb") as source:
                        tar.addfile(info, source)
    record = {
        "schema": 1,
        "scope": "v11 randomized non-code training-only native data",
        "model_sha256": MODEL_SHA,
        "binary_sha256": BINARY_SHA,
        "workloads_sha256": WORKLOAD_SHA,
        "archive_sha256": sha(archive),
        "members": members,
    }
    (out / "manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return {"archive_sha256": record["archive_sha256"],
            "members": len(members), "sessions": 224}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(seal(args.base, args.out), sort_keys=True))


if __name__ == "__main__":
    main()
