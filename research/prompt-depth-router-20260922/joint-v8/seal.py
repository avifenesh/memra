"""Seal the corrected prime-time C/K/D continuation without checkpoint weights."""

import argparse
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tarfile


MODEL_SHA = "1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a"
REQUIRED = (
    "run-meta.json", "training-parent.json", "engagement-result.json",
    "selected-fixed.json", "selected-topk.json",
    "joint-selection-summary.json", "selected-joint.json",
    "policy-qualification-result.json",
    "heldout-summary.json", "quality.json", "final-analysis.json",
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def seal(args):
    if sha(args.model) != MODEL_SHA:
        raise ValueError("Qwen full-head artifact differs before sealing")
    source_record = json.loads(args.source_record.read_text())
    if sha(args.source) != source_record["source_archive_sha256"]:
        raise ValueError("corrected source differs from its host build")
    for name in REQUIRED:
        if not (args.results / name).is_file():
            raise ValueError("v8 native study lacks required receipt " + name)
    engagement = json.loads((args.results / "engagement-result.json").read_text())
    if (
        engagement["status"] != "prime-probability-learned-C-engaged"
        or engagement["source_sha256"] != source_record["source_archive_sha256"]
        or engagement["binary_sha256"] != sha(args.binary)
        or not 0 < engagement["confidence_stops"] < engagement["confidence_decisions"]
    ):
        raise ValueError("v8 learned C engagement differs from sealed binary")
    qualification = json.loads(
        (args.results / "policy-qualification-result.json").read_text()
    )
    if not (
        0 < qualification["confidence_stops"].get("joint-learned", 0)
        < qualification["confidence_decisions"].get("joint-learned", 0)
    ):
        raise ValueError("v8 final qualifier did not vary learned C")
    named = {
        "source/runtime-source-joint-v8.tar.gz": args.source,
        "source/source.json": args.source_record,
        "source/mtp-depth-study": args.binary,
        "source/inputs.tar.gz": args.inputs,
    }
    for folder, prefix in ((args.results, "native"), (args.models, "models")):
        for path in sorted(folder.rglob("*")):
            if path.is_symlink():
                raise ValueError("v8 receipt symlink cannot be sealed")
            if path.is_file():
                named[f"{prefix}/{path.relative_to(folder).as_posix()}"] = path
    files = {}
    for name, path in sorted(named.items()):
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError("missing or empty v8 study evidence " + name)
        files[name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    manifest = {
        "schema": 1,
        "scope": "Qwen3.8 full-head fixed-target draft-only K/C/D code study with prime-time sampled C",
        "model_sha256": MODEL_SHA,
        "source_archive_sha256": sha(args.source),
        "binary_sha256": sha(args.binary),
        "input_bundle_sha256": sha(args.inputs),
        "members": files,
    }
    args.out.mkdir(exist_ok=False)
    archive = args.out / "native-data.tar.gz"
    with archive.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as stream:
                for name, path in sorted(named.items()):
                    data = path.read_bytes()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mtime = info.uid = info.gid = 0
                    info.mode = 0o644
                    info.uname = info.gname = ""
                    stream.addfile(info, BytesIO(data))
    manifest["archive_sha256"] = sha(archive)
    with (args.out / "manifest.json").open("x") as target:
        json.dump(manifest, target, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps({
        "archive_sha256": manifest["archive_sha256"],
        "members": len(files),
    }))


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "model", "source", "source-record", "binary",
        "inputs", "results", "models", "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    seal(parser.parse_args())


if __name__ == "__main__":
    main()
