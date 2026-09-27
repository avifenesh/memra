"""Bank exact native C/K/D receipts without retaining the Qwen weights."""

import argparse
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tarfile


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inputs(args):
    named = {
        "source/baseline-runtime.tar.gz": args.baseline_source,
        "source/baseline-source.json": args.baseline_source_record,
        "source/learned-runtime.tar.gz": args.learned_source,
        "source/learned-source.json": args.learned_source_record,
        "source/mtp-depth-study-baseline": args.baseline,
        "source/mtp-depth-study-learned": args.learned,
        "source/inputs.tar.gz": args.inputs,
    }
    for source_dir, prefix in ((args.results, "native"), (args.models, "models")):
        for path in sorted(source_dir.rglob("*")):
            if path.is_symlink():
                raise ValueError("symlink in research receipts")
            if path.is_file():
                named[f"{prefix}/{path.relative_to(source_dir).as_posix()}"] = path
    for name, path in named.items():
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError("missing or empty pinned evidence file: " + name)
    return named


def seal(args):
    if sha(args.model) != "1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a":
        raise ValueError("Qwen artifact changed before sealing")
    for archive, record_path in (
        (args.baseline_source, args.baseline_source_record),
        (args.learned_source, args.learned_source_record),
    ):
        source = json.loads(record_path.read_text())
        if sha(archive) != source["source_archive_sha256"]:
            raise ValueError("native source archive changed before sealing")
    if not (args.results / "training-summary.json").exists() or not (
        args.results / "heldout-summary.json"
    ).exists():
        raise ValueError("C/K/D native phases are incomplete")
    named = inputs(args)
    files = {
        name: {"sha256": sha(path), "bytes": path.stat().st_size}
        for name, path in sorted(named.items())
    }
    record = {
        "schema": 1,
        "scope": "nonproduction Qwen full-head C/K/D continuing-session study",
        "model_sha256": sha(args.model),
        "baseline_source_archive_sha256": sha(args.baseline_source),
        "learned_source_archive_sha256": sha(args.learned_source),
        "baseline_binary_sha256": sha(args.baseline),
        "learned_binary_sha256": sha(args.learned),
        "native_members": files,
    }
    args.out.mkdir(exist_ok=False)
    archive = args.out / "native-data.tar.gz"
    with archive.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w|") as stream:
                for name, path in sorted(named.items()):
                    data = path.read_bytes()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mtime = info.uid = info.gid = 0
                    info.mode = 0o644
                    info.uname = info.gname = ""
                    stream.addfile(info, BytesIO(data))
    record["native_archive_sha256"] = sha(archive)
    with (args.out / "manifest.json").open("x") as target:
        json.dump(record, target, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps({
        "archive_sha256": record["native_archive_sha256"],
        "members": len(files),
    }))


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "model", "baseline-source", "baseline-source-record",
        "learned-source", "learned-source-record", "baseline", "learned",
        "inputs", "results", "models", "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    seal(parser.parse_args())


if __name__ == "__main__":
    main()
