"""Bank v6 training and failed-C diagnostics as a separate parent artifact."""

import argparse
import csv
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tarfile


SOURCE_SHA = "a391ff6337434e45acac4def721b816ccb4c6027ea93aad83cb90d40390c729a"
BINARY_SHA = "07033ec249485f9da202f266d07d5432554a1f8b65ea6d14f55bee0387327c15"
REQUIRED = (
    "development-summary.json", "model-training-summary.json",
    "topk-training-summary.json", "depth-grid-analysis.json",
    "confidence-grid-summary.json", "selected-fixed.json",
    "selected-topk.json", "joint-selection-summary.json",
    "selected-joint.json", "continuation.exit",
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def seal(args):
    if sha(args.source) != SOURCE_SHA or sha(args.binary) != BINARY_SHA:
        raise ValueError("v6 source or binary differs from training parent")
    source = json.loads(args.source_record.read_text())
    if source["source_archive_sha256"] != SOURCE_SHA:
        raise ValueError("v6 source record differs")
    if (args.results / "heldout-summary.json").exists():
        raise ValueError("v6 failed C study unexpectedly opened fresh heldout")
    if (args.results / "continuation.exit").read_text().strip() != "1":
        raise ValueError("v6 failed-C exit marker differs")
    for name in REQUIRED:
        if not (args.results / name).is_file():
            raise ValueError("missing v6 training parent receipt " + name)
    selection = json.loads(
        (args.results / "joint-selection-summary.json").read_text()
    )
    if sum(
        int(turn["confidence_decisions"])
        for record in selection["records"]
        if record["variant"].startswith("joint-")
        for turn in rows(args.results / record["name"] / "turns.tsv")
    ) != 0:
        raise ValueError("v6 learned C failure no longer reproduces")
    lineage = json.loads(args.lineage.read_text())
    if any(
        sha(args.results / name) != digest
        for name, digest in lineage["control_sha256"].items()
    ) or any(
        sha(args.models / name) != digest
        for name, digest in lineage["model_sha256"].items()
    ):
        raise ValueError("v8 copied training lineage differs from v6 archive")
    named = {
        "source/runtime-source-joint-v6.tar.gz": args.source,
        "source/source.json": args.source_record,
        "source/mtp-depth-study": args.binary,
        "source/inputs.tar.gz": args.inputs,
    }
    for folder, prefix in ((args.results, "native"), (args.models, "models")):
        for path in sorted(folder.rglob("*")):
            if path.is_symlink():
                raise ValueError("v6 training parent has a symlink")
            if path.is_file():
                named[f"{prefix}/{path.relative_to(folder).as_posix()}"] = path
    members = {
        name: {"bytes": path.stat().st_size, "sha256": sha(path)}
        for name, path in sorted(named.items())
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
    manifest = {
        "schema": 1,
        "scope": "v6 frozen training and static controls; learned C unengaged; no fresh heldout",
        "source_archive_sha256": SOURCE_SHA,
        "binary_sha256": BINARY_SHA,
        "input_bundle_sha256": sha(args.inputs),
        "lineage_sha256": sha(args.lineage),
        "members": members,
        "archive_sha256": sha(archive),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"members": len(members), "archive_sha256": sha(archive)}))


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "source", "source-record", "binary", "inputs",
        "results", "models", "lineage", "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    seal(parser.parse_args())


if __name__ == "__main__":
    main()
