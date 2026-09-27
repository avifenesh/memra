"""Bank v7 q=0 diagnostics separately from corrected v8 measurements."""

import argparse
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tarfile


SOURCE_SHA = "4c533a5bdc42661e0bcaff81ae4ee100ec7c13ceb00bfbd3791c45a48d5def75"
BINARY_SHA = "d229aba5d59fa398f47b489975fb1ffed2285c924da02486a955598d5e083777"
DEBUG_PATCH_SHA = "59336e7fbf34fd48cae321e8b6bc0c244c02a3521690318b844723778cd85139"
DEBUG_BINARY_SHA = "a35ab5c084ee87ab70b92239ab3786c4ac1c889da80901b396c425602a28a011"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def seal(args):
    if (
        sha(args.source) != SOURCE_SHA or sha(args.binary) != BINARY_SHA
        or sha(args.debug_patch) != DEBUG_PATCH_SHA
        or sha(args.debug_binary) != DEBUG_BINARY_SHA
    ):
        raise ValueError("v7 diagnostic source, binary or patch differs")
    if json.loads(args.source_record.read_text())["source_archive_sha256"] != SOURCE_SHA:
        raise ValueError("v7 source record differs")
    if (args.results / "heldout-summary.json").exists():
        raise ValueError("v7 diagnostic opened fresh heldout")
    engagement = json.loads((args.results / "engagement-result.json").read_text())
    if not (
        engagement["source_sha256"] == SOURCE_SHA
        and engagement["binary_sha256"] == BINARY_SHA
        and engagement["confidence_decisions"]
        == engagement["confidence_stops"] == 612
        and engagement["format_pass"] == engagement["functional_pass"] == 8
        and engagement["loops"] == 0
    ):
        raise ValueError("v7 always-stop engagement receipt differs")
    if args.debug_log.read_text().count("[ckd-c-debug] pos=0 q=0 ") < 4:
        raise ValueError("v7 native q=0 diagnostic differs")
    named = {
        "source/runtime-source-joint-v7.tar.gz": args.source,
        "source/source.json": args.source_record,
        "source/mtp-depth-study": args.binary,
        "diagnostic/debug.patch": args.debug_patch,
        "diagnostic/debug-depth-study": args.debug_binary,
        "diagnostic/debug-qualifier.stderr.log": args.debug_log,
    }
    for path in sorted(args.results.rglob("*")):
        if path.is_symlink():
            raise ValueError("v7 diagnostic has a symlink")
        if path.is_file():
            named[f"native/{path.relative_to(args.results).as_posix()}"] = path
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
        "scope": "v7 single-head learned C diagnostic; prime graph q=0; no fresh heldout",
        "source_archive_sha256": SOURCE_SHA,
        "binary_sha256": BINARY_SHA,
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
        "source", "source-record", "binary", "debug-patch",
        "debug-binary", "debug-log", "results", "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    seal(parser.parse_args())


if __name__ == "__main__":
    main()
