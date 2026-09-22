"""Bank generic research receipts; keep connection, provisioning and credential files private."""
import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def files_under(path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file())


def pack(root, paths, output):
    files = sorted({p for path in paths for p in files_under(path)})
    manifest = {}
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in files:
                    if path.is_symlink() or not path.is_relative_to(root):
                        raise ValueError("Receipt bundle contains a symlink or external file")
                    name = path.relative_to(root).as_posix()
                    manifest[name] = digest(path)
                    info = archive.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
    with tarfile.open(temporary, "r:gz") as archive:
        names = set()
        for member in archive:
            if not member.isfile() or member.name not in manifest or member.name in names:
                raise ValueError("Unexpected receipt archive member")
            names.add(member.name)
            with archive.extractfile(member) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != manifest[member.name]:
                    raise ValueError("Receipt archive content changed")
        if names != set(manifest):
            raise ValueError("Receipt archive is incomplete")
    temporary.replace(output)
    checks = output.with_name(output.name.replace(".tar.gz", ".sha256"))
    checks.write_text(
        "".join(f"{sha}  {name}\n" for name, sha in manifest.items())
    )
    return {"file": output.name, "sha256": digest(output), "bytes": output.stat().st_size,
            "files": len(files), "file_manifest_sha256": digest(checks)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    receipts = root / "receipts"
    done = json.loads((receipts / "experiment/DONE.json").read_text())
    if done != {"families": ["qwen", "gemma"], "sets_each": 10}:
        raise ValueError("The study is not complete")
    qualification = json.loads((receipts / "qualification/status.json").read_text())
    expected = {(family, shape) for family in ["qwen", "gemma"] for shape in ["short", "long"]}
    if (qualification["state"] != "passed"
            or {(row["family"], row["shape"]) for row in qualification["checks"]} != expected):
        raise ValueError("The full qualification bundle is missing")
    for row in qualification["checks"]:
        for mode in ["cold", "warm"]:
            folder = receipts / "qualification" / row[mode]
            records = json.loads((folder / "runs.json").read_text())
            for record in records:
                run = folder / f"{record['seed']}-{record['arm']}"
                for turn in range(1, 9):
                    for kind in ["prompt", "output"]:
                        if not (run / f"turn-{turn}.{kind}.ids").is_file():
                            raise ValueError("A qualification token tape is missing")
    for family in done["families"]:
        audit = json.loads((receipts / f"experiment/{family}-offline-audit.json").read_text())
        if not audit["complete_planned_matrix"] or audit["audited_runs"] != 50 or audit["audited_turns"] != 400:
            raise ValueError("An offline audit is incomplete")
    a.out.mkdir(parents=True, exist_ok=False)
    archives = []
    family_paths = set()
    for family in ["qwen", "gemma"]:
        paths = []
        for directory in ["experiment", "qualification"]:
            paths += sorted((receipts / directory).glob(f"{family}-*"))
        family_paths.update(paths)
        archives.append(pack(receipts, paths, a.out / f"{family}-records.tar.gz"))
    common = [receipts / "workloads"]
    for directory in ["experiment", "qualification"]:
        common += [p for p in (receipts / directory).iterdir() if p not in family_paths]
    for name in ["source.json", "binaries.sha256", "weight-downloads-verified.json",
                 "learner-tests.log", "history-tests.log", "reuse-tests.log",
                 "policy-continuity-audit.json", "selection-rejection-tests.json",
                 "archive-reader-tests.log"]:
        path = receipts / name
        if not path.is_file():
            raise ValueError(f"Required receipt missing: {name}")
        common.append(path)
    archives.append(pack(receipts, common, a.out / "common-records.tar.gz"))
    shutil.copyfile(root / "runtime-source.tar.gz", a.out / "runtime-source.tar.gz")
    shutil.copytree(root / "audit-tools", a.out / "audit-tools", ignore=shutil.ignore_patterns("__pycache__"))
    source = json.loads((receipts / "source.json").read_text())
    if digest(a.out / "runtime-source.tar.gz") != source["runtime_archive_sha256"]:
        raise ValueError("Runtime source archive differs from the measured source")
    result = {"archives": archives, "shared": [], "runtime_source": source,
              "scope": "Generic source, artifacts, token tapes, timing, controls and audits."}
    (a.out / "bundle.json").write_text(json.dumps(result, indent=2) + "\n")
    (a.out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
