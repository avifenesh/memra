"""Package complete scientific receipts; provider and credential custody stay private."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ("records", "runtime_artifact", "staging", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    state = json.loads((args.records / "native/status.json").read_text())
    if state["status"] != "completed":
        raise ValueError("native execution has not completed")
    args.out.mkdir(parents=True, exist_ok=False)
    selected = {}
    for folder in ("native", "workloads"):
        for path in (args.records / folder).rglob("*"):
            if path.is_file():
                if path.is_symlink():
                    raise ValueError("receipt symlink")
                selected[str(path.relative_to(args.records))] = path
    for name in ("source.json", "native-build-source.json", "weights-verified.json", "classifier-cases.tsv"):
        selected[name] = args.records / name
    # This file has only OS/GPU/CUDA-version metadata, no provider or location.
    selected["hardware.json"] = args.records / "hardware.private.json"
    members = []
    with tarfile.open(args.out / "native-data.tar.gz", "w:gz") as archive:
        for name, path in sorted(selected.items()):
            if not path.is_file() or path.is_symlink():
                raise ValueError("missing or non-regular receipt: " + name)
            members.append({"file": name, "bytes": path.stat().st_size, "sha256": sha(path)})
            archive.add(path, arcname=name, recursive=False)
    inventory = args.out / "native-data.members.jsonl"
    inventory.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in members))
    shutil.copyfile(args.runtime_artifact / "runtime-source.tar.gz", args.out / "runtime-source.tar.gz")
    shutil.copyfile(args.staging / "harness-source.tar.gz", args.out / "harness-source.tar.gz")
    source = json.loads((args.records / "source.json").read_text())
    if sha(args.out / "runtime-source.tar.gz") != source["runtime_source_sha256"]:
        raise ValueError("runtime archive differs from the executed source")
    if sha(args.out / "harness-source.tar.gz") != source["harness_source_sha256"]:
        raise ValueError("orchestration archive differs from the executed source")
    manifest = {
        "schema": 1,
        "source_recipe_commit": source["source_recipe_commit"],
        "harness_commit": source["harness_commit"],
        "member_count": len(members),
        "expanded_data_bytes": sum(r["bytes"] for r in members),
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(args.out.iterdir())
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pin = sha(args.out / "manifest.json")
    (args.out / "manifest.sha256").write_text(pin + "  manifest.json\n")
    print(json.dumps({"manifest_sha256": pin, "members": len(members)}))


if __name__ == "__main__":
    main()
