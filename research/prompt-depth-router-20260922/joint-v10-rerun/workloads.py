"""Freeze a fresh transfer evaluation from v11's untouched training split."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


SOURCE_SHA = "71c538295aeb970856f5feb5d11c888aa2dea4d220f41ab844d6e0587108a3df"
STOPPED_SPLIT_SHA = "2142c62394fbb3f90c684a942e09a227f24f8a32d0011f550dd7b6841ed32dc1"
DOMAINS = ("ifeval", "gsm8k")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def build(source_dir, out):
    source_path = source_dir / "manifest.json"
    if sha(source_path) != SOURCE_SHA:
        raise ValueError("frozen v11 source split differs")
    source = json.loads(source_path.read_text())
    if (
        source["schema"] != 1
        or source["v10_manifest_sha256"] != STOPPED_SPLIT_SHA
    ):
        raise ValueError("rerun source lineage differs")
    out.mkdir(parents=True, exist_ok=False)
    groups = {"qualification": {}, "heldout": {}}
    for domain in DOMAINS:
        used = set()
        for target_phase, source_phase, count in (
            ("qualification", "qualification", 1),
            ("heldout", "training", 16),
        ):
            entries = source["groups"][source_phase][domain]
            if len(entries) != count:
                raise ValueError("fresh transfer conversation count differs")
            groups[target_phase][domain] = entries
            for entry in entries:
                src = source_dir / entry["file"]
                if sha(src) != entry["sha256"]:
                    raise ValueError("fresh transfer prompt text differs")
                shutil.copy2(src, out / entry["file"])
                ids = entry["task_ids"]
                if len(ids) != 8 or used.intersection(ids):
                    raise ValueError("fresh transfer tasks repeat")
                used.update(ids)
        if len(used) != 136:
            raise ValueError("fresh transfer task inventory differs")
    result = {
        "schema": 1,
        "scope": "code-trained Qwen C/K/D transfer to fresh non-code tasks",
        "generator_sha256": sha(Path(__file__)),
        "source_v11_manifest_sha256": SOURCE_SHA,
        "excluded_stopped_v10_manifest_sha256": STOPPED_SPLIT_SHA,
        "ifeval_source_sha256": source["ifeval_source_sha256"],
        "gsm8k_source_sha256": source["gsm8k_source_sha256"],
        "groups": groups,
    }
    save(out / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source, args.out)
    print(json.dumps({
        domain: {
            phase: len(result["groups"][phase][domain])
            for phase in ("qualification", "heldout")
        }
        for domain in DOMAINS
    }, sort_keys=True))


if __name__ == "__main__":
    main()
