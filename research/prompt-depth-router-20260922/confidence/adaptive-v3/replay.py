"""Independently verify and replay the archived v3 native measurements."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

from analyze import report as analyze
from costs import cost_table
from learn import read_costs
from oracle import read_sessions, report as calibration_oracle
from quality import evaluate as evaluate_quality, CASES
from render import render as render_markdown


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def unpack(candidate, out):
    manifest_file = candidate / "manifest.json"
    pinned = (candidate / "manifest.sha256").read_text().split()[0]
    if sha(manifest_file) != pinned:
        raise ValueError("v3 external manifest pin differs")
    manifest = json.loads(manifest_file.read_text())
    if (
        manifest["schema"] != 1
        or sha(candidate / "native-data.tar.gz") != manifest["native_archive_sha256"]
        or sha(candidate / "runtime-source-adaptive-v3.tar.gz")
            != manifest["source_archive_sha256"]
        or sha(candidate / "source.json") != manifest["source_record_sha256"]
    ):
        raise ValueError("v3 scientific archive or source pin differs")
    source_record = json.loads((candidate / "source.json").read_text())
    if source_record["source_archive_sha256"] != manifest["source_archive_sha256"]:
        raise ValueError("v3 source recipe does not name the archived source")
    seen = set()
    with tarfile.open(candidate / "native-data.tar.gz", "r:gz") as source:
        for member in source:
            logical = PurePosixPath(member.name)
            if (
                not member.isfile() or logical.is_absolute() or ".." in logical.parts
                or str(logical) != member.name or member.name in seen
                or logical.suffix == ".gguf" or member.size > (100 << 20)
            ):
                raise ValueError("unsafe or duplicate v3 scientific member")
            seen.add(member.name)
            expected = manifest["native_members"].get(member.name)
            if expected is None or expected["bytes"] != member.size:
                raise ValueError("v3 scientific member inventory differs")
            dest = out.joinpath(*logical.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as stream, dest.open("xb") as target:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    target.write(chunk)
            if sha(dest) != expected["sha256"]:
                raise ValueError("v3 scientific member digest differs")
    if seen != set(manifest["native_members"]):
        raise ValueError("v3 scientific archive is incomplete")
    return manifest


def replay(candidate, workloads, out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = unpack(candidate, out)
    workload_manifest = json.loads((workloads / "manifest.json").read_text())
    if sha(workloads / "manifest.json") != manifest["workloads_sha256"]:
        raise ValueError("archived Qwen workload manifest pin differs")
    if sha(Path(__file__).with_name("workloads.py")) != workload_manifest["generator_sha256"]:
        raise ValueError("frozen Qwen workload generator differs")
    frozen = {
        item["file"]: item["sha256"]
        for group in workload_manifest["groups"].values() for item in group
    }
    for name, expected in frozen.items():
        if sha(workloads / name) != expected:
            raise ValueError("frozen Qwen workload changed")
    raw = out / "native"
    calibration = {
        k: [
            raw / f"calibration-{session}-trace-c{k}"
            for session in range(3)
        ]
        for k in (1, 2, 3)
    }
    costs = json.loads((raw / "costs.json").read_text())
    if cost_table(
        calibration, manifest["source_archive_sha256"],
        manifest["model_sha256"],
    ) != costs:
        raise ValueError("measured K-dependent cost table does not replay")
    sessions, sources, excluded = read_sessions(calibration[3])
    expected_oracle = calibration_oracle(
        sessions, read_costs(raw / "costs.json"), sources, excluded,
    )
    expected_oracle["costs_sha256"] = sha(raw / "costs.json")
    archived_oracle = json.loads((raw / "oracle.json").read_text())
    if len(archived_oracle["source_files"]) != len(sources):
        raise ValueError("oracle source inventory differs")
    for actual, expected in zip(archived_oracle["source_files"], sources):
        if (
            actual["sha256"] != expected["sha256"]
            or actual["output_sha256"] != expected["output_sha256"]
        ):
            raise ValueError("oracle trace or output source differs")
    expected_oracle["source_files"] = archived_oracle["source_files"]
    if expected_oracle != archived_oracle:
        raise ValueError("calibration hindsight and fixed-C selection do not replay")
    if not manifest["heldout_completed"]:
        raise ValueError("v3 candidate has no completed heldout comparison")
    outcome = analyze(raw, workload_manifest)
    if (
        manifest["model_sha256"] !=
            "1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a"
        or any(
            item["binary_sha256"] != manifest["binary_sha256"]
            for item in outcome["sources"]
        )
    ):
        raise ValueError("v3 result used another model or binary")
    save(out / "RESULTS.json", outcome)
    if {
        turn["function"]
        for session in workload_manifest["groups"]["heldout"]
        for turn in session["turns"]
    } != set(CASES):
        raise ValueError("functional probe does not cover every frozen request")
    quality = evaluate_quality(raw, workload_manifest)
    save(out / "QUALITY.json", quality)
    (out / "RESULTS.md").write_text(render_markdown(
        outcome, quality,
        json.loads((raw / "costs.json").read_text()),
        json.loads((raw / "oracle.json").read_text()),
    ))
    return {"manifest_sha256": sha(candidate / "manifest.json"),
            "results_sha256": sha(out / "RESULTS.json"),
            "quality_sha256": sha(out / "QUALITY.json"),
            "markdown_sha256": sha(out / "RESULTS.md"),
            "native_members": len(manifest["native_members"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.candidate, args.workloads, args.out), sort_keys=True))


if __name__ == "__main__":
    main()
