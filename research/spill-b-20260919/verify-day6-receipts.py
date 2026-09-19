#!/usr/bin/env python3
"""Offline day-6 receipt integrity; does not execute CUDA or qualify active tiering."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent / "rented-5090-20260919"
SOURCE = "ac67071e"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def raw(path):
    return path.read_bytes() if path.exists() else gzip.decompress(Path(str(path) + ".gz").read_bytes())


def fields(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def descriptors(value, directory):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            data = raw(directory / value["path"])
            assert len(data) == value["bytes"]
            assert digest(data) == value["sha256"]
        for child in value.values():
            descriptors(child, directory)
    elif isinstance(value, list):
        for child in value:
            descriptors(child, directory)


build = ROOT / "day6-build"
assert (build / "build.exit").read_text().strip() == "0"
binary_hash = (build / "binary.sha256").read_text().split()[0]
count = 0
for name in ["day6-build", "day6-baseline-8192", "day6-baseline-32768", "day6-active-8192"]:
    directory = ROOT / name
    assert (directory / "source.commit").read_text().startswith(SOURCE)
    for row in json.loads((directory / "archive-manifest.json").read_text()):
        if "archive" in row:
            archive = (directory / row["archive"]).read_bytes()
            assert digest(archive) == row["sha256"]
            data = gzip.decompress(archive)
        elif name == "day6-build":
            archive = (directory / row["file"]).read_bytes()
            assert digest(archive) == row["sha256"]
            data = gzip.decompress(archive)
        else:
            data = raw(directory / row["file"])
        assert digest(data) == row["raw_sha256"], (name, row)
        count += 1
    if name == "day6-build":
        continue
    capture = json.loads((directory / "collector/command.capture.json").read_text())
    descriptors(capture, directory / "collector")
    assert capture["qualification"] is False and not capture["timed_out"]
    assert (directory / "hashes.sha256").read_text().split()[0] == binary_hash
    gpu = next(csv.DictReader(raw(directory / "gpu-before.csv").decode().splitlines()))
    assert float(gpu[" power.limit [W]"].split()[0]) == 400
    assert float(gpu[" power.max_limit [W]"].split()[0]) == 600
    assert len(raw(directory / "apps-before.csv").decode().splitlines()) == 1
    receipt = directory / "receipt"
    if "active" in name:
        assert capture["exit_code"] == 2
        refusal = (receipt / "REFUSED.txt").read_text()
        assert "native CUDA materializer + scheduler binding" in refusal
        assert "nonzero demote/reload engagement" in refusal
        assert not (receipt / "BASELINE.txt").exists()
        print("active: explicit native materializer/scheduler refusal, NOT qualification")
        continue
    context = int(name.rsplit("-", 1)[1])
    assert capture["exit_code"] == 0
    baseline = fields(receipt / "BASELINE.txt")
    identity = fields(receipt / "identity.txt")
    assert int(baseline["committed"]) == context
    assert baseline["active_engaged"] == baseline["prefix_engaged"] == "false"
    assert identity["binary_sha256"] == binary_hash
    assert identity["program"] == "native-decode_step_h-tokenwise-trunk-no-mtp"
    assert len((receipt / "prompt.u32le").read_bytes()) == (context - 128) * 4
    assert len((receipt / "tokens.u32le").read_bytes()) == 128 * 4
    for file, key in [("prompt.u32le", "prompt_sha256"), ("plan.debug", "plan_debug_sha256")]:
        assert digest((receipt / file).read_bytes()) == identity[key]
    for file, key in [("prefix-state.tsv", "prefix_state_manifest_sha256"),
                      ("final-state.tsv", "final_state_manifest_sha256"),
                      ("tokens.u32le", "tokens_sha256"), ("logits.tsv", "logit_rows_sha256")]:
        assert digest((receipt / file).read_bytes()) == baseline[key]
    logits = list(csv.DictReader((receipt / "logits.tsv").open(), delimiter="\t"))
    assert [int(row["committed"]) for row in logits] == list(range(context - 128, context + 1))
    final = (receipt / "final-logits.f32le").read_bytes()
    assert digest(final) == logits[-1]["logits_f32le_sha256"]
    assert all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", final))
    for stage, pos in [("prefix", context - 128), ("final", context)]:
        rows = {r["plane"]: r for r in csv.DictReader((receipt / f"{stage}-state.tsv").open(), delimiter="\t")}
        assert rows["position-u64le"]["sha256"] == digest(struct.pack("<Q", pos))
        assert rows["layer-64-absent-unexecuted-mtp"]["valid_bytes"] == "0"
        if stage == "final":
            assert rows["last-logits-f32le"]["sha256"] == digest(final)
    telemetry = csv.DictReader(raw(directory / "collector/command.gpu.csv").decode().splitlines())
    peak = max(int(r[" memory.used [MiB]"].split()[0]) for r in telemetry)
    print(f"baseline {context}: 128 generated, prompt/plan/binary/state/logit hashes intact; peak sampled {peak} MiB; tiers NOT engaged")
print(f"PASS: {count} archived/original file digests and 3 collector descriptor trees; integrity only")
