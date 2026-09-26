"""Reproduce the CPU report from hash-bound raw samples; never execute an archive."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

GROUPS = [
    ("classified_instructions", "Classified fixture instructions"),
    ("semantic_fallbacks", "Mixed/unknown fixture instructions"),
    ("quoted_16k_input", "16 KiB quoted input"),
    ("large_marked_reference", "About 210 KB marked reference"),
    ("near_byte_limit_quoted_input", "256 KiB quoted input"),
    ("word_limit_fallback", "512-word limit fallback"),
    ("byte_limit_fallback", "Oversized-input fallback"),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def report(receipts, expected_pin):
    raw = (receipts / "manifest.json").read_bytes()
    if sha(raw) != expected_pin:
        raise ValueError("CPU manifest identity changed")
    manifest = json.loads(raw)
    expected_files = set(manifest["files"]) | {"manifest.json", "manifest.sha256"}
    if {p.name for p in receipts.iterdir()} != expected_files:
        raise ValueError("CPU receipt inventory is not closed")
    for name, record in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("unsafe receipt path")
        path = receipts / name
        data = path.read_bytes()
        if path.is_symlink() or len(data) != record["bytes"] or sha(data) != record["sha256"]:
            raise ValueError("CPU receipt identity changed: " + name)
    metadata = json.loads((receipts / "metadata.json").read_text())
    status = json.loads((receipts / "status.json").read_text())
    if status["status"] != "passed" or any(c["returncode"] for c in status["commands"]):
        raise ValueError("CPU qualification did not pass")
    with tarfile.open(receipts / "source.tar.gz") as archive:
        names = [m.name for m in archive]
        if len(names) != len(set(names)) or set(names) != set(manifest["source_files"]):
            raise ValueError("source inventory changed")
        for name, digest in manifest["source_files"].items():
            member = archive.getmember(name)
            if Path(name).name != name or not member.isfile():
                raise ValueError("unsafe source member")
            if sha(archive.extractfile(member).read()) != digest:
                raise ValueError("source member identity changed")
        for name, digest in metadata["source_hashes"].items():
            if manifest["source_files"][name] != digest:
                raise ValueError("source and measurement bindings disagree")
    timing_data = gzip.decompress((receipts / "timings.jsonl.gz").read_bytes())
    if len(timing_data) != manifest["timing_bytes"] or sha(timing_data) != manifest["timing_sha256"]:
        raise ValueError("expanded timing identity changed")
    records = [json.loads(line) for line in timing_data.splitlines()]
    expected_cells = {(name, batch) for name, _ in GROUPS for batch in range(5)}
    if len(records) != len(expected_cells) or {(r["group"], r["batch"]) for r in records} != expected_cells:
        raise ValueError("missing or duplicate timing batch")
    by_group = {}
    for row in records:
        values = row["samples_ns"]
        if len(values) != 20000 or row["calls"] != 20000 or any(type(v) is not int or v < 0 for v in values):
            raise ValueError("invalid raw timing samples")
        ordered = sorted(values)
        for p in (50, 95, 99):
            if ordered[(len(ordered) - 1) * p // 100] != row[f"p{p}_ns"]:
                raise ValueError("batch percentile differs from raw samples")
        if ordered[-1] != row["max_ns"]:
            raise ValueError("batch maximum differs from raw samples")
        by_group.setdefault(row["group"], []).extend(values)
    startup = sorted(json.loads((receipts / "process-startup.json").read_text())["samples_ns"])
    tests = (receipts / "tests.stdout").read_text()
    if '{"cases":95,"failures":0}' not in tests or "11 passed; 0 failed" not in tests:
        raise ValueError("behavioral coverage record changed")
    fitting_tests = (receipts / "profile-tests.stderr").read_text()
    if "Ran 4 tests" not in fitting_tests or "\nOK\n" not in fitting_tests:
        raise ValueError("profile-fitting regressions did not complete")
    text = [
        "# CPU classifier result", "",
        "The bounded lexical classifier passed 95/95 fixed synthetic behavioral cases,",
        "11 Rust regressions and four profile-fitting regressions. These cases establish",
        "the documented behaviors; they are not a blind estimate of real-world accuracy.", "",
        f"Host: {metadata['cpu_model']}, {metadata['logical_cpus']} logical CPUs available,",
        f"`{metadata['machine']}`. Rust 1.97.1, release optimization. The function is",
        "single-threaded and in-process. Timings include the measurement clock's overhead.", "",
        "| Input group | Calls | Median us | p95 us | p99 us | Maximum us |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in GROUPS:
        values = sorted(by_group[name])
        p = [values[(len(values) - 1) * rank // 100] / 1000 for rank in (50, 95, 99)]
        text.append(f"| {label} | {len(values):,} | {p[0]:.3f} | {p[1]:.3f} | {p[2]:.3f} | {values[-1]/1000:.3f} |")
    text += [
        "",
        "Each group has five interleaved batches of 20,000 calls. Successful classification,",
        "semantic fallback and input-limit fallback remain separate. Maximum observed",
        "times include scheduling outliers; the implementation makes no real-time deadline claim.",
        "",
        f"The standalone process plus stdin/stdout and Python orchestration measured",
        f"{startup[(len(startup)-1)//2]/1e6:.3f} ms median over {len(startup)} launches.",
        "The native study calls the Rust function directly instead of launching this CLI per request.",
        "",
        "This is CPU component evidence. It establishes no model-decoding speedup.",
        "",
        f"Source recipe: `{metadata['source_commit']}`.",
        f"Measured executable SHA-256: `{metadata['binary_sha256']}`.",
        f"CPU manifest SHA-256: `{expected_pin}`.",
        "Exact source, raw per-call samples, interface checks and compiler/command identity",
        "are in `cpu-receipts/`. Exact executables and complete CI job output are retained privately.",
        "",
    ]
    return "\n".join(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    text = report(args.receipts, args.manifest_sha256)
    if args.check:
        if args.check.read_text() != text:
            raise SystemExit("CPU report differs from raw receipts")
        print("CPU report reproduced from all 700,000 raw samples")
    else:
        print(text, end="")
