"""Check expanded research archives against the repository's current boundary policy."""
import argparse
import hashlib
import importlib.util
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

LANE = Path(__file__).resolve().parent
ROOT = LANE.parents[1]

ARCHIVE_COUNTS = {
    "research/mtp-calibrated-depth-20260920/receipts/gemma-records.tar.gz": 4343,
    "research/mtp-calibrated-depth-20260920/receipts/qwen-records.tar.gz": 4033,
    "research/mtp-continuing-session-20260921/receipts/common-records.tar.gz": 30,
    "research/mtp-continuing-session-20260921/receipts/gemma-records.tar.gz": 2493,
    "research/mtp-continuing-session-20260921/receipts/qwen-records.tar.gz": 4140,
    "research/mtp-continuing-session-20260921/receipts/runtime-source.tar.gz": 1257,
}

RECORD_RULE_PINS = {
    (
        "research/mtp-calibrated-depth-20260920/receipts/gemma-records.tar.gz",
        "gemma-mtp-followup-calibration-0/20263000-k2/turn-7.answer.txt",
    ): {
        "sha256": "8ebaff29d51fb516486a97dbff12133f910b0e148a0334923bb591c9646d2c5c",
        "rules": ["provider_name_aws"],
        "reason": "Generated hypothetical instance-type example; no deployment or account identity.",
    },
}


def archive_metadata():
    declared = {}
    runtime = None
    for dirname in ["mtp-calibrated-depth-20260920", "mtp-continuing-session-20260921"]:
        directory = f"research/{dirname}/receipts"
        manifest = json.loads((ROOT / directory / "manifest.json").read_text())
        expected = {name for name in ARCHIVE_COUNTS if name.startswith(directory + "/")
                    and not name.endswith("runtime-source.tar.gz")}
        rows = manifest["archives"]
        names = [f"{directory}/{row['file']}" for row in rows]
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError(f"Missing, duplicate or unexpected archive declaration in {dirname}")
        for name, row in zip(names, rows):
            declared[name] = row
        if dirname == "mtp-continuing-session-20260921":
            runtime = manifest["runtime_source"]
            declared[f"{directory}/runtime-source.tar.gz"] = {
                "sha256": runtime["runtime_archive_sha256"], "bytes": runtime["archive_bytes"],
            }
    return declared, runtime


def member_hashes(path, row, expected_count):
    checks = path.with_name(path.name.replace(".tar.gz", ".sha256"))
    data = checks.read_bytes()
    if hashlib.sha256(data).hexdigest() != row["file_manifest_sha256"]:
        raise ValueError(f"Member manifest hash changed: {path}")
    expected = {}
    for line in data.decode().splitlines():
        sha, name = line.split("  ", 1)
        if name in expected:
            raise ValueError(f"Duplicate member manifest entry: {path}/{name}")
        expected[name] = sha
    if len(expected) != expected_count or row.get("files", expected_count) != expected_count:
        raise ValueError(f"Member manifest count changed: {path}")
    return expected


def store_result(output, result, check):
    if check:
        if json.loads(output.read_text()) != result:
            raise ValueError("Committed boundary verification differs from the reproduced result")
    else:
        output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare with the committed receipt without rewriting it")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("archive_boundary_policy", ROOT / "tools/check-public-boundary.py")
    boundary = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = boundary
    spec.loader.exec_module(boundary)
    policy = boundary.load_policy(ROOT / "tools/public-boundary-policy.toml")
    allowlist = boundary.load_allowlist(ROOT / "tools/public-boundary-allowlist.jsonl")
    declared, runtime = archive_metadata()
    archived_runtime = None
    reports = []
    for relative, expected_count in ARCHIVE_COUNTS.items():
        path = ROOT / relative
        print(f"Checking expanded archive: {path.relative_to(ROOT)}", file=sys.stderr, flush=True)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Required regular archive missing: {path}")
        data = path.read_bytes()
        row = declared[relative]
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"Archive hash or size changed: {path}")
        outer = boundary.evaluate_content(policy, path.relative_to(ROOT).as_posix(), data)
        source = path.name == "runtime-source.tar.gz"
        expected = None if source else member_hashes(path, row, expected_count)
        seen = set()
        checked = bypassed = 0
        reviewed = []
        reviewed_records = []
        unresolved = []
        with tarfile.open(path) as archive:
            for member in archive:
                if source and member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"Non-regular archive member: {path.name}/{member.name}")
                if member.name in seen:
                    raise ValueError(f"Duplicate archive member: {path.name}/{member.name}")
                seen.add(member.name)
                content = archive.extractfile(member).read()
                if expected is not None and hashlib.sha256(content).hexdigest() != expected.get(member.name):
                    raise ValueError(f"Unexpected or changed archive member: {path.name}/{member.name}")
                if path.name == "common-records.tar.gz" and member.name == "source.json":
                    archived_runtime = json.loads(content)
                checked += 1
                if source:
                    if boundary.is_bypass(member.name, policy.bypass_paths):
                        bypassed += 1
                    violation = boundary.evaluate_content(policy, member.name, content)
                    if violation:
                        covered = boundary.exempt_rules(allowlist, violation) or frozenset()
                        entry = allowlist.get((violation.path, violation.sha256))
                        if entry and boundary.entry_expired(entry, datetime.now(timezone.utc).date().isoformat()):
                            covered = frozenset()
                        missing = [rule for rule in boundary.violation_rules(violation) if rule not in covered]
                        item = {"file": member.name, "sha256": violation.sha256,
                                "rules": list(boundary.violation_rules(violation))}
                        (unresolved if missing else reviewed).append(item)
                else:
                    hits = boundary.scan_secret_bytes(
                        content, policy.secret_union, policy.secret_groups, policy.secret_patterns,
                    )
                    if hits:
                        rules = [hit[0] for hit in hits]
                        sha = hashlib.sha256(content).hexdigest()
                        pin = RECORD_RULE_PINS.get((path.relative_to(ROOT).as_posix(), member.name))
                        covered = set(pin["rules"]) if pin and pin["sha256"] == sha else set()
                        missing = [rule for rule in rules if rule not in covered]
                        item = {"file": member.name, "sha256": sha, "rules": rules}
                        if missing:
                            unresolved.append(item)
                        else:
                            reviewed_records.append({**item, "reason": pin["reason"]})
        if checked != expected_count or (expected is not None and seen != set(expected)):
            raise ValueError(f"Archive member coverage changed: {path}")
        reports.append({
            "file": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "compressed_rules": list(boundary.violation_rules(outer)) if outer else [],
            "expanded_files_checked": checked, "policy_bypass_files": bypassed,
            "source_rule_pins": reviewed, "unresolved_expanded_matches": unresolved,
            "record_rule_pins": reviewed_records,
        })
    if runtime != archived_runtime:
        raise ValueError("Runtime source declaration differs from the sealed common receipt")
    result = {
        "policy_sha256": hashlib.sha256((ROOT / "tools/public-boundary-policy.toml").read_bytes()).hexdigest(),
        "archives": reports,
    }
    output = LANE / "receipts/boundary-verification.json"
    print(json.dumps(result, indent=2))
    if any(row["unresolved_expanded_matches"] for row in reports):
        raise SystemExit(1)
    store_result(output, result, args.check)
    if args.check:
        print("Committed boundary verification reproduced exactly.", file=sys.stderr)


if __name__ == "__main__":
    main()
