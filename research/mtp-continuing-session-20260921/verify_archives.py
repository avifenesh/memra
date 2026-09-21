"""Check expanded research archives against the repository's current boundary policy."""
import hashlib
import importlib.util
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

LANE = Path(__file__).resolve().parent
ROOT = LANE.parents[1]

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


def main():
    spec = importlib.util.spec_from_file_location("archive_boundary_policy", ROOT / "tools/check-public-boundary.py")
    boundary = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = boundary
    spec.loader.exec_module(boundary)
    policy = boundary.load_policy(ROOT / "tools/public-boundary-policy.toml")
    allowlist = boundary.load_allowlist(ROOT / "tools/public-boundary-allowlist.jsonl")
    archives = sorted((ROOT / "research/mtp-calibrated-depth-20260920/receipts").glob("*-records.tar.gz"))
    archives += sorted((LANE / "receipts").glob("*-records.tar.gz"))
    archives += [LANE / "receipts/runtime-source.tar.gz"]
    reports = []
    for path in archives:
        print(f"Checking expanded archive: {path.relative_to(ROOT)}", file=sys.stderr, flush=True)
        data = path.read_bytes()
        outer = boundary.evaluate_content(policy, path.relative_to(ROOT).as_posix(), data)
        source = path.name == "runtime-source.tar.gz"
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
                content = archive.extractfile(member).read()
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
        reports.append({
            "file": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "compressed_rules": list(boundary.violation_rules(outer)) if outer else [],
            "expanded_files_checked": checked, "policy_bypass_files": bypassed,
            "source_rule_pins": reviewed, "unresolved_expanded_matches": unresolved,
            "record_rule_pins": reviewed_records,
        })
    result = {
        "policy_sha256": hashlib.sha256((ROOT / "tools/public-boundary-policy.toml").read_bytes()).hexdigest(),
        "archives": reports,
    }
    output = LANE / "receipts/boundary-verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if any(row["unresolved_expanded_matches"] for row in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
