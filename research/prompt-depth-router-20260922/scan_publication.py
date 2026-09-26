"""Apply the exact public policy to expanded candidate records."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile

SCANNER_SHA = "0000058db5d59a35421999679372f43e760bdf7e72fde72b1748e74a396518aa"
POLICY_SHA = "ef9eebfc074f4b9db708682ec0ac99b3533fe6f4c4433490c3e1cb7fd7f8bd32"
RUNTIME_SHA = "863dc131a061930e3ea8b2a5b01653cd2dc9b82ba89bac426ea0953723ae0251"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    review = json.loads(args.review.read_text())
    manifest_bytes = (args.candidate / "manifest.json").read_bytes()
    if sha(manifest_bytes) != review["candidate_manifest_sha256"]:
        raise ValueError("public candidate manifest changed")
    manifest = json.loads(manifest_bytes)
    if {p.name for p in args.candidate.glob("*.tar.gz")} != {
        "native-data.tar.gz", "runtime-source.tar.gz", "harness-source.tar.gz"
    }:
        raise ValueError("candidate archive set is incomplete")
    for name, record in manifest["files"].items():
        path = args.candidate / name
        content = path.read_bytes()
        if len(content) != record["bytes"] or sha(content) != record["sha256"]:
            raise ValueError("candidate member identity changed")
    runtime = args.candidate / "runtime-source.tar.gz"
    if sha(runtime.read_bytes()) != RUNTIME_SHA:
        raise ValueError("unknown source; no archived scanner loaded")
    with tarfile.open(runtime) as archive:
        scanner = archive.extractfile("tools/check-public-boundary.py").read()
        policy_bytes = archive.extractfile("tools/public-boundary-policy.toml").read()
    if sha(scanner) != SCANNER_SHA or sha(policy_bytes) != POLICY_SHA:
        raise ValueError("scanner or policy identity changed")
    with tempfile.TemporaryDirectory(prefix="public-boundary-policy-") as directory:
        path = Path(directory)
        (path / "scanner.py").write_bytes(scanner)
        (path / "policy.toml").write_bytes(policy_bytes)
        spec = importlib.util.spec_from_file_location("pinned_public_boundary", path / "scanner.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        policy = module.load_policy(path / "policy.toml")
        matches, archives = [], []
        prior = {(r["file"], r["sha256"]): set(r["rules"]) for r in review["source_rule_pins"]}
        approved = {
            (r["archive"], r["file"], r["sha256"], rule)
            for r in review["record_rule_pins"] for rule in r["rules"]
        }
        for archive_path in sorted(args.candidate.glob("*.tar.gz")):
            data = archive_path.read_bytes()
            archives.append({"file": archive_path.name, "sha256": sha(data), "bytes": len(data)})
            rel = "research/prompt-depth-router-20260922/native-receipts/" + archive_path.name
            outer = module.evaluate_content(policy, rel, data)
            if outer:
                matches.append({"archive": archive_path.name, "file": "@compressed",
                                "sha256": sha(data), "rules": sorted(module.violation_rules(outer)),
                                "prior_source_approval": False})
            source_archive = archive_path.name != "native-data.tar.gz"
            with tarfile.open(archive_path) as stream:
                for member in stream:
                    if not member.isfile():
                        raise ValueError("candidate contains a non-regular member")
                    content = stream.extractfile(member).read()
                    if source_archive:
                        violation = module.evaluate_content(policy, member.name, content)
                        rules = set(module.violation_rules(violation)) if violation else set()
                    else:
                        rules = {v[0] for v in module.scan_secret_bytes(
                            content, policy.secret_union, policy.secret_groups, policy.secret_patterns)}
                    if rules:
                        digest = sha(content)
                        matches.append({
                            "archive": archive_path.name, "file": member.name,
                            "sha256": digest, "rules": sorted(rules),
                            "prior_source_approval": source_archive and rules <= prior.get((member.name, digest), set()),
                        })
    unapproved = [
        row for row in matches if not row["prior_source_approval"]
        and any((row["archive"], row["file"], row["sha256"], rule) not in approved for rule in row["rules"])
    ]
    report = {"scanner_sha256": SCANNER_SHA, "policy_sha256": POLICY_SHA,
              "archives": archives, "matches": matches, "unapproved": len(unapproved)}
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"))
    print("SCAN_META " + json.dumps({k: v for k, v in report.items() if k != "matches"}))
    for row in matches:
        print("SCAN_MATCH " + json.dumps(row))
    print("SCAN_DONE " + sha(encoded.encode()))
    if unapproved:
        raise SystemExit("expanded public-boundary matches need exact content review")


if __name__ == "__main__":
    main()
