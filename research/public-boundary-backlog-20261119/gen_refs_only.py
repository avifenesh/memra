#!/usr/bin/env python3
"""Generate the `refs_only` allowlist entries for memra#1357's backlog: versions that the ref scan
reports only at published refs nobody can rewrite, in two reviewed classes.

- `released_box_id`: a version whose only rule is `provider_bare_id` (a rented box named by its
  provider and id). 4784aa555c scrubbed the form from main's live docs on 2026-09-22; these
  versions predate it, and dated receipts on main keep it under their own grants. A box id is no
  credential.
- `auth_test_fixture`: `crates/memra-server/src/auth.rs` versions whose only rule is
  `minted_api_key`, where every key-shaped match sits after the opening of the `mod tests` under
  `#[cfg(test)]` and its random part is one character repeated: test constants, no key material.

A version is listed only when EVERY ref that carries it is frozen: a merged or closed PR's head or
a release tag. A version that any open PR's head or live branch also carries is left out, since a
`refs_only` entry exempts its bytes at every ref: it stays a finding until a commit or a branch
deletion removes it there (review of memra#1372). A `provider_bare_id` version in a file that main
still carries the form in, under a checkout grant, is refused rather than listed: the class's
reason ("a form main no longer carries in this file") would be false for it.

Usage, from a clone that mirrors origin's refs as boundary-refs.yml fetches them:

    gh pr list -R avifenesh/memra --state all --limit 5000 --json number,state > states.json
    python3 research/public-boundary-backlog-20261119/gen_refs_only.py states.json [--write]

`--repo` scans another clone (one that mirrors origin) with this tree's policy and allowlist.
Without --write it prints the entries it would add; with it, it appends them to the allowlist's
`refs_only` section. It refuses (exit 1) when a selected version trips a rule outside its class.
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2] / "tools"
spec = importlib.util.spec_from_file_location("public_boundary", TOOLS / "check-public-boundary.py")
pb = importlib.util.module_from_spec(spec)
sys.modules["public_boundary"] = pb
spec.loader.exec_module(pb)

FIXTURE_PATH = "crates/memra-server/src/auth.rs"
REASONS = {
    "released_box_id": (
        "provider_bare_id (a historical version only at merged or closed PR heads or release "
        "tags: a released rented box named by its provider id, a form main no longer carries in "
        "this file; no credential)"
    ),
    "auth_test_fixture": (
        "minted_api_key (a historical version only at merged or closed PR heads or release "
        "tags: test constants in the #[cfg(test)] mod tests, each key one character repeated "
        "after its prefix; no key material)"
    ),
}


def frozen(ref: str, states: dict) -> bool:
    if ref.startswith("tags/"):
        return True
    match = re.fullmatch(r"origin-pull/([0-9]+)", ref)
    return match is not None and states.get(int(match.group(1))) in ("MERGED", "CLOSED")


TEST_MOD = re.compile(r"^#\[cfg\(test\)\]\s*\n\s*mod tests \{", re.MULTILINE)


def fixture_only(policy, data: bytes) -> bool:
    """Every minted_api_key match sits after the opening of the file's `#[cfg(test)]` `mod tests {`
    (the attribute alone could gate one function or sit in a comment), with a random part of one
    character. The module's end is not bounded: a key after its closing brace would pass."""
    text = data.decode("utf-8", errors="ignore")
    gate = TEST_MOD.search(text)
    keys = list(policy.secret_patterns["minted_api_key"].finditer(text))
    return gate is not None and bool(keys) and all(
        key.start() > gate.end() and len(set(key.group(0).rsplit("-", 1)[-1])) == 1 for key in keys
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("states", type=Path, help="gh pr list --json number,state output")
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--repo", type=Path, help="scan this clone's refs with this tree's policy and allowlist"
    )
    args = parser.parse_args()
    if args.repo is not None:
        pb.ROOT = args.repo.resolve()
    states = {pr["number"]: pr["state"] for pr in json.loads(args.states.read_text())}
    policy = pb.load_policy(pb.POLICY_PATH)
    allowlist = pb.load_allowlist(pb.ALLOWLIST_PATH)
    refs = pb.published_refs(pb.DEFAULT_REF_GLOB)
    if not pb.ref_set_looks_complete(refs):
        print("gen: this clone lacks heads, tags or PR heads; fetch as boundary-refs.yml does")
        return 1
    receipts = {
        e["path"]
        for e in allowlist.values()
        if not pb.is_refs_only(e) and "provider_bare_id" in (e.get("rules") or ())
    }
    violations, carriers = pb.evaluate_refs(policy, refs)
    picked, refused = [], []
    for v in sorted(violations, key=lambda v: (v.path, v.sha256)):
        if (v.path, v.sha256) in allowlist:
            continue  # listed already, for some or all of its rules
        rules = pb.violation_rules(v)
        at = carriers[(v.path, v.sha256)]
        if not all(frozen(ref, states) for ref in at):
            continue  # also on a live ref: a finding there until a commit or a deletion
        if rules == ("provider_bare_id",) and v.path in receipts:
            # main keeps the form in this file under a checkout grant: the class's reason would
            # be false, so the version is left for review.
            refused.append(f"{v.path} {v.sha256[:12]} provider_bare_id in a file main still grants")
            continue
        if rules == ("provider_bare_id",):
            kind = "released_box_id"
        elif rules == ("minted_api_key",) and v.path == FIXTURE_PATH:
            kind = "auth_test_fixture"
        else:
            refused.append(f"{v.path} {v.sha256[:12]} {','.join(rules)}")
            continue
        picked.append((kind, v))
    fixtures = [v for kind, v in picked if kind == "auth_test_fixture"]
    if fixtures:
        # Read each fixture version back and hold it to the class: after `mod tests` opens, one
        # repeated character. A real key in a later version of the file is refused, not listed.
        spec_of = {
            v.sha256: f"{sorted(carriers[(v.path, v.sha256)])[0]}:{v.path}" for v in fixtures
        }
        oids = pb.oids_at(sorted(spec_of.values()))
        loaded = pb.blobs_at(sorted(set(oids.values())))
        for v in fixtures:
            data = loaded.get(oids.get(spec_of[v.sha256], ""))
            if data is None or not fixture_only(policy, data):
                refused.append(f"{v.path} {v.sha256[:12]} minted_api_key outside the fixture class")
    if refused:
        print(f"gen: {len(refused)} frozen versions outside both classes, left for review:")
        for line in refused:
            print(f"  {line}")
    entries = [
        {
            "category": v.category,
            "path": v.path,
            "reason": REASONS[kind],
            "rules": list(pb.violation_rules(v)),
            "scope": pb.REFS_ONLY,
            "sha256": v.sha256,
        }
        for kind, v in picked
        if not any(line.startswith(f"{v.path} {v.sha256[:12]} ") for line in refused)
    ]
    counts = {kind: sum(1 for k, _ in picked if k == kind) for kind in REASONS}
    print(f"gen: {len(entries)} entries ({', '.join(f'{k} {n}' for k, n in counts.items())})")
    rows = sorted((json.dumps(e, sort_keys=True) for e in entries), key=lambda row: (
        json.loads(row)["category"], json.loads(row)["path"], json.loads(row)["sha256"]))
    if not args.write:
        print("\n".join(rows))
        return 1 if refused else 0
    text = pb.ALLOWLIST_PATH.read_text()
    if not any(line.startswith(pb.REFS_ONLY_SECTION[0]) for line in text.splitlines()):
        text += "\n".join(pb.REFS_ONLY_SECTION) + "\n"
    existing = [json.loads(ln) for ln in text.splitlines() if ln.startswith("{")]
    if any(pb.is_refs_only(e) and e["category"] > "secret_pattern" for e in existing):
        print("gen: the refs_only section holds rows that sort after these; merge by hand")
        return 1
    pb.ALLOWLIST_PATH.write_text(text + "".join(row + "\n" for row in rows))
    print(f"gen: appended {len(rows)} rows to {pb.ALLOWLIST_PATH.name}")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
