#!/usr/bin/env python3
"""Census check for the cell-tree sparse checkout on one host.

  python3 census_check.py REPO COMMIT TOOLS_DIR [TREE]

TOOLS_DIR is the tools/ directory of the code under test (main's or the branch's), exported with
`git archive`. Prints the host's git version, whether the include census answered, the kept
research files (count and sha256 of the newline-joined list), the sparse patterns' sha256 and the
inputs digest of COMMIT. With TREE, also builds the cell tree there with that code's `tree_at` and
prints whether a research receipt is checked out and the tree's size.
"""
import hashlib
import subprocess
import sys
from pathlib import Path

repo, commit, tools = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
tree = Path(sys.argv[4]) if len(sys.argv) > 4 else None
sys.path.insert(0, str(tools))
import cell_evidence as ce  # noqa: E402

commit = ce.resolve(repo, commit)
print("host", subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip())
print("git", subprocess.run(["/usr/bin/git", "--version"], capture_output=True, text=True).stdout.strip())
print("commit", commit)
kept = ce.research_kept(repo, commit)
if kept is None:
    print("census none (whole checkout)")
else:
    names = [n for n in ce.git(repo, "ls-tree", "-r", "-z", "--name-only", commit, "--", "research")
             .decode().split("\0") if n]
    print("census kept", len(kept), hashlib.sha256("\n".join(kept).encode()).hexdigest())
    pats = ce.sparse_patterns(names, kept)
    print("patterns", len(pats), hashlib.sha256("\n".join(pats).encode()).hexdigest())
print("inputs_digest", ce.inputs_digest(repo, commit))
# The census's other callers on this host: a research-only pair (the record-reuse check) and a
# code pair's change impact (what `need` prints).
drift = ce.research_drift(repo, ce.resolve(repo, "30aae1b8c4"), commit)
print("research_drift 30aae1b8c4..", "none" if drift is None else
      f"{len(drift)} {hashlib.sha256(chr(10).join(drift).encode()).hexdigest()}")
impact, _, _ = ce.ci.impact_between(repo, ce.resolve(repo, "1f3ede4c0e"), ce.resolve(repo, "30aae1b8c4"))
text = ce.ci.render(impact)
print("need 1f3ede4c0e..30aae1b8c4", hashlib.sha256(text.encode()).hexdigest()[:16],
      text.splitlines()[0][:80] if text else "")
if tree is not None:
    ce.tree_at(repo, commit, tree)
    receipts = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", commit, "--",
                               "research/c8-serving-20261107/runs"], capture_output=True, text=True).stdout.split()
    sample = receipts[0] if receipts else None
    print("receipt_checked_out", sample is not None and (tree / sample).exists(), sample)
    du = subprocess.run(["du", "-sh", "--exclude=target", str(tree)], capture_output=True, text=True).stdout.split()
    print("tree_size", du[0] if du else "?")
