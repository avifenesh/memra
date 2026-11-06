#!/usr/bin/env python3
"""Red arm for serve-smoke arm 13: run the lane's tools/serve-smoke.sh against the landing tip's
hashed cell-evidence build (base tree at 0a34ce3444, the code main has), the way
cell_evidence.run_serving does, then restore the tree's own script.
usage: serve_smoke_red.py WT TREE OUT"""
import shutil
import subprocess
import sys
from pathlib import Path

wt, tree, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(wt / "tools"))
import cell_evidence as ce  # noqa: E402

head = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
assert head.startswith("0a34ce3444"), head
before = ce.binary_ids(tree / "target/release")
shutil.copy2(wt / "tools/serve-smoke.sh", tree / "tools/serve-smoke.sh")
out.mkdir(parents=True, exist_ok=True)
try:
    rc = ce.run_serving(tree, {}, out, ce.build_env("120a", tree))
finally:
    subprocess.run(["git", "-C", str(tree), "checkout", "-q", "--", "tools/serve-smoke.sh"], check=True)
same = ce.binary_ids(tree / "target/release") == before
(out / "run.txt").write_text(f"tree {head}\nserve-smoke exit {rc}\nbinaries unchanged {same}\n"
                             f"memra-server sha256 {before['memra-server']['sha256']}\n")
print(f"serve-smoke exit {rc}; binaries unchanged {same}")
