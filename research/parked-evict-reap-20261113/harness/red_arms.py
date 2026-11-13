#!/usr/bin/env python3
"""red_arms.py TREE OUTDIR ARM: memra#1278's CPU red arms, one per call, on the build host tree
TREE (which holds the fix). `keep` saves the fix's worker.rs to OUTDIR/fix.worker.rs; each arm
writes its change into TREE; `restore` puts the fix back. Copies take a fresh mtime, so cargo
rebuilds memra-server. Each arm changes one thing, and its named test must fail (cargo exit 101
with the test's panic).

  ladder-reap-off   reclaim_parked_ladder does not reap after an eviction (main's program for the
                    reclaim-on-defer ladder): the ladder test fails
  climb-reap-off    reclaim_climb does not reap after an eviction (main's program for the
                    restorable spare's climb): the climb test fails
  retry-reap-off    alloc_with_single_reclaim_retry does not reap before its retry (main's program
                    for the four reclaim retries): the retry test fails
  main-worker       main's worker.rs (OUTDIR/../main.worker.rs) with the census alone appended:
                    the census fails
  ladder-site-off   the reclaim-on-defer ladder is handed a reap that does nothing: the census fails
  drain-reap-off    the admission drain trims without its reap: the census fails
  settle-reap-off   the prefix reclaim's settle does not reap: the census fails
  spec-miss-reap-off the spec pool's miss retries without its reap: the census fails
"""
import pathlib
import shutil
import sys

tree = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
arm = sys.argv[3]
WORKER = tree / "crates/memra-server/src/worker.rs"
FIX = out / "fix.worker.rs"
MAIN = out.parent / "main.worker.rs"


def sub(old, new):
    s = WORKER.read_text()
    assert s.count(old) == 1, (old[:80], s.count(old))
    WORKER.write_text(s.replace(old, new))
    print(f"{arm}: replaced {old!r} -> {new!r}")


if arm == "keep":
    shutil.copyfile(WORKER, FIX)
elif arm == "restore":
    shutil.copyfile(FIX, WORKER)
elif arm == "ladder-reap-off":
    sub("        evicted += 1;\n        reap();\n", "        evicted += 1;\n        let _ = &mut reap;\n")
elif arm == "climb-reap-off":
    sub("        reap();\n        short = reread()", "        let _ = &mut reap;\n        short = reread()")
elif arm == "retry-reap-off":
    sub("            reap();\n            alloc()\n", "            let _ = reap;\n            alloc()\n")
elif arm == "main-worker":
    fix = FIX.read_text()
    start = fix.index("/// memra#1278: under on-demand KV planes")
    census = fix.index("    /// The census: every parked eviction in the worker")
    module = (
        fix[start:fix.index("mod parked_evict_reap_1278_tests {")]
        + "mod parked_evict_reap_1278_tests {\n"
        + fix[census:]
    )
    WORKER.write_text(MAIN.read_text() + "\n" + module)
    print(f"{arm}: main's worker.rs + the census module ({len(module)} bytes)")
elif arm == "ladder-site-off":
    sub('                            || vmm_reap_for("reclaim-on-defer"),\n', "                            || {},\n")
elif arm == "drain-reap-off":
    sub('                        vmm_reap_for("admission-drain");\n', "")
elif arm == "settle-reap-off":
    sub("    vmm_reap_for(why);\n", "")
elif arm == "spec-miss-reap-off":
    sub('                            vmm_reap_for("spec pool miss");\n', "")
else:
    sys.exit(f"unknown arm {arm}")
