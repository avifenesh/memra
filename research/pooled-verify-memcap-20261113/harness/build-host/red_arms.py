#!/usr/bin/env python3
"""red_arms.py SRC OUTDIR ARM: memra#1266's CPU red arms, one per call, on the CPU build host tree SRC
(which holds the fix). Writes the arm's files into SRC; `restore` puts the fix back from
OUTDIR/fix.*. Each arm must make its named test fail (cargo exit 101 with the test's panic).

  admits-open   pooled_verify_admits lets every member join (the cap off): the join-rule unit
                test fails.
  main-engine   main's spec.rs (11b368f7d7) + the lockstep census alone: the census fails.
  shortfall-off spec_round_shortfall never reports a shortfall: the trigger unit test fails.
  main-worker   main's worker.rs and server lib.rs + the Cohorts-pass census alone: the census
                fails.
"""
import pathlib
import shutil
import sys

src = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
arm = sys.argv[3]
SPEC = src / "crates/memra-engine/src/spec.rs"
WORKER = src / "crates/memra-server/src/worker.rs"
SLIB = src / "crates/memra-server/src/lib.rs"
MAIN = out.parent  # main.spec.rs, main.worker.rs, main.server-lib.rs live one level up


def keep():
    for f, n in ((SPEC, "spec.rs"), (WORKER, "worker.rs"), (SLIB, "server-lib.rs")):
        dst = out / f"fix.{n}"
        if not dst.exists():
            shutil.copy2(f, dst)


def sub(path, old, new):
    s = path.read_text()
    assert s.count(old) == 1, (path.name, old[:80], s.count(old))
    path.write_text(s.replace(old, new))


def module(text, start, end_marker="\n}\n"):
    """The text of `mod NAME { .. }` cut from the fix's file, from its attribute line."""
    at = text.index(start)
    at = text.rindex("#[cfg(test)]", 0, at)
    end = text.index(end_marker, at) + len(end_marker)
    return text[at:end]


out.mkdir(parents=True, exist_ok=True)
keep()
fix_spec = (out / "fix.spec.rs").read_text()
fix_worker = (out / "fix.worker.rs").read_text()
if arm == "restore":
    shutil.copy2(out / "fix.spec.rs", SPEC)
    shutil.copy2(out / "fix.worker.rs", WORKER)
    shutil.copy2(out / "fix.server-lib.rs", SLIB)
elif arm == "admits-open":
    sub(SPEC, "    held == 0\n        || held\n", "    true || held\n")
elif arm == "shortfall-off":
    sub(WORKER, "    (allocatable < need).then(|| need - allocatable)\n",
        "    let _ = (allocatable, need);\n    None\n")
elif arm == "main-engine":
    census = module(fix_spec, "mod pooled_verify_cap_1266_tests")
    # Only the census function: the unit tests call functions main does not have.
    head = census[: census.index("    const MIB")]
    fn = census[census.index("    /// The census: the MTP lockstep"):]
    SPEC.write_text((MAIN / "main.spec.rs").read_text() + "\n" + head + fn)
elif arm == "main-worker":
    census = module(fix_worker, "mod pre_cohort_reclaim_1266_tests")
    head = census[: census.index("    const MIB")]
    fn = census[census.index("    /// The census: the Cohorts pass"):]
    WORKER.write_text((MAIN / "main.worker.rs").read_text() + "\n" + head + fn)
    shutil.copy2(MAIN / "main.server-lib.rs", SLIB)
else:
    raise SystemExit(f"unknown arm {arm}")
print("arm", arm, "written")
