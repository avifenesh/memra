#!/usr/bin/env python3
"""red_arms.py SRC OUTDIR ARM: memra#1266's CPU red arms, one per call, on the build host tree
SRC (which holds the fix). Writes the arm's files into SRC; `restore` puts the fix back from
OUTDIR/fix.*. Copies take a fresh mtime (copyfile), so cargo rebuilds every crate a copy
changes, the engine library the server links included. Each arm changes one thing and its named test must fail (cargo exit 101 with the
test's panic).

  accumulation-off  pooled_verify_chunk forgets the members it took (`held` never grows): the
                    walk test fails (K=5 + K=5 at 1,616 MiB pools again)
  fa-growth-off     pooled_verify_chunk leaves the FA pool's growth out: the FA test fails
  rider-cap-off     pooled_rider_take takes every rider: the rider test fails
  margin-low        POOLED_VERIFY_MARGIN at 128 MiB, under the measured 195 MiB: the margin
                    test fails
  main-engine       main's spec.rs (11b368f7d7) + the lockstep census alone: the census fails
  snapshots-off     pre_cohort_need leaves the burst snapshots out: the need test fails
  parked-first      the ladder runs the parked rung before the prefix rung: the ladder test fails
  unreadable-acts   the ladder reads an unreadable device as 0 and evicts: the ladder test fails
  main-worker       main's worker.rs, server lib.rs and engine spec.rs + the Cohorts-pass
                    census alone: the census fails
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
        shutil.copyfile(f, out / f"fix.{n}")


def sub(path, old, new):
    s = path.read_text()
    assert s.count(old) == 1, (path.name, old[:80], s.count(old))
    path.write_text(s.replace(old, new))


def census_only(text, module, census_head):
    at = text.index(module)
    at = text.rindex("#[cfg(test)]", 0, at)
    end = text.index("\n}\n", at) + 3
    mod = text[at:end]
    head = mod[: mod.index("    use super::*;") + len("    use super::*;\n")]
    fn = mod[mod.index(census_head):]
    return head + "\n" + fn


out.mkdir(parents=True, exist_ok=True)
if arm == "keep":
    keep()
elif arm == "restore":
    shutil.copyfile(out / "fix.spec.rs", SPEC)
    shutil.copyfile(out / "fix.worker.rs", WORKER)
    shutil.copyfile(out / "fix.server-lib.rs", SLIB)
elif arm == "accumulation-off":
    sub(SPEC, "        out.held += c.need;\n", "")
elif arm == "fa-growth-off":
    sub(SPEC, "                .saturating_add(fa_growth(out.rows + c.rows, depth))\n", "")
elif arm == "rider-cap-off":
    sub(SPEC, "    let mut take = want.min(rider_depths.len());\n    while take > 0 {",
        "    let mut take = want.min(rider_depths.len());\n    while take > usize::MAX - 1 {")
elif arm == "margin-low":
    sub(SPEC, "pub const POOLED_VERIFY_MARGIN: usize = 256 << 20;",
        "pub const POOLED_VERIFY_MARGIN: usize = 128 << 20;")
elif arm == "main-engine":
    fix = (out / "fix.spec.rs").read_text()
    SPEC.write_text((MAIN / "main.spec.rs").read_text() + "\n" + census_only(
        fix, "mod pooled_verify_cap_1266_tests", "    /// The census: the MTP lockstep"))
elif arm == "snapshots-off":
    sub(WORKER, "    Some(round.saturating_add(snapshots).saturating_add(margin))",
        "    let _ = snapshots;\n    Some(round.saturating_add(margin))")
elif arm == "parked-first":
    sub(WORKER, """    let (entries, bytes) = prefix_rung(need);
    out.prefix_entries = entries;
    out.prefix_bytes = bytes;
    while read().is_some_and(|r| r < need) && parked_rung() {
        out.parked_sessions += 1;
        fence();
    }
""", """    while read().is_some_and(|r| r < need) && parked_rung() {
        out.parked_sessions += 1;
        fence();
    }
    let (entries, bytes) = prefix_rung(need);
    out.prefix_entries = entries;
    out.prefix_bytes = bytes;
""")
elif arm == "unreadable-acts":
    s = WORKER.read_text()
    old = """    match read() {
        None => {
            out.unreadable = true;
            return out;
        }
        Some(r) if r >= need => {
            out.after = r;
            return out;
        }
        Some(_) => {}
    }
"""
    assert s.count(old) == 1
    WORKER.write_text(s.replace(old, """    if read().unwrap_or(0) >= need {
        return out;
    }
""").replace("""    let before = match read() {
        None => {
            out.unreadable = true;
            return out;
        }
""", """    let before = match read().or(Some(0)) {
        None => {
            out.unreadable = true;
            return out;
        }
"""))
elif arm == "main-worker":
    fix = (out / "fix.worker.rs").read_text()
    WORKER.write_text((MAIN / "main.worker.rs").read_text() + "\n" + census_only(
        fix, "mod pre_cohort_reclaim_1266_tests", "    /// The census: the Cohorts pass"))
    shutil.copyfile(MAIN / "main.server-lib.rs", SLIB)
    # main's worker builds main's PlainRiderRow: the engine's spec.rs is main's too.
    shutil.copyfile(MAIN / "main.spec.rs", SPEC)
else:
    raise SystemExit(f"unknown arm {arm}")
print("arm", arm, "written")
