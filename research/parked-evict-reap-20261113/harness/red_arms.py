#!/usr/bin/env python3
"""red_arms.py TREE OUTDIR ARM: memra#1278's CPU red arms, one per call, on the build host tree
TREE (which holds the fix). `keep` saves the fix's worker.rs to OUTDIR/fix.worker.rs; each arm
writes its change into TREE; `restore` puts the fix back. Copies take a fresh mtime, so cargo
rebuilds memra-server. Each arm changes one thing, and its named test must fail (cargo exit 101
with the test's panic).

The helpers' own program (a fake-device test fails):
  ladder-reap-off   reclaim_parked_ladder does not reap after an eviction (main's program for the
                    reclaim-on-defer ladder): the ladder test
  climb-reap-off    reclaim_climb does not reap after an eviction (main's program for the
                    restorable spare's climb): the climb test
  retry-reap-off    alloc_with_single_reclaim_retry does not reap before its retry (main's program
                    for the reclaim retries): the retry test
  retry-rule-off    the retry runs only when the reclaim freed something (the rule before the
                    review): the reap-alone retry test
  retry-trim-off    reap_then_trim does not trim: the flushed-prefix retry test
  flush-reap-off    flush_then_reap does not reap (the failed materialization's flush before the
                    review): the materialization test
  rung-reap-off     spec_right_size_ladder does not reap a failed rung (the ladder before the
                    review): the rung test
  reap-line-always  vmm_reap_line prints for every reap: the reap-line test

The call sites (the census fails):
  main-worker       main's worker.rs (OUTDIR/../main.worker.rs) with the census alone appended
  ladder-site-off   the reclaim-on-defer ladder is handed a reap that does nothing
  drain-reap-off    the admission drain trims without its reap
  settle-reap-off   the prefix reclaim's settle does not reap
  spec-miss-reap-off the spec pool's miss retries without its reap
  flush-site-off    the failed materialization's flush is handed a reap that does nothing
  park-compact-fail-off the park-compact grow's failure drops the session without a reap
  affinity-fail-drop-off the plain-affinity failure reaps before the entry drops
  right-size-site-off the right-size ladder is handed a reap that does nothing
  shed-reap-off     the kv-flex shed does not reap what it shed
  probe-prefix-reread  a new prefix flush followed by a reading, no settle and no reap
  probe-parked-reread  a new parked eviction followed by a reading, no reap
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
ANCHOR = "\n#[cfg(test)]\nmod tests {\n"


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
    sub(
        "            let released = reap(&first_err);\n",
        "            let released = {\n                let _ = reap;\n                0\n            };\n",
    )
elif arm == "retry-rule-off":
    sub(
        "            if reclaimed || released > 0 {\n",
        "            let _ = released;\n            if reclaimed {\n",
    )
elif arm == "retry-trim-off":
    sub("    reaped.saturating_add(trim())\n", "    let _ = trim;\n    reaped\n")
elif arm == "flush-reap-off":
    sub("    let evicted = flush();\n    reap();\n", "    let evicted = flush();\n    let _ = reap;\n")
elif arm == "rung-reap-off":
    sub("        reap();\n        if ask <= need {\n", "        let _ = &mut reap;\n        if ask <= need {\n")
elif arm == "reap-line-always":
    sub(
        "    (released > 0 || pending > 0 || VMM_REAP_TEARDOWNS.contains(&why))\n",
        "    (released > 0 || pending > 0 || VMM_REAP_TEARDOWNS.contains(&why) || true)\n",
    )
elif arm == "main-worker":
    fix = FIX.read_text()
    start = fix.index("/// memra#1278: under on-demand KV planes")
    census = fix.index("    /// The census: every eviction in the worker")
    module = (
        fix[start : fix.index("mod parked_evict_reap_1278_tests {")]
        + "mod parked_evict_reap_1278_tests {\n"
        + fix[census:]
    )
    WORKER.write_text(MAIN.read_text() + "\n" + module)
    print(f"{arm}: main's worker.rs + the census module ({len(module)} bytes)")
elif arm == "ladder-site-off":
    sub('                                vmm_reap_for("reclaim-on-defer");\n', "")
elif arm == "drain-reap-off":
    sub('                        vmm_reap_for("admission-drain");\n', "")
elif arm == "settle-reap-off":
    sub("    let reaped = vmm_reap_for(why);\n", "    let reaped = 0;\n")
elif arm == "spec-miss-reap-off":
    sub(
        'let released = vmm_retry_reap(engine, loaded, &first_err, "spec pool miss");',
        "let released = 0;",
    )
elif arm == "flush-site-off":
    sub('vmm_reap_trim_for(engine, loaded, "prefix materialization flush");\n', "\n")
elif arm == "park-compact-fail-off":
    sub('                vmm_reap_trim_for(engine, loaded, "park-compact grow failed");\n', "")
elif arm == "affinity-fail-drop-off":
    sub("                    drop(e);\n                    drop(ckpt);\n", "")
elif arm == "right-size-site-off":
    old = (
        "                                        || {\n"
        '                                            vmm_reap_trim_for(engine, loaded, "spec right-size");\n'
        "                                        },\n"
    )
    sub(old, "                                        || {},\n")
elif arm == "shed-reap-off":
    sub('        vmm_reap_for("kv-flex shed");\n', "")
elif arm == "probe-prefix-reread":
    sub(
        ANCHOR,
        "\n#[allow(dead_code)]\n"
        "fn review_probe_prefix_reread(\n"
        "    engine: &Engine,\n"
        "    loaded: &HashMap<String, LoadedModel>,\n"
        "    px: &mut PrefixCache,\n"
        ") -> Option<AdmissionHeadroom> {\n"
        "    let _ = px.evict_all();\n"
        "    admission_headroom(engine, loaded, None)\n"
        "}\n" + ANCHOR,
    )
elif arm == "probe-parked-reread":
    sub(
        ANCHOR,
        "\n#[allow(dead_code)]\n"
        "fn review_probe_parked_reread<P: ParkedEntryAge, S: ParkedEntryAge, D: ParkedEntryAge>(\n"
        "    engine: &Engine,\n"
        "    loaded: &HashMap<String, LoadedModel>,\n"
        "    reuse: &mut HashMap<PoolKey, Vec<P>>,\n"
        "    spec_reuse: &mut HashMap<PoolKey, Vec<S>>,\n"
        "    dspark_reuse: &mut HashMap<PoolKey, Vec<D>>,\n"
        "    metrics: &mut ReuseMetrics,\n"
        ") -> Option<AdmissionHeadroom> {\n"
        "    let _ = evict_oldest_parked_freeing(reuse, spec_reuse, dspark_reuse, metrics);\n"
        "    admission_headroom(engine, loaded, None)\n"
        "}\n" + ANCHOR,
    )
else:
    sys.exit(f"unknown arm {arm}")
