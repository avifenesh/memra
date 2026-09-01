# Settling the merge's perf FAIL — interleaved A/B, 2026-08-31

## What happened

`tools/local-ci.sh --perf` on the merge commit `1465cce4c` recorded:

    qwen9b-plain-short: 72.28 tok/s [FAIL] — tok/s -46.76% vs median 135.77
    perf stage: 1 fail, 0 warn

The harness had already flagged the window itself, twice, before recording:

    qwen9b-plain-short: window went DIRTY mid-cell — waiting up to 600s + retrying once
    qwen9b-plain-short: co-resident did not leave in 600s — treating it as persistent;
                        rows from here are window_clean=false
    qwen9b-plain-short: DIRTY twice — recording with window_clean=false

Cause of the dirty window: a FOREIGN GPU job belonging to another session on this shared rig
(`scripts/run_local_candidates.py --model llmlingua`, ~4.2 GB resident on a 24 GB card, 45+ minutes
and still running). It is not this lane's process and was not killed — another lane's work is not
mine to terminate, so the window could not be cleaned, only measured around.

## Why a bisect was NOT the next step

`tools/local-ci.sh`'s own FAIL banner says a tok/s FAIL is a drift TRIPWIRE against a cross-day
median, not a proven regression, and prescribes the settlement: build the last-green commit's binary,
run the cell INTERLEAVED A/B/A/B, N>=5 each, in ONE thermal window under one exclusive lock hold, and
compare medians WITHIN that window only. That method is valid in a dirty window precisely because
both arms share it. The end-of-day debt protocol also scopes bisect to a FAIL in a CLEAN window.

Mechanism was ruled out independently, by reading the diff rather than trusting the number. The merge
touches exactly four engine files — `cu/qmatvec.cu` (glm5 vrows MoE kernel twins), `src/lib.rs` and
`src/hybrid_forward.rs` (the door plumbing), and a new test file — and both new flags are documented
`off` in `docs/FLAGS.md`. `qwen9b-plain-short` is a PLAIN (non-spec) decode of a Qwen3.5-9B NVFP4
model; it never enters the glm5 verify-rows MoE path, and with both doors OFF that path is not even
reachable. There is no mechanism by which this diff moves that cell.

## The A/B

`target/release/run-gen`, one binary per arm, same model / prompt-ids / `MEMRA_NGEN=128` as the cell,
five interleaved rounds, ONE `flock /tmp/memra-5090.lock` hold across the whole sequence,
`NVIDIA_TF32_OVERRIDE=0`. Arms: `merged` = `1465cce4c`, `parent` = `dfbdfd9b9` (`1465cce4c^1`).
Raw output: `merge-perf-ab.out`.

| round | merged | parent |
|---|---|---|
| 1 | 73.27 | 73.00 |
| 2 | 61.03 | 68.22 |
| 3 | 69.90 | 70.05 |
| 4 | 66.34 | 64.00 |
| 5 | 76.68 | 66.89 |
| **median** | **69.90** | **68.22** |

## Verdict

**The diff is exonerated; the window is the offender.**

- merged / parent median ratio = **1.0246** — the merge is if anything marginally FASTER, i.e. noise.
- The PARENT reads **68.22** here against its OWN clean-window row of **136.69** taken earlier the
  same evening on this same rig: **-50.1%**. A commit cannot regress a cell it predates, so a
  halving that reproduces on the parent is machine state, not code.
- Correctness on the merged tree is fully green (13 gate runs incl. the dedup suite 6/6 and all five
  walk suites on the doors-E+D+H compose arm), and an acceptance drop — the clock-independent half,
  which WOULD be real — did not occur; this is a plain cell with no acceptance term.

The `window_clean=false` FAIL row is KEPT in `research/tune-data/perf-ci.jsonl`. It is not deleted
and not amended: the journal is append-only, and a suppressed red is worse than a red with its
settlement written next to it. The row that certifies the skip-window debt is the separate
CLEAN one at `92ea07376` (136.69, `window_clean=true`) — that measurement, not this one, is what
cleared the rig-hold perf debt.

**Owed, and named rather than quietly dropped:** a clean-window `--perf` row on `1465cce4c` once the
rig is actually free. This settlement establishes that the merge did not regress the cell; it does
not substitute for a green row on the merged tree.
