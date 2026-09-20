# WP-D day 9 — nested timeout containment and calibrated G2 rehearsal

Repository: **avifenesh/memra**, branch `lane/spill-d-20260919`.
Development plumbing only. Nothing here is main-merged, released, deployed,
scored, or serving-qualified. This lane remains open for lead integration.

## Source and delivery milestones

- `ea5a4de552c0bbe309d1785049faa99f6f36dd0b`: merged the requested
  `0214c4b8` integration, then pushed with hooks enabled.
- `241c779e`: nested process-group containment plus inherited-lock regression.
- `145efc6e`: F's exact final `7644c41f` probe source and calibrated runner.
  Native binary built from this revision with CUDA 13.1 / sm_120a,
  `cargo build --release -j 16 -p memra-engine --bin h2d-probe`; exit 0.
- `d2a7be53`: current A/B/C/F collector runbook and pinned peer-archive replay.
- `f9285d95`: conservative calibration-count headroom and CPU verification driver.
  This is the actual successful rehearsal worker source. Probe bytes are unchanged
  from the native build and F's final source; the offline verifier checks all three.

Binary SHA-256:
`a424b675b2e752828029fb908753d23adc5e1d61b0002d6ecc964f686223942c`.
Native source, binary, collector and worker hashes are retained, not inferred from
branch names. F's isolated auto-discovered target was `h2d_probe`; this integrated
manifest explicitly registers `h2d-probe` using the same source file.

## Timeout finding: reproduced, fixed, replayed

`day9/containment/red.log` reproduces the original behavior: the outer collector
reports `(-9, True)`, yet its grandchild writes the survival marker. For this red
run only the new keyword was admitted to the old function signature and ignored;
original session/kill behavior was unchanged. The intentional detached-child red
control independently observes the surviving process retaining the inherited flock.

`tee_run(..., shared_group=True)` now keeps nested probes in the worker's group.
`tier-envelope.py::visit` explicitly supplies that option and propagates the
inherited canonical lock FD. An inner timeout terminates the whole worker group;
the outer collector reaps it and writes the failed capture. Outer cleanup also
terminates descendants with closed stdout. A successful nested visit leaves its
worker alive for the next visit. This is cooperative process-group containment:
commands must not daemonize or call `setsid`; it is not a hostile-process sandbox.

Four process tests check outer timeout, inner timeout, successful nested return,
and the intentionally detached negative control. Positive arms verify the grandchild
is absent/dead (zombie permitted pending OS reaping), its late marker is absent,
and an independent open description can acquire the flock. They use an unlinked
CPU fixture inode, not a third rig lock. A separate runner test verifies the
actual `visit` call supplies both group and FD arguments.

- macOS: `day9/containment/green.log`, then complete 78-test final Python suite.
- Linux: `day9/native/build/linux-containment.log`, all four process tests pass.
- The detached red control is killed by test cleanup; no sleeping probe is left.

## One actual G2 rehearsal matrix

`day9/native/rehearsal/`: four collector cells, **16 measured visits** total.
Each size/direction has N=1 per arm/order: AB then BA, pageable versus cacheable
pinned (CUDA flags zero). Discarded calibration selects one fixed copy count.
Inner copy count does not increase N. Complete-byte identity passes for every
visit; each raw invocation contains the distinct-pattern controls and comparator
red rejection. Bare-JSON probe rows are preserved; only the worker emits `RESULT `.

| Bytes | Direction | Copies/visit | Minimum measured visit | Visits |
| ---: | --- | ---: | ---: | ---: |
| 4,096 | H2D | 45,909 | 510.757477 ms | 4 |
| 4,096 | D2H | 36,940 | 383.715461 ms | 4 |
| 16,777,216 | H2D | 1,405 | 434.483288 ms | 4 |
| 16,777,216 | D2H | 1,405 | 434.053902 ms | 4 |

These are duration-floor observations, **not performance medians/comparisons**.
Every measured visit exceeds 250 ms. All four collectors exit 0 and retain
`executed-not-qualified`, `qualification=false`, `scoring_eligible=false`, and
`medians_published=false`. Raw 250 ms telemetry brackets all visit windows;
maximum observed inter-sample gap is below 273 ms. Configured power is 400 W,
maximum 600 W: restricted-power, allocation-warm rented RTX 5090 development,
not steady-state/full-power evidence. Before/after compute-app snapshots are
empty; before/after CPU build-name snapshots are empty (not continuous surveillance).
All GPU launches run under the collector's canonical `/tmp/memra-5090.lock`.

### Retained pre-rehearsal failure

`day9/native/g2-4096-h2d/` is an earlier **discarded-calibration-only** attempt.
It ran five AB calibration pairs, no measured rehearsal pair, then exited 2:

> tier-envelope: REFUSED: calibration did not converge; no rehearsal/scored samples run

The collector correctly calls this **failed**, not an explicit expected refusal.
Exact-ratio count updates approached the 350 ms target from below as per-copy
launch overhead shrank (last observed fast-arm duration 340.339621 ms). Count
updates now include 25% headroom. A fixed-launch-overhead regression test covers
that convergence issue. The one actual rehearsal matrix was then run once;
no completed sample set was discarded or selected for faster measurements.

## Replay and runbook

`verify-day9-native.py` checks all exported hashes, five collector journals,
source/binary identities, raw-sample bindings, full measured order/count matrix,
complete-byte hashes, visit-duration floors and UTC telemetry coverage. It also
checks the failed attempt contains calibration only. Result:
`day9/rehearsal-validation.json`: 16 visits, four executed-not-qualified cells,
one retained failed calibration cell. Optional deployment/cost fields in exported
CELL rows are null; original/export hashes are in `native/export-manifest.json`.
Original receipts remain in the lane-owned remote receipt directory.

The actual collector `--validate` accepts all pinned A `day7/`, B
`day8-active-8192/`, C `day8/`, and F `h2d-copies/` archives. No peer raw bytes were
rewritten. `validate-day8-archives.py` extracts immutable commits, validates them,
and cleans its disposable extraction. Verbatim summary:

```text
CAPTURE ARCHIVES MATCH: 14 cells; 12 executed-not-qualified; 1 failed command; 1 refused command; qualification=false
```

Full `--validate` JSON and immutable archive identities are retained under
`day9/peer-validation/`. This is archive integrity, not G1/G2 acceptance.
`DAY8-CELLS.md` now gives the exact host-only B active spelling, conditioned 32k
admission, all eight 8/4 GiB C expert ON/OFF gen/spec cells, the precise host-record
refusal wrapper/diagnostic, and the calibrated G2 commands. B's day-eight verdict
remains copy/restore bit-identical with no observed reclaim; it is not G1 PASS.

## Checks and limits

`day9/checks-final/`: all ten scoped command checks pass:

- `cargo fmt --all -- --check`.
- Python unittest discovery: **78 passed**.
- `py_compile` of tier tools, battery tests and replay/verification drivers;
  native-replay driver has a separate successful compile log.
- Individual `bash -n` checks of bootstrap and both legacy gate scripts.
- `shellcheck tools/tier-rig-bootstrap.sh`.
- Eight runbook shell blocks parsed with `bash -n`.
- `git diff --check`.
- `check-flags.sh`: 864 runtime literal reads, zero uncovered names.
- Additional standalone probe Rust tests: **3 passed** (`day9/probe-tests.log`).

The first checks run's runbook-syntax command had an escaped-newline bug; its
SyntaxError remains in `day9/checks/`. Corrected driver and complete green rerun
are separate. No test failure was silently overwritten.

Remaining gates: scored N>=5 same-window G2, full 20-cell size/direction envelope,
PRO hardware qualification, full engine/server battery, serving and storage/NVMe
qualification were **not run**. They are not inferred from this plumbing rehearsal.
No new dependency, numeric program, runtime default, MEMRA env read, or third rig
lock was introduced. Main, other lane worktrees, shared native checkout and model
artifacts were untouched. All pushes used the configured hooks, no bypass.

This relaunch used approximately **0.4 agent-hours**, below the 2.5-hour stop and
within the stated **9-agent-day** WP-D budget. Earlier days' cumulative agent time
is not reconstructed. Disposable local archives/test executables were removed;
raw native evidence is retained intentionally, and the active lane is handed back
for lead integration rather than declared merged or abandoned.
