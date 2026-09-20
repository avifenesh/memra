# WP-D day 10 — single-PRO profile, G2 envelope and transport smoke

Repository: **memra**, branch `lane/spill-d-20260919`. Integrated
`origin/lane/spill-integ5-20260920` into merge commit
`9f849978bac72cf2a4b608873a49341eafb7ab56` and pushed with all configured hooks.
The first broad fetch raced other local fetchers' remote-ref updates; one
branch-scoped fetch succeeded. The initial push command timed out; the bounded
retry and remote SHA readback established the push, rather than assuming it.

## Delivered commits and profile review

- `527373c3a`: reviewed the lead's `pro-single` collector/bootstrap profile and
  corrected the omitted `pro-single` enum in `runs.schema.json`.
  `/tmp/memra-gpu.lock`, GPU match `RTX PRO 6000`, >=90,000 MiB admission,
  97,887 MiB/600 W stub and canonical `flock --close -n -x` wrapper were correct.
  No tool dispatch, engine, default or numeric-program change was needed.
- Three new profile tests cover CLI plan/schema, actual canonical lock exclusion,
  contention refusal before child execution, accepted receipt validation,
  bootstrap dry-run/wrapper and red GPU-memory/power/name inputs.
- Successful bootstrap receipt: `rented-pro6000-20260920/bootstrap/BOOTSTRAP.json`.
  **31 recorded steps: 30 exit 0 and one allowed prerequisite probe exit 127**
  (`openssl-development`: `ERROR: [Errno 2] No such file or directory: 'pkg-config'`).
  Package installation followed and native builds succeeded; final status is
  `bootstrap-complete-not-tier-qualified`, not "31 steps green".
  Provider/instance/rate/cost/pidfile omitted, receipt path
  normalized, private-original SHA256 retained. Bootstrap is not tier qualification.
- `564efd3d3`: own native release build of `h2d-probe` and `pp-transport-smoke`,
  exit 0 in 3m03s; raw `rented-pro6000-20260920/build/`. Source was the merge tip
  above. `binary-identity.json` binds both executable hashes and separately checks
  runtime source/Cargo paths unchanged through the smoke checkout. Its broad
  all-crates diff contains only the new Python battery tests, not runtime changes.
- `44840aed3`: committed G2 protocol/worker before calibration or scoring.
- `7bcc27ba1`: scored G2 raw data, transfer manifest, replay summary and results.
- `ac5a8badc`: completed smoke plus capture-validation receipts.

## G2 scored campaign

See [G2-RESULTS.md](G2-RESULTS.md) for the full median table and
[G2-PROTOCOL.md](G2-PROTOCOL.md) for the pre-registered protocol.

**Requested five-size, both-direction matrix completed:** 4 KiB / 64 KiB /
1 MiB / 16 MiB / 256 MiB; pinned vs pageable; **N=10 visits per arm**, comprising
5 AB + 5 BA pairs. Two hundred scored samples, 52 calibration samples excluded.
Frozen copies/visit: 45,597 / 41,839 / 16,742 / 1,658 / 108. Every scored visit
>=497.164 ms against the 250 ms minimum. Every captured full-byte/hash comparison
and comparator-red control passed. One collector lock covered calibration and
scoring; bounded waiting did not bypass contention.

Thermal regime **T1: sequential calibration-warmed, no steady-state soak**;
1,074 telemetry samples, median 250 ms/max gap 271 ms, GPU 37–41 C, power envelope
600/600 W. Collector window 269.149 s. Raw measurement/clock/power/link ranges
and per-run hashes are adjacent to the summary; **140 transferred files hash-match**.

Measured direction: pinned improves both directions at 64 KiB through 256 MiB;
4 KiB H2D is slower pinned, while 4 KiB D2H is faster pinned. This is a
size/direction-dependent development copy envelope on one target-class card,
not a board/default decision, DMA-only bandwidth, multi-rig or serving claim.

## Single-card transport smoke

`rented-pro6000-20260920/smoke/attempt-7/command.log`, collector exit 0 after
seven lock-contention refusals; **21 transferred smoke/validation files hash-match**.
Exact key output:

```text
SMOKE HOST_BOUNCE=0
peer-arm copy (cuMemcpyPeerAsync, same-ctx degenerate): bytediff=0 OK
Pp2Rt built: cross_device=false
boundary roundtrip step 0 slot 0: bytediff=0 OK
boundary roundtrip step 1 slot 1: bytediff=0 OK
boundary roundtrip step 2 slot 0: bytediff=0 OK
boundary roundtrip step 3 slot 1: bytediff=0 OK
pp-transport-smoke PASS
SMOKE HOST_BOUNCE=1
peer-arm copy skipped: MEMRA_PP_HOST_BOUNCE=1
Pp2Rt built: cross_device=false
boundary roundtrip step 0 slot 0: bytediff=0 OK
boundary roundtrip step 1 slot 1: bytediff=0 OK
boundary roundtrip step 2 slot 0: bytediff=0 OK
boundary roundtrip step 3 slot 1: bytediff=0 OK
pp-transport-smoke PASS
```

One card has no peer pairs. `boundary_transport(false, _)` selects `Local` even
with the host-bounce flag set: this proves loopback and flag plumbing, **not a
cross-card peer or pinned host-bounce transport**. No cross-device claim is made.

## Capture validation

Local archive command actually ran:

```sh
python3 tools/tier-battery.py --validate research/spill-d-20260919/rented-pro6000-20260920
```

Verbatim summary fields (`validate-d.json`):

```json
{
  "kind": "capture-integrity",
  "cells": 2,
  "failed_commands": 0,
  "refused_commands": 0,
  "qualification": false
}
```

The requested all-BOX3 `--validate /root/spill-receipts` also ran, twice, while
other lanes were still collecting. It correctly returned exit 2, verbatim:

```text
REFUSED: interrupted/invalid CELL journal; not a completed capture
```

At 16:07:00 UTC, read-only per-cell diagnosis found **17 completed captures and
one incomplete/invalid journal**, `c-day9/pressure-spec-on-attempt1`; prior pass
was blocked by B's then-active VMM cell. Raw aggregate outputs and per-cell
`validate-all-box3-diagnosis.json` are retained. **No all-BOX3 aggregate PASS**
is claimed. Lead can rerun the same aggregate command once peer collectors close;
no D GPU work is left running and no D lock is held.

## Checks actually run

`day10/checks-final/` records all **14 command checks, exit 0**:

- `cargo fmt --all -- --check`.
- Python unittest discovery: **85 tests passed** (including seven new day-10 tests).
- `py_compile`: tier tools, every Python battery test, G2 worker/summary and verification drivers.
- Individual `bash -n`: bootstrap, both legacy KV gate wrappers, smoke runner.
- `shellcheck`: bootstrap and smoke runner.
- Bootstrap embedded Python syntax compile.
- G2 native raw/capture/telemetry replay with complete N/order/byte/hash checks.
- Local BOX3 capture validation: two completed D cells, zero failed/refused commands.
- `git diff --check`.
- `check-flags.sh`: 864 runtime literal reads, zero uncovered names.
- Eight existing runbook shell blocks parsed with `bash -n`.

Initial `day10/checks/` had one existing timeout-tree test fail with Darwin
`PermissionError: [Errno 1] Operation not permitted` at `os.killpg`; all raw output
is retained. Its isolated four-test rerun and complete suites passed (83,
then 85 tests, including the final 85-test run). The cause is unproven; **no suppression or fix is claimed**.

CPU tests and native copy/smoke evidence are separate. Full engine/server/model
exactness, multi-card, storage/NVMe and serving batteries were not run here.
No external dependency, V4.1 change, new numeric program, runtime env read,
third GPU lock, cross-box timing comparison or hook override was introduced.
Published performance boards did not change.

## Handoff and budget

Approximately **0.9 agent-hours** this session, below the 3.5-hour stop and within
the stated **9-agent-day WP-D budget**; prior sessions' cumulative hours are not
reconstructed. The active lane/worktree and own remote clone remain for lead
integration, not declared merged to main or abandoned. No unrelated work was
staged; reference checkout, artifacts and peer worktrees were not modified.
Temporary test directories self-cleaned. All raw measurements remain retained.
