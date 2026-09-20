# WP-D day 5 — first-rental bootstrap and collector repairs

Repository **avifenesh/memra**, branch **lane/spill-d-20260919**. Checked source:
**`83e6167bb5863b9727c9706499a5bcc9416afc30`**. This receipt commit adds documentation
and retained logs only. Lane work is committed/pushed, **not merged to main, deployed,
or newly GPU-qualified**. No runtime Rust, numerical program, V4.1 code, kernel, flag
read, shared contract, or A/B/C implementation changed in D's repair commits.

## Commits

- `9459d148018e8f1315fa2a64896f8595906c4fdd`: first-action merge of
  `origin/lane/spill-integ2-20260919` at `9612aa34e5dedf72a502abe20436248590d9fc47`;
  includes lead bootstrap fix `01e7b77f` and first-hour raw receipts. Pushed with hooks.
- `ee2fb624`: bootstrap explicit unproven-storage mode, pidfile status, job cap and
  operator notes; corrected the stale negative arch-list test.
- `21849259`: capture UTC/seconds and integrity validator, topology fixture/parser,
  first-hour regression tests, runbook repairs and C-owner goldens patch handoff.
- `5e4f8b48`: lead-reported spot-loss recovery: sync **after every cell**, immutable
  artifact re-downloads, pushed-worktree recovery, receipt-led bounded wait/replacement.
- `83e6167b`: CLI-level storage refusal/opt-in test and reproducible day-5 check runner.

Each milestone above was committed and pushed with `core.hooksPath=tools/hooks`.
Every push passed perf-board, 864-name flags census, publish/stub-ABI/arch censuses,
docs registry and public-boundary checks. No bypass, tag, main push or skip override.

## Numbered defect disposition

| Input defect | Repair and regression | Evidence / limit |
|---|---|---|
| 1. `compute_120a` absent from arch listing | Kept lead's `compute_120` predicate plus mandatory `-arch=sm_120a` compile probe. Fixed the existing stale mutation test. New test proves base-only listing passes and compile rejection stops before GPU acceptance. | Tests use stubs; original real successful compile and both earlier blocked receipts remain immutable. |
| 2. Overlay had no sanctioned continuation | Bootstrap `--nvme-root <actual-path> --allow-unproven-storage`; collector `--storage-root <actual-path> --allow-unproven-storage`. Default refuses unproven explicit storage. Both retain findmnt/lsblk failures verbatim; NVMe remains unproven/null; every opted-in CELL and capture carries **"overlay/unproven — not NVMe, not spill speed"**. | Stub bootstrap, collector and CLI red/green tests; missing tools and failed ancestry retained. `storage-bench` CLI requires an explicit storage root. This is development exactness only, not a waived NVMe/perf gate. |
| 3. `pgrep -f` self-match | `--status` / `--already-running --pidfile PATH`: 0 active, 1 inactive/missing, 2 error. Advisory pidfile lock is held for bootstrap lifetime, not inferred from process-name text or recycled PID. Duplicate invocation refuses before output creation. | Tests cover SSH-shaped self-match text, missing file, stale/live-but-unlocked PID, held lock, duplicate invocation and release. Existing interrupted-bootstrap test remains green. Pidfile is process metadata, not another GPU campaign lock. |
| 4. Missing C byte goldens | Runbook copies the checked-in pre-streaming TSV beside the output receipt and writes SHA-256 before executing the gate. Collector output has its own new child directory. C runner reference now requires the same precondition. | Runbook regression and `git apply --check C-RUNNER-GOLDENS.diff` pass. **C owner must apply the handoff patch**; D did not edit C's runner. Never mint substitute goldens with the current loader. |
| 5. Key injection/proxy | Bootstrap comments and runbook document operator-only `vastai attach ssh <id> "<public-key with a NEW comment>"`; explain unchanged association and prefer proxy over direct-IP. No addresses, ids, locations or credentials copied. | Documentation/operator action; no key attachment or provider state change performed by D. |
| 6. Bootstrap receipt/check mismatch | Actual archived successful receipt satisfies status, acceptance, false qualification, source and all six binary-pin fields. Runbook also requires the exact six-name roster (empty metadata cannot pass). | Test compares archived `source.commit` and `binaries.sha256`; independent audit rehashes all 40 raw steps across 3 bootstraps plus the retained acceptance executable. **Fresh current six-binary re-read did not run**: 3 bounded SSH attempts refused; lead subsequently reported preemption. No schema mismatch found; no live pass invented. |
| 7. Missing readable capture duration / validator | Capture JSON and CELL end now share UTC start/end and elapsed seconds; preserve monotonic ns. Capture-integrity mode accepts CELL, capture JSON, or directories, validates raw hashes/status/lock metadata and empty diagnostics, and reports failed commands without qualifying them. Strict byte/telemetry schemas remain distinct; explicit wrong schema refuses. | All **9 real first-hour CELL bundles** pass integrity: **8 executed-not-qualified, 1 failed** (original missing-goldens C). Legacy seconds derived from recorded ns; original files unmodified. Tests reject tampered durations/hashes, path escapes, incomplete journals, mismatched schemas and unlabeled unproven storage. |

Failure quoting additionally recognizes Rust `Error:` case-insensitively; no failure cause
is inferred. `--jobs` is now capped at 16, matching the current box access contract.

## Topology and runbook

`topology.rented-5090.fixture.json` retains real topology/NUMA output and only the
GPU link-info block from the original receipt, with the GPU address redacted. Companion
fixture notes pin the source receipt hash. `tools/tier-topology.py` reports separate
current/effective-max/device-max/host-max generation and width fields; unknown data stays
unknown. Fixture: **current Gen1 x16**, effective/host ceiling **Gen4 x16**, device ceiling
Gen5. A downshifted inventory sample is not an active downgrade test or bandwidth proof.

`RIG-DAY1.md` now distinguishes observed 2026-09-19 **one-run** durations from unchanged
booking budgets: bootstrap 518.976 s; four build steps 9.939/294.238/49.519/23.664 s;
six A collector durations, D1 0.645826 s, failed C 0.377596 s and corrected C 6.877514 s.
These are collector/command observations, not GPU device timings, steady-state medians,
NVMe speed, serving performance, or proof of B execution. The historical build used -j32;
the future cap is 16. Source remains the original `01e7b77f` binary, not this lane tip.

The lead's spot-loss report is explicitly attributed rather than presented as independently
queried provider state. Saved A/D1/C receipts survived; the partial B download and scratch
worktree did not. Receipt sync is mandatory after **every** completed or failed cell;
artifacts must be recoverable from immutable locators plus complete hashes. A start refusal
`Required resources are currently unavailable` is not a recovery plan. Decisions use which
receipts/unique bytes are already off-box and which pending work is reproducible; no
automated destroy/start loop was added.

## Executed verification

Reproduce: `python3 research/spill-d-20260919/verify-day5.py --out <new-directory>`.
`day5-final-checks/checks.jsonl` records exact commands, checked revision, UTC, elapsed,
exit status and raw-log SHA-256. All **21 commands exited 0**. All 21 gzip logs were
independently decompressed/rehashed, and every file in `source-manifest.json` matched
current source after the run.

| Check actually run | Result |
|---|---|
| `cargo fmt --all -- --check` | pass |
| macOS `cargo check -p memra-tier -p memra-kv --offline --all-targets` | pass |
| Same check, `--target x86_64-unknown-linux-gnu` | pass; cross-target compilation, not Linux execution |
| `cargo test -p memra-tier --offline --no-fail-fast` | **164 passed**, 0 failed/ignored, including 4 doctests |
| `cargo test -p memra-kv --offline --no-fail-fast` | **59 passed**, 0 failed/ignored |
| tier clippy, all targets, `-D warnings` | pass |
| Python battery discovery | **48 tests passed** |
| `py_compile` for three tier Python tools | pass |
| Bootstrap Python heredoc compilation, `bash -n`, shellcheck | all pass; shellcheck actually installed/run |
| C owner patch applicability | pass; unapplied handoff only |
| Real first-hour `--validate` | **9/9 archive integrity**, one failed command retained |
| Strict byte rows / telemetry / synthetic campaign | 22 rows, 11 pairs /3 samples /22 synthetic runs pass; no GPU qualification |
| Frozen fixtures, first-hour plan reproduction, Cargo target census | pass |
| `git diff --check`, `check-flags.sh` | pass; 864 covered runtime names |

The inherited macOS `memra-gguf/src/source.rs:20` unused `AsRawFd` warning remains visible
and untouched. Initial `day5-checks-attempt1` has one verification-driver quoting failure
in the *extra heredoc syntax command* (not bootstrap code); corrected driver then passed
in `day5-checks`, followed by the final exact-source run above. Earlier raw outputs remain
retained, not overwritten.

`day5-development/archive-audit.json` records the 3 immutable bootstrap receipt hashes and
40 successful raw-step rehashes. `bootstrap-admission*.json` records the 3 failed, read-only
proxy attempts (well over 30 s apart); final stderr is retained with host/port redacted.
No GPU cell, artifact operation, checkout mutation or other remote state change occurred.

## Remaining blockers / parent actions

1. Apply/review **C-RUNNER-GOLDENS.diff** in the C-owned lane and run C's own runner tests.
2. On accessible replacement hardware, run the strict source/current-binary admission check
   on its own fresh successful bootstrap. The first rig's archived pins are consistent;
   its inaccessible/lost current binaries cannot now be freshly re-read. Do not reuse its
   device acceptance for the replacement.
3. Real NVMe ancestry/O_DIRECT/SSD speed, B active/prefix restore, native bank/row consumers,
   D directed peer/context/pool/fence gates, PRO pair/four-card pressure and serving
   qualification remain pending. No G0–G7 promotion. No home-rig, paused 0731 or V4.1 work.
4. The inherited lead Results prose includes rental locations; flagged to lead for public
   boundary cleanup. D left lead-owned source/receipts unchanged.

## Hygiene and effort

Approximately **0.4 agent-hours** this invocation (entry ~13:33 UTC; receipts ~13:53 UTC),
no new GPU time. Prior cumulative actual effort is unknown; the WP-D implementation budget
remains **9 agent-days**, separate from the lead allowance. No invented conversion from
wall hours to the owner's agent-day scale, no remaining-budget claim without prior totals.

The lane stays open for subsequent milestones. No unrelated dirty work was present at entry;
only D-owned tooling/tests/runbook/evidence were edited after the explicit integration merge.
No stash, new dependency, provider credential, machine id, third GPU lock or runtime flag
was introduced. Temporary test directories self-clean; canonical lock inodes are retained.
Final receipt commit/remote SHA is reported after the push rather than guessed here.
