# WP-D day 6 — collector, inherited lock, topology and recovery

Repository **avifenesh/memra**, branch **lane/spill-d-20260919**. Final source
checked: **`c79c3b1de55e8f3d2d7fd05af961e68b499d73c1`**. This final receipt adds
only the reusable check runner, logs and this report. Changes are committed and
pushed to the lane, **not merged, deployed or newly model/GPU-qualified**.

## Milestones (each committed and pushed with hooks)

| Commit | Result |
| --- | --- |
| `41507fb9` | Preserved literal child `--`; terminal explicit-refusal classification; per-cell power limit/max with CSV integrity; A2 spelling. Retained/finished the inherited dirty files after inspecting `ac6baf93..HEAD`. |
| `645602c8` | Second actual topology fixture, read-only power/sysfs/pidfile observations, three archived second-rental CELL integrity validations. |
| `cdc519fd` | Collector inherited-FD lock proof and process-group teardown; exact legacy-script fragment, CPU/Linux tests and capture schema documentation. |
| `2efc46c2` | Lead-reported two host stops, receipt-led ≤30 min replacement policy, per-cell sync and restricted-power runbook. |
| `c79c3b1d` | A's newly available storage-cell diagnostic join, with current capture validation first; no automatic qualification. |

At the initial A fetch (`fd49a668`) no fragment existed. A final re-fetch found
`0e7b4456b33935917136455ffaa9214d95db6329`; its supplied `storage_capture.py` was
imported byte-for-byte and verified against git. The earlier carry-forward
record is superseded by `day6/STORAGE-CELL.md`.

## Checks that actually ran

Reproduce with `python3 -B research/spill-d-20260919/verify-day6.py --out <new-dir>`.
The final integrated pass is `day6/final-checks-storage/`: **23/23 commands exit
0**, with exact commands, source revision, source manifest and compressed raw
SHA-256s. All raw hashes and source hashes were independently rechecked afterward.
The pre-storage pass is retained at `day6/final-checks/`, not overwritten.

| Criterion | Execution and result |
| --- | --- |
| Rust formatting | `cargo fmt --all -- --check`: pass |
| macOS and Linux target compilation | `cargo check -p memra-tier -p memra-kv --offline --all-targets`, native and `x86_64-unknown-linux-gnu`: pass; cross-check is not native Linux/GPU execution |
| Tier suite | `cargo test -p memra-tier --offline --no-fail-fast`: 167 tests + 4 compile-fail doctests passed |
| KV suite | `cargo test -p memra-kv --offline --no-fail-fast`: 59 tests passed |
| Clippy | memra-tier all-targets with `-D warnings`: pass |
| Python | 61 tests passed; all tier Python tools plus A's helper `py_compile`: pass |
| Shell | bootstrap `bash -n`, embedded-Python compile and shellcheck: pass; legacy fragment applied in disposable tests and checked with `bash -n` / shellcheck source following |
| Fragment applicability | `git apply --check` on C goldens and legacy inherited-lock fragments: pass |
| Capture integrity | 9 first-rental + 3 second-rental archived CELLs passed integrity; the original failed C cell remains failed |
| Other preserved contracts | Byte/telemetry/dry-campaign validation, pinned wire fixtures, first-hour plan equality and target metadata: pass |
| Hygiene | `git diff --check`, 864-name flags census: pass; no new environment reads/dependencies/kernels/runtime numerical paths |

An existing unrelated macOS warning remains in `memra-gguf/src/source.rs:20`
(`unused import: std::os::fd::AsRawFd`); it was not edited or silently suppressed.
A prior Python attempt failed the existing 100 ms startup assumption (empty raw
log). Its timeout allowance was increased to 1 s, keeping the 10 s child sleep,
timeout assertion and raw-output assertion. Both failed and green logs are kept.

## Actual remote scope

- Read-only probe: one successful attempt. One device, idle Gen1, host/device
  ceilings Gen5 x16; **not measured bandwidth or P2P qualification**.
- Actual `power.limit=400.00 W`, `power.max_limit=600.00 W`; no power setting was
  changed. Per-cell field names live in `CAPTURE-CONTRACT.md`.
- sysfs exposed NVMe/md0 names but their /dev nodes were absent. NVMe ancestry,
  O_DIRECT performance and spill-speed claims remain blocked.
- Default bootstrap pidfile was absent: `--status` exit 1, `not-running`; no
  matching bootstrap pidfile under receipts. This does **not** prove process-wide
  bootstrap absence. The operator must supply its actual pidfile if different.
- Five inherited-lock CPU tests ran successfully on Linux after one SSH timeout
  and a bounded retry. Temporary source tree cleaned on exit; no model binary or
  GPU compute cell ran. Tests check inherited owner vs same-inode nonowner,
  missing/foreign/closed/unlocked proof, concurrency, parent/child FD lifetimes,
  failure/timeout teardown and survival of an unrelated process.

## Fragments / remaining lead work

1. Apply/review `LEGACY-EXTERNAL-LOCK.diff` alongside `tools/tier-lock-proof.py`.
   D did **not** edit either legacy gate. Run complete native identity/failure
   model gates before treating that shell interface as serving-qualified.
2. Existing `C-RUNNER-GOLDENS.diff` remains a C-owner handoff, not newly applied.
3. FLAGS.md fragment: **none**. Explicit FD arguments introduce no environment
   read or experimental numerical/default door, so no decide-by row is needed.
4. Locate an actual active bootstrap pidfile if one exists elsewhere. No rig
   configuration changes, resets, restarts, artifact writes or other-lane worktree
   edits were made.
5. Multi-card transport, model-scale mixed tiering and full serving/performance
   gates remain outside this CPU/read-only milestone; do not promote their state.

Elapsed effort this relaunch: approximately **0.6 agent-hour**, under the 2-hour
stop budget. The campaign labels this day **6 of 9**; prior-session cumulative
agent-hours were not reconstructed and no invented remaining-budget percentage
is reported. The lane stays open for lead integration; its worktree is not an
abandoned scratch tree. No unrelated worktree changes were absorbed.
