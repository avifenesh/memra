# WP-D day 4 — CPU seam and rented-rig tooling milestone

Repository **avifenesh/memra**, branch **lane/spill-d-20260919**, worktree `wt-spill-d`.
Checked source **`b39b6ca6`** (full SHA in `day4-checks/checks.jsonl`). The following
receipt/docs commit does not change the checked Rust implementation or tools/tests.
**Lane committed and pushed; not merged to main, deployed, or GPU/live-qualified.**

## Commits and push outcomes

- `e7d44de6`: first action, `git merge --no-ff lane/spill-integ-20260919`, integrating
  `020d20479cd686835c0fb7743040947d0fc2723b`. Clean worktree at entry.
- `2179d814`: exported CPU directed PeerCapacity construction seam and regression.
- Initial merge and seam push attempts both refused, verbatim:
  `publishable workspace member(s) memra-tier absent from the crate list in .github/workflows/publish.yml`.
  No bypass; lead notified. No shared workflow edited by D.
- `976e762d`: independently inspected and merged lead repair `200a3c66` (lists memra-tier
  first in publish order). `git push -u origin lane/spill-d-20260919` then succeeded.
- `8d0a56d4`: container bootstrap, durable cell journals, schema/storage join and tests;
  committed and pushed successfully with hooks.
- `b39b6ca6`: recover a torn final CELL append on resume while refusing corruption of a
  completed row; committed and pushed successfully with hooks.

Successful pushes ran perf-board, 864-name flags census, publish/stub-ABI/arch censuses,
docs registry and public boundary (0 matches). The first successful push explicitly said
`engine files touched but no model dir; perf-ci not enforced here`. This is hook behavior,
**not** GPU performance evidence. No skip override, no --no-verify, no tag or main push.

## Delivered files and behavior

- `crates/memra-tier/src/peer/test_support.rs`, peer module and peer tests:
  `FakePeerCapacity<G>` consumes **the caller's shared governor**; routes default denied,
  context/pool/link faults are per direction, CPU backing matches requested bytes, explicit
  lifetime/release hooks remain. D's old private capacity delegates to this exported code;
  frozen v1.1 schedules exercise it. No duplicate governor or native CUDA claim.
  Usage: **PEER-SEAM.md**; no lead-owned Cargo feature needed.
- `tools/tier-rig-bootstrap.sh`: minimum ancestry `020d2047`; no systemd; optional locked
  power request records refusal/current cap and only requires adequate maximum. Captures
  df/lsblk before selecting persistent build/scratch paths. RunPod `/workspace`; Vast
  explicit operator-confirmed persistent root; NVMe proof via findmnt/lsblk ancestry,
  never overlay/network fallback. Two 8 GiB full readbacks remain required on real rigs.
- Bootstrap writes raw logs in the durable receipt directory immediately, fsyncs start/end
  journal rows, and atomically checkpoints BOOTSTRAP.json. Resume preserves the old receipt
  and reruns all source/hardware checks in a fresh attempt. SIGKILL regression demonstrates
  retained partial log/active step, then successful **stub** resume.
- `tools/tier-battery.py`: durable CELL start/end UTC/run-id/elapsed/optional private cost
  and provider-id metadata; raw logs flush during execution. Resume reads the last CELL,
  recovers only a torn terminal append, requires identical argv and creates a new attempt.
  No interrupted result becomes a pass. Existing dry campaign rows now append per cell too.
- `--validate <jsonl>` validates **both checked-in schemas**, then semantic byte-pair or
  telemetry checks. `--storage-samples` joins explicit run-id envelopes containing A's
  unchanged StorageSample. Orphan/missing/duplicate ids and invalid counters refuse;
  failed status, fallbacks and null physical counters remain unchanged, not scored.
- `--first-hour` emits `FIRST-HOUR-PLAN.json`: bootstrap/build → A → D1 → C → B, 60-minute
  booking, AB and BA correctness only, zero performance cells/medians. Future scoring
  still requires >=5 pairs **in each** order.
- `RIG-DAY1.md`: exact integrated A/B/C runner CLIs, standalone lock ownership, fit/time
  budgets, provider SSH port through operator identity, persistence/resume rules. Updated
  stale "runners absent"/"D does not push" statements. Missing native adapters remain
  BLOCKED; full runner rosters may exceed the first hour, so bounded existing binary cells
  remain its executable path. `CELLS.md` documents both validators and storage envelopes.

## Executed checks — all exit 0

Reproduce: `python3 research/spill-d-20260919/verify-day4.py --out <new-directory>`.
Exact argv, initial worktree state, revision, UTC, exit and SHA256 are retained in
`day4-checks/checks.jsonl`; **all 19 complete merged-output logs** are gzip sidecars.
Independently decompressed/rehashed all 19: matching hashes. Worktree differences during
checks were D documentation/receipt files only, not changed Rust/tools/test code.

| Check actually run | Exact result / output summary |
|---|---|
| `cargo fmt --all -- --check` | exit 0, empty output |
| macOS `cargo check -p memra-tier -p memra-kv --offline --all-targets` | exit 0; `Finished dev profile ... in 1.65s` |
| Linux-target same check + `--target x86_64-unknown-linux-gnu` | exit 0; `Finished dev profile ... in 0.70s` |
| Linux-target memra-tier-only check | exit 0; `Finished dev profile ... in 1.43s` |
| `cargo test -p memra-tier --offline --no-fail-fast` | bank 35, contracts 44, peer 18, placement 6, storage 37, doctests 4: **144 passed, 0 failed, 0 ignored** |
| `cargo test -p memra-kv --offline --no-fail-fast` | **51 passed, 0 failed, 0 ignored** |
| `cargo clippy -p memra-tier --offline --all-targets -- -D warnings` | exit 0 |
| Python unittest discovery, `tests/battery/test_*.py` | `Ran 36 tests in 13.067s` / `OK` |
| `python3 -m py_compile tools/tier-{battery,placement,topology}.py` (expanded argv) | exit 0, empty output |
| `bash -n tools/tier-rig-bootstrap.sh` | exit 0, empty output |
| `shellcheck tools/tier-rig-bootstrap.sh` | exit 0, empty output; actually available/run |
| runs validator | `BYTE-RECEIPTS MATCH: 22 records / 11 forced pairs; not full battery qualification` |
| telemetry validator | `TELEMETRY MATCH: 3 samples; NOT hardware qualification` |
| first-hour plan reproducibility | `FIRST-HOUR MATCH: AB/BA correctness only; no perf medians` |
| retained CPU campaign | `CPU CAMPAIGN MATCH: 22 runs; synthetic protocol evidence only, NOT GPU qualification` |
| frozen fixture reference | `tier-contract-v1: 8 payload + 3 canonical wire SHA256 pins match` |
| cargo metadata target/dependency assertions | `METADATA MATCH: engine tier dependency and all first-hour binary targets registered; NOT native compile` |
| `git diff --check` | exit 0, empty output |
| `bash tools/check-flags.sh` | `runtime literal reads=864`; `no uncovered runtime names` |

The inherited macOS memra-gguf `source.rs:20` AsRawFd unused-import warning remains;
it was not hidden or changed. Cross-target checks are not Linux execution or CUDA builds.
`git diff --exit-code 020d2047 -- crates/memra-tier/src/contracts.rs crates/memra-tier/tests/contracts/`
was empty/exit 0: lead-owned contracts/schedules unchanged.

### Bootstrap dry-run evidence

`day4-bootstrap-final/`: **44 external commands stubbed**, all exit 0; all raw log hashes
independently match. `SOURCE.json` binds actual tool hashes/source. BOOTSTRAP's all-ones
remote SHA is deliberately fake. RunPod paths/NVMe output and acceptance output are stubbed;
no provider connection, install, sleep gap, GPU allocation or native build happened.
No real provider id/cost was captured. Earlier `day4-bootstrap-dry/` is retained and marked
superseded, not presented as final-source evidence.

Tests cover power-request refusal with adequate max/current restriction; low/invalid max;
Vast missing persistence; overlay refusal; resume after failure; actual SIGKILL of a stub
bootstrap while a log is open; duplicate output protection; CELL start-only/torn-tail resume;
corrupt journal refusal; both schema CLIs; integer-type teeth; storage joins and AB/BA plan.

## Numbered blockers / remaining scope

1. **No rented GPU/nvcc here.** Provisioning remains blocked on the operator provider session.
   No SSH/rental/model action was attempted by this worker. Lead reports PR #518 CI running
   a first native compile; D has not inspected its result and claims no native compile pass.
2. **Actual runtime/consumer qualification:** A GPU owner, B active/prefix binding, C native
   bank/row consumers and D real directed peer grants/fences still need integrated hardware
   gates. Linux O_DIRECT execution, model/serving correctness and telemetry remain unrun.
3. **Hardware ladder:** single 5090 smoke is not pair/four-tier or 4×PRO evidence. Official
   Step PP, real Qwen 262144, full Hy3/PLE, all 12 peer routes and shared-fabric pressure remain
   required. No NVLink/TP assertion, paused 0731 oracle, V4.1 code or format substitution.
4. **Interruption boundaries:** durable volume survival is provider/offer-specific and needs
   operator confirmation plus private off-box receipts. A/B/C native runners remain separate,
   with their existing journal formats/locks; D does not falsely grant them `--resume` or
   per-inner-command cost instrumentation. After interruption inspect their last receipts and
   restart idempotently in fresh output/scratch state, as documented. Real storage joining
   needs explicit A raw-cell → D run-id mapping; no guessed positional join.

G0–G7 remain pending. No production default/performance decision or release qualification.
Shared docs amendments: DAY4-SHARED-FRAGMENTS.md (no new flags/kernels/default-off doors).

## Hygiene and effort

Approximately **0.3 agent-hours** active day-4 work; no GPU/rental time charged. Prior days'
actual hours were not supplied, so cumulative consumption is unknown. WP-D implementation
budget remains **9 agent-days**, separate from the lead's 4-day allowance; no invented
hours-to-owner-scaled-days conversion. Worktree remains open for subsequent WP-D milestones,
not an abandoned/closed lane. Only D-owned files edited; integration's shared changes arrived
through explicit lead merges. No stashes; test temporary directories self-clean; original
canonical lock inodes retained. Final branch/remote SHA and worktree status are reported in
the handoff after this receipt commit is pushed.
