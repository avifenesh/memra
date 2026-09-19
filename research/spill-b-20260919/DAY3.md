# WP-B day 3 — CPU admission milestone; runtime patch unapplied

Repository **avifenesh/memra**, worktree `/Users/avifen/tiyuvta/wt-spill-b`, branch
`lane/spill-b-20260919`. Exact checked code/runner tip:
**`86d84de6c9552e3d854910d50e011aa9e8667655`**. Subsequent receipt-only commit
adds this report and raw output, not implementation. No push, PR, tag, main merge,
deployment, model execution, GPU qualification or support promotion.

## Re-entry and commits

- Clean lane at `bb431f339a1062e3d9a377a6990f16c76ce3461a`.
- **`fb9375f24c066c006d8a2cdcb1f48fb8ecb2b7a2`**: requested `git merge --no-ff
  lane/spill-integ-20260919`, integration tip
  `98e558dc03a0d999cf38db7aa8f27c1a1a4b12fb`. **No conflicts**; ort merged all
  day-2 lanes. No frozen semantic resolution or contract amendment was needed.
- **`fa763c97`**: bounded TierAdmissionPlan scheduler, direct local/peer descriptor
  path, owned immutable image sealing, synchronous shared-governor residency guard,
  CPU tests and 48-row policy fixture.
- **`96fff7e7`**: reviewed unapplied HostPrefixCache patch, hunk review, policy source
  notes, fitting-rig runner and its green/red stub tests.
- **`86d84de6`**: cross-tenant scheduler fairness test, safe telemetry cleanup,
  artifact/source-bound fitting-envelope preflight and corresponding cell notes.

Frozen `contracts.rs` and `tests/contracts/` compare unchanged against `98e558dc`.
Runtime `worker.rs`, `admit_memory.rs`, `worker/host_glm.rs` also compare unchanged.
No root Cargo/lock/shared export, D peer implementation, A storage, C bank or shared
docs were edited. No new MEMRA_* read, kernel, hardware default or published number.

## Deliverables and acceptance status

| Day-3 requirement | Actual result |
|---|---|
| Scheduler state machine | **CPU PASS.** `tiered/scheduler.rs` owns frozen TierAdmissionPlan, performs advisory lookup → one reservation → nonblocking prefetch/load → fenced ready callback, bounded uncharged queue, frozen priority/deadline/tenant/FIFO order. Tests cover concurrent interleaving, repeated quota-blocked ticks, cancel during pending prefetch, late success, live-epoch rollback, timeout, no publication without consumer fence, cross-tenant turn-taking and exact charge retirement. Not wired to server. |
| Immutable sealing/COW | **CPU PASS.** `SealedImage::copy_committed` checks live ActiveEpoch and committed high water, verifies exact bytes, physically clones payloads before epoch-0 sealing. Tests reject uncommitted high-water, mutate the source after sealing, rollback the active generation, and verify the sealed bytes remain unchanged. |
| Direct local/peer transitions | **CPU B-path PASS; D interoperability PENDING.** One direct descriptor ticket takes Reserved→Loading→Ready, never fabricates HostReady, retains source provenance and demands consumer fences. B fake CPU owners exercise both devices. **D's PeerCapacity fake is private** in `tests/peer/fake.rs`; B does not instantiate it. D's ten peer tests run separately in the integrated suite. |
| HostPrefixCache runtime slice | **REVIEWED, UNAPPLIED, UNBUILT.** `HOSTPREFIX-PATCH.diff` plus `PATCH-REVIEW.md`: identity binding to copied native plain image, leases, source/destination quota guards, retained host twin, off-path preservation, handoff refusal, and constructor coverage. It is not enabled and not merge-ready runtime code. |
| Recompute/load fixture | **CPU PASS.** 48 token × bandwidth × prefill-rate rows; all parsed/tested. `RECOMPUTE-FIXTURE.md` cites local E §2.7 and py-kvcache [P1] break-even law and distinguishes reference example sizes from Qwen/native geometry. No measured calibration or default. |
| Rig runner | **CPU dry-run/teeth PASS; LIVE UNRUN.** Scratch worktree/branch, actual before/after commands, 8k/32k fitting baseline probes first, existing prefix identity/teeth/failure cells, raw-first logs/JSONL/hash records, 250ms GPU telemetry setup, canonical lock, native-active gate fail-closed. Two Python tests; 20 stub commands; deliberately injected exit 7 refused without hiding stderr. Not a completed generic GPU gate. |

### Changed paths

- `crates/memra-kv/src/tiered/{scheduler,mod,integration,hostprefix,tests}.rs`.
- `research/spill-b-20260919/{HOSTPREFIX-PATCH.diff,PATCH-REVIEW.md,RECOMPUTE-FIXTURE.md,CELLS.md}`.
- `research/spill-b-20260919/fixtures/recompute-load.csv`.
- `research/spill-b-20260919/{rig-cells-b.sh,rig-cells-b.py,rig-stub-b.sh,test-rig-cells-b.py,verify-day3.py}`.
- This report, `day3-checks/`, development logs and explicit CPU-stub raw receipts.

## Exact executed verification

Reproducer: `python3 research/spill-b-20260919/verify-day3.py`.
Every command ran on the Mac at the exact code tip above. Complete stdout/stderr
and command/exit/source/hash records are in **`day3-checks/commands.json`** and
its named `.log` files; raw output was closed before parsing or summarizing.

| Command | Actual result / exact relevant output |
|---|---|
| `cargo fmt --all -- --check` | exit 0; no output |
| `cargo check -p memra-kv -p memra-tier --offline --all-targets` | exit 0; `Finished dev profile [unoptimized + debuginfo] target(s) in 0.68s` |
| same check plus `--target x86_64-unknown-linux-gnu` | exit 0; `Finished dev profile [unoptimized + debuginfo] target(s) in 0.75s` — Linux type/cfg check only, NOT linked/executed Linux tests |
| `cargo test -p memra-kv -p memra-tier --offline` | exit 0; memra-kv **47 passed**; bank **27**; contracts **36**; peer **10**; placement **6**; storage **26**; memra-tier doctests **4**. Every nonempty suite: **0 failed, 0 ignored, 0 filtered out**. 156 total including doctests. Empty library/doc harnesses are not counted as coverage. |
| `cargo clippy -p memra-kv -p memra-tier --offline --all-targets --no-deps -- -D warnings` | exit 0; no changed-crate warning |
| `git diff --check` | exit 0; no output |
| `bash tools/check-flags.sh` | exit 0; `runtime literal reads=864`; `no uncovered runtime names` |
| `git apply --check research/spill-b-20260919/HOSTPREFIX-PATCH.diff` | exit 0; no output; verifies application shape, NOT Rust/server compilation |
| `bash -n research/spill-b-20260919/rig-cells-b.sh research/spill-b-20260919/rig-stub-b.sh` | exit 0; no output |
| `python3 research/spill-b-20260919/test-rig-cells-b.py` | exit 0; `Ran 2 tests`, `OK`; green plan and intentional child exit-7 error both exercised |
| `git diff --exit-code 98e558dc -- crates/memra-tier/src/contracts.rs crates/memra-tier/tests/contracts` | exit 0; unchanged frozen surfaces |
| `git diff --exit-code 98e558dc -- crates/memra-server/src/worker.rs crates/memra-server/src/admit_memory.rs crates/memra-server/src/worker/host_glm.rs` | exit 0; patch not applied |

Preserved existing Darwin dependency warning (not repaired out of scope):
`crates/memra-gguf/src/source.rs:20:5: unused import: std::os::fd::AsRawFd`.
Linux check emitted no warning. Development runs were green; the intentional runner
failure is `injected stub failure, not a GPU failure`, captured as child exit 7 and
runner exit 1. The raw dry-run manifest always says `gpu_claim: false`.

Latest explicit stub receipt at the checked tip:
`raw/cpu-stub-20260919T073624Z-69985/` plus `day3-dry-run.log`.
Earlier stub receipts are retained as development evidence, not relabeled as final-tip
runs. No GPU/device/server/build result is inferred from a stub log.

## Numbered blockers / lead decisions

1. **D-owned fixture visibility:** approve/export D's private PeerCapacity test adapter,
   or provide a public construction seam, then run one combined B scheduler/direct
   restore → D capacity/fence/release schedule. Separate B direct and D peer test passes
   are not that integrated result. No out-of-lane edit was made to pretend otherwise.
2. **Full server/engine compile and patch enablement:** no nvcc/GPU on this Mac. Patch
   review details remaining bootstrap, model-generation/identity provenance, exact native
   layout and byte-accounting audit, and fixed-arena handoff. Applying the patch alone
   leaves `tier=None`; legacy green tests would prove only off-path compatibility.
3. **Real consuming active materializer/native gate:** current CPU images/owner fakes
   are not CUDA attention operands. Generic host/NVMe active eviction during generation,
   token/logit/full-state identity and fence/rollback/churn/graph/spec boundaries remain
   unimplemented/unrun. Runner refuses final qualification rather than manufacturing
   a GPU receipt from legacy prefix tests.
4. **Rig/artifact/fit envelope:** lead must provide an exclusive non-serving Linux 5090
   window, immutable local-NVMe Qwen artifact/token inputs and a source/artifact-bound
   full run-gen peak-memory envelope. Fit refusal never substitutes a KV/weight format.
   Full 128k/262144 and peer/PRO-pair gates remain blocking under `/tmp/memra-gpu.lock`.
5. **Measured policy and serving/default evidence:** no chunk sweep, calibrated restore
   frontier, real route bandwidth, sustained shared-governor fairness, both-order N≥5
   A/B or serving latency distribution has run. No performance/default/support claim.

## Time, worktree and closure

Approximately **0.65 agent-hours** for this continuation (about 39 minutes including
re-entry, implementation, inspection and checks). Prior agents' actual time is not
recomputed. WP-B remains inside its **10 agent-day budget**; no hours/day conversion
or completion of GPU-dependent milestones is implied.

All task-created patch scratch copies and unit-test temporary directories were removed.
The open lane worktree/branch remain intentionally available for the lead's next
integration milestone; no merge/bank/abandon decision has closed the lane. Only B code
and receipts were staged; no unrelated dirty work was absorbed. Final receipt commit
records a clean lane `git status --short` (empty) after committing the listed evidence.
