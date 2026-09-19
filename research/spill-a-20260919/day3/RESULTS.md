# WP-A day 3 — CPU substrate delivery and remaining gates

Repository **avifenesh/memra**, branch `lane/spill-a-20260919`.
Verified Rust/source tip: **`3adf6325f5fb41c87ff22391dbdef3ec4937d16d`**.
Later receipt-only edits change the verifier/report, not the Rust bytes pinned in
`source-manifest.json`. **Local commits only: not pushed, main-merged, deployed,
Linux-executed, GPU-qualified or live-verified. Full tier suite is RED on a D-owned
v1.1 directed-route test; the A storage suite passes.**

## Inherited versus completed

Resumed at `5150c257` with five uncommitted files. Read the complete diff before
editing. The inherited direct-write/governor implementation was partly written:
formatting was unfinished and `tests/storage/mod.rs` referenced absent `day3.rs`.
Kept the implementation; added its missing tests and checked alignment/short-write
behavior. **No inherited work was discarded.** Replacing benchmark/test fake
budgets with B's production Governor was inherited intent, completed here.

- `87308125`: finish inherited governor/direct-write slice and three tests.
- `cfc4473e`: merge lead engine dependency/bin wiring at `914229ae` after securing
  the dirty work. No hand-edited shared manifest/dependency/lock changes.
- `603f425f`: extent-scoped leases and bounded background CPU byte reads.
- `325e628f`: merge v1.1 schedules (`ce49cd86`) at lead request.
- `b0296f9a`: repair reproduced quarantined-pool unknown-shutdown defect.
- `3adf6325`: strict background schedules, io_uring proposal and fail-closed runner.

Only A-owned source/tests/receipts were edited. Lead merges also carry lead-owned
shared tests/docs and engine manifests, not independent A amendments. Main was
inspected clean and never edited. No new dependencies, MEMRA_* reads, CUDA kernels,
numeric program, default selection or board movement.

## Implemented surfaces / C asks

### Shared governor — done for this CPU layer

`storage_bench.rs` and A storage tests use `memra_tier::tier::Governor`, not a
second runtime ledger. Pool physical backing is charged once as **pageable**
(the fake is NOT CUDA-pinned); slices hold the common pin. Tests exercise optional
refusal, protected mandatory headroom, mandatory admission, hard quota refusal,
Busy release while slices/pool remain, and explicit final credit. Background
fixtures share one actual governor for store, queue and fake pool.

### C ask 1: bounded/lazy selected extent — implemented, additive

`object_store::{ExtentLease, ExtentStore::lease_extent/read_extent/release_extent}`
reserves one selected framed chunk plus its framed root, not all table payload.
The caller's NVMe request is a ceiling; the actual extent+root requirement is
charged. Selection is sealed inside the lease: there is no arbitrary sibling
read argument. Admission verifies selected bytes; each read verifies them again.
Failure releases the new charge. Foreign/released capabilities fail closed.

Correction to the original premise: `lookup` already read only root metadata,
not every payload; **`lease`** was the whole-object validation/charging path.
Frozen `ObjectStore::lease` remains that explicit whole-object mode, unchanged.
A bounded-count test proves only the selected payload is read; a missing sibling
is tolerated by selected-extent reads but rejected by the whole-object mode;
corruption after leasing is rejected and explicit release still succeeds.

Root parsing still scans canonical metadata bounded at MAX_ROOT=1 MiB. This is
NOT an unbounded model-scale catalog or an O(1) root index; larger tables require
bounded manifests/sharding rather than raising limits silently. Duplicate active
extent reservations conservatively include a root each; deduplicated physical
NVMe accounting is not claimed.

### C ask 2: background CpuTransfers — implemented for FileBackend CPU bytes

`CpuTransfers::enable_background(workers)` is explicit and must precede any
submission. It uses existing `BoundedReader` std threads/channels. Original
synchronous `drive` behavior remains unchanged unless this mode is selected.
No frozen trait/wire change and no owner/device capability made Send.

Owner-side `BackgroundStore::prepare_extent` checks root/shape, reserves selected
extent and bounded aligned scratch via the common governor, and opens a retained
file descriptor. Worker owns only descriptor/source and CPU destination slot;
it verifies complete encoded content address, header, padding, and valid checksum
before copying bytes. Admission failure is an indexed failed completion **after
acceptance**, never a false zero-accept error. `poll`/`progress` harvest at most the
configured in-flight bound without waiting; all original item positions survive.

Cancel revokes publication but leaves producer buffers/charges alive. A complete
host destination can be taken once; its lifetime prevents retirement until the
consumer releases it. Extent/scratch charges remain until explicit retirement;
retired tombstones still consume capacity until acknowledge. Host readiness never
constructs GPU readiness. Frozen cancellation and post-completion-cancel schedules
run on the background implementation, alongside saturation, missing/corrupt item,
exact-byte, take-once, consumer-retirement and charge-conservation checks.

**Bounded background I/O is not end-to-end serving asynchrony:** metadata lookup,
canonical decode, file open and admission remain on the owner. Only FileBackend
currently implements the additive preparation interface. Native pin/DMA and
prepared catalog/descriptor caching remain unqualified follow-up work. Scratch
allocation is pageable and bounded per item, not a per-request CUDA pin. Native
allocator/catalog overhead accounting still needs its actual adapter receipts.

### Linux O_DIRECT — implemented and cross-checked, not executed

`io/direct.rs` validates 4096-byte buffer/offset/length alignment and checked
range arithmetic for reads AND writes. Exact loops retry EINTR, reject zero or
unaligned short progress, and never retry an unaligned residual. Creation is
exclusive; FileBackend publishes only after complete framed direct write (and
fsync when Persistent), cleaning private staging files on failure. No buffered
header/direct-payload mixture or silent fallback. Linux x86_64 target checks pass;
aarch64 constant is source-checked against cached libc, not target-executed.

CLI: `buffered`, `uncached` (Linux direct-read/buffered-write), and `direct`
(Linux direct-read/write). Mac `uncached` stays labelled F_NOCACHE development
fallback, while `direct` refuses Unsupported. Tests on Mac exercise aligned
buffer/exact-loop rejection with ordinary descriptors; they are NOT O_DIRECT
qualification. A Linux-only test exercises actual direct extent publication when
run there. Cross-target checking compiled it but did not execute it.

## v1.1 defect — exact before/after

Lead's added fixture first needed only a type binding update from its old
`support::Governor` to A's production `SharedGovernor`; no assertion was changed.
That compiler diagnostic is retained in `raw/v11-merge-type-error.log.gz`.

Exact command both times:

```sh
cargo test -p memra-tier --offline --test storage \
  day2::revision_v11_pinned_quarantine -- --exact --nocapture
```

Before, exit **101**, `raw/quarantine-before.log.gz`:

```text
thread 'day2::revision_v11_pinned_quarantine' ... revision_v11.rs:215:5:
assertion `left == right` failed
  left: Ok(())
 right: Err(Busy)
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 35 filtered out
```

After, exit **0**, `raw/quarantine-after.log.gz`:

```text
test day2::revision_v11_pinned_quarantine ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 35 filtered out; finished in 0.00s
```

`pool::Inner::drop` now deliberately retains quarantined backing **and its common
governor pin** through unknown shutdown. No fake retirement or relaxed test. This
CPU fake has no genuine all-use recovery authority, so unknown storage remains
retained until process exit. Ordinary no-quarantine shutdown still releases its
pin, proven by the paired strict lease schedule. A real drain-proof recovery API
is not invented here.

## Rig runner / io_uring

- `../rig-cells-a.sh`: fresh Linux/NVMe validation, canonical 5090 flock across
  window, unique owned scratch and raw receipts, build, CPU roundtrip/restore,
  existing pinned-worker A2, raw tee before JSON parsing, error-preserving status.
- `../CELLS.md`: exact scope/update and queued A2/M1 interfaces.
- `../IO-URING-PROPOSAL.md`: lead dependency fragment (exact release deliberately
  unselected), same bounded ownership interface, cancellation/CQE rules and M1
  decision cell. No dependency/ring implemented or kernel module installed.
- Runner's GPU A2/M1 interfaces remain absent in the CLI. Five-second unsupported
  probes retain stderr and the runner **exits nonzero**, even if a future binary
  accepts spellings without the missing collector. No trace/telemetry skip-pass.
- Stub test repeats the same **32-command** plan twice, including 24 filesystem
  invocations and four blocked-interface probes. No GPU, lock, build or filesystem
  benchmark ran in dry-run. This partial runner is not full A2/M1 delivery.

## Executed checks

`commands.json` lists exact argv/exit; lossless `raw/*.log.gz` retain output.
Reproducer: `python3 research/spill-a-20260919/day3/verify.py`; it runs independent
checks even when the aggregate tier suite fails, and finally returns nonzero.

| Exact command | Exit / observed result |
|---|---|
| `cargo fmt --all -- --check` | 0; empty output |
| `cargo check -p memra-tier -p memra-kv --offline --all-targets` | 0; `Finished dev ... in 0.10s`; existing memra-gguf AsRawFd warning on Mac |
| same check plus `--target x86_64-unknown-linux-gnu` | 0; `Finished dev ... in 0.09s`; **compile/type-check only** |
| `cargo test -p memra-tier --offline` | **101**; bank 31 pass, contracts 44 pass; peer 14 pass / 1 fail, stopped before storage/placement/docs |
| `cargo test -p memra-tier --offline --test storage` | 0; **37 passed; 0 failed; 0 ignored** |
| `cargo test -p memra-tier --offline --doc` | 0; **4 passed** |
| `cargo test -p memra-kv --offline` | 0; **44 passed** |
| `cargo clippy -p memra-tier --offline --all-targets -- -D warnings` | 0; `Finished dev ... in 1.59s` |
| `git diff --check` | 0; empty output |
| `bash tools/check-flags.sh` | 0; `runtime literal reads=864`, `no uncovered runtime names` |
| `bash -n research/spill-a-20260919/rig-cells-a.sh` | 0; empty output |
| `python3 research/spill-a-20260919/day3/test_runner.py` | 0; `RUNNER_DRY_PASS: two identical 32-command plans; 24 CPU filesystem + existing A2 + 4 blocked probes; no hardware ran` |
| `python3 crates/memra-tier/tests/contracts/fixture_reference.py --check` | 0; `tier-contract-v1: 8 payload + 3 canonical wire SHA256 pins match` |

Aggregate failure, quoted rather than inferred:

```text
fake::revision_v11_directed_capacity_lane_fix_required
... tests/contracts/revision_v11.rs:567:18:
denying one directed route must not deny its reverse: Unsupported
test result: FAILED. 14 passed; 1 failed; 0 ignored
```

D-owned source/test was not edited. Lead was notified; A is not claiming that its
storage pass fixes this aggregate failure. Initial local iteration errors were
missing Digest import, private helper visibility and a test fixture expecting an
optional admission while mandatory pageable capacity was still held; repaired
before final checks. No scored numerical measurement was produced.

## Numbered blockers / lead integration

1. Merge D's directed-route fix and re-run the integrated tier suite; current
   exact tip fails that strict v1.1 test. Peer green cannot be inferred from A.
2. No Linux OS execution or actual local-NVMe/O_DIRECT measurement. Cross-target
   check is not syscall/alignment/durability/physical-I/O evidence.
3. No nvcc/GPU here (command lookup found no nvcc). Full engine/storage-bench
   Cargo build, real pinned H2D/D2H, consumer/graph retirement, 5090/PRO and native
   model/serving gates have not run. CPU linked CLI is exercised in storage tests.
4. Full A2 GPU/M1 trace engine and 250 ms/physical-I/O/steady-state collector remain
   implementation work. Partial runner refuses qualification, not skip/pass.
5. Root catalog/descriptor caching/sharding and native metadata/allocator overhead,
   NUMA, process-shared orphan GC and real all-use quarantine recovery remain open.
6. io_uring is a proposal only; actual pinned version/ring implementation and
   both-order on-rig comparison remain required. No backend/default win claimed.

Approximate resumed-session effort: **0.5 agent-hours**, GPU time **zero**, versus
WP-A's **7 agent-day** budget. Prior interrupted-agent time is unknown and not
silently counted as zero. Lane stays open for lead integration/follow-up, so its
worktree/branch are retained intentionally, not a closed-lane cleanup omission.
