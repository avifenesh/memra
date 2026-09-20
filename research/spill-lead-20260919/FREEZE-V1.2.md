# Generic spill freeze v1.2 — additive CPU conformance

Repository **avifenesh/memra**, branch `lane/spill-e-20260919`.
Base: `b3487a03b0ee3f833c1157e7b7d68f2cb35a3843`. Date: 2026-09-19.
This is a conformance revision, not a changed runtime/wire program. No frozen
trait, persisted structure, fixture byte, manifest or lockfile changed:
**WIRE_VERSION remains 1**. No external dependency, MEMRA read, model/CUDA/FFI
code, dispatch/default, numerical program or hardware claim is introduced.

## Source-shared schedules and bindings

Canonical source: `crates/memra-tier/tests/contracts/revision_v12.rs`, re-exported
by `conformance.rs`. All v1/v1.1 schedules remain unchanged and enabled. New
bindings are in contracts `v12_bindings.rs`, existing lead-owned reference fixture
modules, and auto-discovered `crates/memra-kv/tests/contracts_v12.rs`. No A/B/C/D
implementation or lane-owned test file is edited.

| Schedule | Binding and assertions | Evidence boundary |
|---|---|---|
| `ready_owner` | Actual DeviceOwner factory with CPU backing; unbound/foreign owner, wrong allocation generation, independent state/src/dst stale epochs, each item's fence issuer/device/generation/missing-wait refusal, exact successful destination/ticket, Busy while bound, retired binding becomes unknown, explicit release to zero. | Fake CUDA owner **identity** schedule only. The caller must observe actual producer/wait/consumer/graph completion before binding/retiring; no fake FenceId proves a CUDA event. |
| `framed_logical_bytes` | Every segment independently: I/O bytes 0, logical, logical+8192 and u64::MAX do not affect acceptance; a one-byte short valid payload still returns exact ShortIo; checksum corruption and removed sibling refuse. | Snapshot mutation tests Completion::require, not physical fault injection or device traffic. |
| Per-trait byte bindings | TransferEngine::poll on framed reference NVMe transfer; PeerBackend::poll plus actual peer-publication short-valid refusal and framed success; actual C BankService/BoundedRowService completions over A ExtentStore→ObjectReader; ObjectStore reads exact logical payload from padded framed chunks; TierStore preserves full bundle with distinct valid/storage lengths. | Bank/row completion telemetry describes materialization, not coalesced disk traffic. Their snapshots are not native injected-short-I/O evidence. ObjectStore and TierStore expose no completion counter: no invented poll method. Tier binding is the reference control-flow fixture, not a new native B adapter. |
| Existing `peer_directed_grants` | Directly bound to D's exported `peer::test_support::FakePeerCapacity`, one injected governor, independent context/pool denial and forward downgrade/restoration, zero residual charge. | v1.1 was already PeerCapacity-generic; no trait rewrite or duplicate capacity backend. CPU grants are not P2P route proof. |
| `bank_source_install` | Actual C BankSource installation across valid, missing/ambiguous supplied source, wrong layout/generation, missing object, mismatched manifest, missing/ambiguous expectation, wrong length. Fresh pool/queue per case; no payload read during install; all charges drained. | Faulted metadata-only ObjectStore for installation; real framed ExtentStore separately exercises successful source→bank/row reads. No native loader/checkpoint support claim. |
| `kv_materializer` | Same generic schedule on B's actual QwenMaterializer (q8_0 K/q5_1 V) and PackedMaterializer (264-byte opaque record). All ten program identities, all three epochs, missing/corrupt sibling, stable allocation ID, full byte order, wrong consumer fence issuer/device/generation with retained live operands, successful retirement, double-retire and read-after-retire refusal, Busy accounting until owner release. | Packed record is a separate opaque fixture, never substitute model KV. CPU native-layout validation is not CUDA materialization/attention or graph-address proof. |

New tests: **7 contracts + 2 KV integration tests**, none ignored. The two
materializer bindings share expected outcomes; byte construction/capture differ
because the two record programs differ. Existing short/missing/corrupt/cancel,
quarantine, scale and source-identity tests remain intact.

## Owner duties and unchanged decisions

`consumer_fenced` means the consumer wait is installed, not that the consumer
finished. The native CUDA owner creates/adopts allocations, binds the exact
destination, validates complete outcomes, installs actual stream waits, and keeps
all source/destination/graph pins through last-use. Only observed full retirement
permits retire_binding/acknowledgement and explicit quota release. I/O workers
cannot mint device readiness. C's `READYVIEW-OWNERSHIP.md` maps existing owner,
copy/compute wait and unknown-shutdown seams; this revision does not implement them.

PR #518's logical-length correction and PR #519's GC/catalog/transaction fixes
are retained. io_uring stays **deferred**, pending the bounded-pread rig baseline.
No new `.cu`/FFI means no KERNELS inventory change. Docs registry gates remain
required; docs/TESTING now routes the CPU suites, collector and bootstrap.

## Verification and integration status

Initial targeted execution ran on macOS: all **9 new tests passed**, and package
Clippy `--all-targets --no-deps -- -D warnings` passed. The existing dependency-only
`memra-gguf/src/source.rs:20` unused AsRawFd warning remains untouched. An initial
compile error in the new test used `backing` instead of BankLease's `resource`
accessor; corrected before the passing run, no API change or suppression.

The full lane battery and exact source/log hashes are recorded in `v1.2/checks.json`
and lossless combined-output archives beside it. The separate
`INTEGRATION-DRYRUN-DAY5.md` records the exact refreshed A/B/C/D union, conflicts,
checks and unrun surfaces. Those receipts, not this schedule description, determine
whether the complete CPU integration passed. No engine/server/nvcc/GPU/serving,
Linux runtime, direct-I/O, NVMe ancestry, PRO-pair or four-card gate runs here.
Executed CPU/cross-compile checks are **not qualified hardware or model support**.

### Full lane receipt

Checked source: **`ebf437b799eb926faeaf56c97d2fc6ecaa4b66cc`**. All ten commands in
`v1.2/checks.json` actually ran and exited 0: fmt, macOS all-targets check,
Linux cross-target all-targets check, both package suites, strict package Clippy,
diff whitespace, flags, docs census, all eleven independent wire/payload pins,
and unchanged frozen contract/fixtures/root manifest/lock comparison.
**239 tests passed, zero failed/ignored** (KV 59 + new KV 2; tier unit 2,
bank 43, contracts 52, peer 18, placement 6, storage 53, compile-fail doctests 4).
No engine/server build or hardware/model gate ran. Docs census: 122 kernel-file
references resolve, five support tokens valid, router 40/60 lines, 864 runtime
names, 898 flag-table rows shaped correctly. No unrelated worktree modifications
were absorbed; the only untracked files during the run were this lane's verifier
and receipts. Receipt-only follow-ups do not change the tested Rust bytes.
