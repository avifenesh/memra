# Day 7 — owner-thread expert proxy and native row device publication

Repository: `avifenesh/memra`, lane `lane/spill-c-20260919`. Required integration
merge `c5d055ad` includes `origin/lane/spill-integ4-20260920` at `5ffe935c`.
This is rented RTX 5090 **development correctness evidence**, not a default,
performance promotion, production qualification, PP qualification, or model
support-state promotion. All GPU cells use the canonical collector/lock and
record **400 W configured / 600 W maximum**, with 250 ms telemetry. Storage is
ordinary container-overlay development storage, not a claimed NVMe spill result.

## Owner boundary

`bank/owner_proxy.rs` implements the approved owner-thread-registry alternative:

- `ExpertBankOwner` and the registered `ExpertDispatchBank`/Rc leases stay on the
  CUDA owner thread. The guard is !Send; `ExpertBankProxy` and `ExpertLeaseToken`
  are Send+Sync identities only, with private fields and non-reused ids.
- Moving a proxy/token to a worker compiles, but using it there returns
  `WrongOwner` **before** validation/staging/publication/retirement. This is not
  a channel/RPC implementation, and does not claim PP worker qualification.
- Pending demands are bounded. Payload is borrowed only in an owner callback.
  Close refuses outstanding demands; unknown completion retains leases/charges,
  including on owner/thread teardown. Drop never fabricates completion.
- CPU tests actually send handles across a thread, check owner-only stage/publish,
  wrong/foreign/stale ids, bounded capacity, failed-finish retry, reentrant refusal,
  and teardown without automatic finish. No unsafe Send/Sync implementation.

`MoeSlotCache` stores only the proxy and token. The typed installation is absent
by default and requires an untouched SLRU cache. All synchronous admission
routes converge on bank staging; detached prefetch is disabled and freeze-profile
restage refuses. Native slot addresses, routing, expert layouts and arithmetic
remain unchanged. The qualification path explicitly drains H2D, and a slot is not
published in table/queues until completion is observed. Unknown completion keeps
it quarantined, preventing a later resident fast path from bypassing the token.
This last failure-path tightening was found during self-review (`d0acf6f0`).

`banked_residency/native.rs` authenticates the already-open GGUF inode, compares
native HostExps bytes to its exact expert extents, uses original ids and authoritative
per-expert layouts, and derives MTP checkpoint numbering from its ModelPlan block.
Resident slabs, parallel expert banks, auxiliary scale planes and non-approved
artifacts refuse; they are never silently substituted. The bank is bounded to 16
host records, at most 16 MiB per record and 256 MiB cached data. Both gates expose
`--experts-via-tier`; no new MEMRA environment read or runtime default was added.

Typed-door decision: default OFF, qualification-only; **decide-by: 2026-10-04**.
Keep native transfer/owner integration separate from any later tuning decision.

## Immutable expert artifact

- Source: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF@5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`.
- File: `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, 18,209,036,576 bytes.
- SHA256: `df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf`.
- Downloaded with bounded `curl --retry` to a new file, checked before rename,
  checked again with its SHA sidecar; **ARTIFACT_SHA256_MATCH**. Existing artifacts
  were not changed. Raw download transcript/script and sidecar are retained.

## Expert receipts

All cells use prompt ids `[55, 88, 13]`, `MEMRA_NGEN=32`, and the same existing
`MEMRA_MOE_RESIDENT=0` cache baseline. Baseline gates ran first. Source revisions
and binary hashes are separate sidecars; a receipt-only commit does not rebuild a
binary. Do not compare run-gen's numeric class with run-spec's loaded-MTP class:
compare each ON/OFF pair with its own baseline.

Initial baseline (`e2187043`) and banked run-gen (`3d9e28b0`) both report:

> prefill argmax=198  decode argmax=198  logit maxdiff=6.482e-1  MATCH

Their full 32-token tapes are byte-identical. The banked run reports:

> [experts-via-tier] physical_reads=28758 owner_close=Ok(())

Baseline and banked run-spec (`102901d1`) each exercise K=1..8 and report:

> === SELF-CONSISTENCY PASS ===

Each of the eight rows reports `self-consistency: PASS (identical to plain target)`;
the ON/OFF plain tapes and all eight acceptance rows match exactly. Banked spec:

> [experts-via-tier] physical_reads=31710 owner_close=Ok(())

Raw receipts: `rented-5090-20260919/day7/{baseline-gen,baseline-spec3,experts-gen,experts-spec}/`.
Final source after failure-path tightening and comment cleanup is `18b2f092`.
Its `final-gen-off`/`final-gen-on` repeats report the same argmax MATCH and
byte-identical 32-token tapes; `final-spec-off`/`final-spec-on` report K=1..8
SELF-CONSISTENCY PASS with unchanged acceptance rows. Initial measurements
remain preserved, not rewritten as receipts for that later source. The final
`device-integrated` receipt at `efd3fbeb` reports the same device-publication
verdict above; recovered from the collector after the session restart and
hash-replayed locally before this report was sealed.

## Device-publication rows — banked scratch integration

The scratch source `5022c173` imports A's `tier_transfer.rs` **byte-identically**
from `06fda471` and binds `GatheredRows::submit_device` to `CudaTransfers` and its
native `CudaPinnedLease`. It performs H2D, validates producer completion and
consumer-ready publication, D2H-checks that exact sealed allocation, records the
real D2H consumer event, observes retirement, acknowledges, and exclusively hands
back the **same allocation** to the existing PLE projections. No extra H2D/D2D
copy and no different mathematical program is substituted.

Important boundary: after exclusive handback, native cudarc owns the projection
allocation/lifetime. This is **not** a claim that the transfer governor retains
ownership through those later projection kernels. The byte-check D2H is a
qualification instrument, not a performance arm.

`qwen4exp_gpu_gate --rows-via-tier --device-publish` reports verbatim:

> qwen4exp-gpu-gate PASS [rows-via-tier: BIT-IDENTICAL PLE outputs + convolution state; encodings=2 cases=16 values=6144 tier_calls=16 forced_read_chunks=144 device_uploads=16 exclusive_handback=true budget_drained=true]

The whole existing tiny gate passes too. These are independent F32 and BF16 tiny
fixtures (not format substitutions for a scored checkpoint). Initial scratch raw
receipts: `day7/device-publish2/`, `device-verdict.tsv`, source and binary sidecars.
Scratch replay on the final expert lane is `efd3fbeb`; its full tier CPU suite
passes. No V4.1/Engram model implementation or serving path was added.

`DAY7-DEVICE-PUBLISH.diff` banks **only C's integration delta**. To replay, start
from C's `18b2f092`, take `crates/memra-engine/src/tier_transfer.rs` from A's
`06fda471`, then apply that diff. It includes the module-export fragment for the
lead to resolve against integrated module ownership; do not separately duplicate
A's implementation. Device changes are NOT active in the ordinary C branch.

## Checks and retained failures

- Full `cargo test -p memra-tier --offline --no-fail-fast`: **188 passed** including
  doc tests, on Mac and native Linux. The three new owner-proxy tests pass.
- Tier strict all-target clippy and Linux-target check pass.
- Engine lib + run-gen/run-spec Linux-target DOCS_RS strict clippy pass, with the
  existing compile-only MMQ hash placeholder. This proves types, not CUDA work.
- Native Engine lib + run-gen/run-spec strict clippy passes on the CUDA toolchain.
- Device scratch Engine lib + gate Linux-target strict clippy passes.
- fmt, diff whitespace, flags census, and unskipped push hooks pass.
- `verify-day7.py` replays raw log/telemetry digests, power metadata, nonempty tapes,
  all eight spec rows, and nonzero bank/device engagement. Its tampered-log red
  probe refuses `receipt hash mismatch`. The collector retains its honest
  `executed-not-qualified`/`qualification:false` status.
- Native **macOS Engine** check did run and failed on existing Linux-only
  `spill_pread.rs` libc APIs (`O_DIRECT`, `posix_fadvise`, etc.). That is not changed
  to a green result by the Linux-target compile. Its raw errors are retained.
- The first strict clippy found a redundant Copy clone; fixed before initial push.
- The SLRU trace whole-source pin fired after admission wiring changed. Inspected
  the unchanged replacement-policy code, regenerated, and confirmed only the
  source hash changed: all 2,013 recorded synthetic decisions are unchanged.
- Two baseline-spec collector attempts refused `[Errno 11] Resource temporarily
  unavailable`; no lock bypass. The third ran after the lock became available.
- The first device gate refused missing frozen `bank-bytes-goldens.tsv`. Copied
  the historical pre-streaming golden (SHA256
  `4bca9c6a544c0fa09b29d8919ae16936543ff9bed001ec39f96826b8fed43a3e`)
  beside the receipt and reran. **No goldens were minted.**
- Mac GitHub HTTPS became unavailable mid-run. Two bounded attempts failed.
  A bundle safely moved missing source to the build scratch, but remote push
  refused missing authentication. No credentials were copied. A temporary
  loopback SOCKS forward on the existing approved SSH master let the local Git
  credential helper push over TLS; every push still ran hooks, and the forward
  was canceled after each use.

## Scope / lead integration

These gates establish the measured default-OFF owner/byte paths on a rented RTX
5090. They do not establish PP/RPC service, full production artifact admission,
auxiliary-scale experts, a fast bank default, or PRO-pair release qualification.
No performance decision is drawn from the timing text in correctness logs.

Lead owns integrated module wiring and the combined-lane review/battery. The
scratch device delta and native receipts are banked, not merged to main. The
primary C worktree remains an active handoff lane; only task-created disposable
scratch resources are cleaned. Existing prior C scratch work (including its
uncommitted module fragment) and all other lanes remain untouched.
