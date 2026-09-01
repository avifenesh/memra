# Boot-time PP peer byte-integrity probe — results

Date: 2026-08-11
Branch: `lane/cx-peerprobe`
Base: `1592253f`

## Verdict

**PASS for the local DEV/BUILD phase.** PP-N now validates peer-copy bytes at boot instead of
treating CUDA capability or API success as a correctness receipt. Native P2P fails startup on any
API error or byte mismatch, quoting the exact mismatch count. A process explicitly launched with
`MEMRA_PP_HOST_BOUNCE=1` logs the same corruption and proceeds only on the already-selected
host-staged transport. `MEMRA_PEER_PROBE=0` is a loud diagnostics-only escape hatch.

The injected failure test, full release build, engine library tests, locked single-device
transport smoke, and manifest-aware RTX 5090 `kernel-check` all pass. This workstation has one
GPU, so it cannot produce a native peer receipt. The real two-card PP-2 clean pass, measured probe
latency, and golden-output hash are **PENDING for the Vast window**; no native result is claimed or
fabricated here.

## Implementation

`PpNRt` performs two deterministic probe classes before the first model weight upload:

- fixed 16 KiB through legacy allocations, immediately after peer access is enabled and before
  default-pool grants;
- production stream-ordered `BoundarySlot` payloads at 1, 8, and 16 `[n_embd] f32` rows after
  pool grants, from the authoritative model width.

Every adjacent cross-device PP boundary is checked in both directions. The legacy preflight uses
one device allocation in each CUDA context, a deterministic xorshift source pattern, inverse
destination poison, the production `cuMemcpyPeerAsync` API on the source stage stream, synchronous
destination readback, and an exhaustive byte comparison. That allocation class keeps the first
pass independent of the pool grants that follow. The production class constructs real
`BoundarySlot` objects backed by stream-ordered allocations and invokes the same shared TX/RX
functions as decode, including events and the RX-side local copy; its slots are inverse-poisoned
before each transfer. Both classes are process-once boot work; logs emit copy count, skipped count,
mismatch total, elapsed milliseconds, and the largest clean production payload.

With `MEMRA_PP_HOST_BOUNCE=1`, diagnostic peer and pool access are temporary. Pool access is reset
to `PROT_NONE` and every probe-enabled direction is disabled with `cuCtxDisablePeerAccess` before
the host-staged transport proceeds. Failure to tear down either access class refuses startup.

The model loader initializes the geometry pass before `layer_engine()` uploads the head or first
layer. Cache creation repeats the initializer idempotently before the first forward. A different
model width in the same process fails rather than reusing an unverified geometry.

The pure compare/policy seam accepts injected host bytes. Its corruption test flips three known
offsets in a 16 KiB readback and proves:

- native policy returns `Err("3 mismatched byte(s)")`;
- host-bounce policy returns `ProceedWithHostBounce { mismatches: 3 }`.

No CUDA context or peer hardware is created by that test.

## NVIDIA R570 basis

NVIDIA's R570 Data Center driver release notes document that GPUDirect P2P over PCIe on Ada and
older GPUs can suffer run-time silent corruption when a host does not preserve the required order
of GPU-initiated posted transactions under Relaxed Ordering. NVIDIA's `simpleP2P` excerpt reports
peer access as available and prints bandwidth before byte verification fails. The same notes say
the driver-525+ mitigation is selected using PCIe host-bridge IDs, while a guest OS may not expose
the exact topology and the mitigation may therefore not be applied when needed. Source:
[NVIDIA Data Center GPU Driver R570, version 570.133.20, known issues](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-570-133-20/index.html#known-issues).

This citation does not attribute memra's Blackwell Vast failure to NVIDIA's Ada-and-older
mechanism. The independent memra receipt is narrower: the Vast host reported peer capability and
successful CUDA calls while 16,320 of 16,384 bytes were wrong, and NVIDIA `simpleP2P` also failed
there (`research/p2pvast-20260810/RESULTS.md`). Both receipts support the gate design: capability,
success status, and apparent bandwidth cannot replace a byte comparison.

## Local validation

All host builds/tests used `nice -n 15 taskset -c 0-7`. Every GPU command acquired
`/tmp/memra-gpu.lock`.

| Gate | Result | Raw receipt |
| --- | --- | --- |
| Injected corruption test | **PASS**, 1 passed / 0 failed; exact three-byte native failure and host-bounce proceed decision | `raw/local/corrupted-readback-test-followup.log` |
| Focused PP policy/geometry tests | **PASS**, 4 passed / 0 failed | `raw/local/pp-policy-tests-followup.log` |
| Engine library tests | **PASS**, 63 passed / 0 failed / 1 pre-existing CUDA-only ignored on exact final source | `raw/local/cargo-test-engine-lib-followup.log` |
| Full `cargo build --release` | **PASS**, sm_120a, exact-final incremental rebuild 31.45s | `raw/local/cargo-build-release-followup.log` |
| RTX 5090 `kernel-check` with both required manifests | **ALL GREEN (101 cells, 1 skipped)** on exact final binary; the optional `sigrouter-served-replay` capture was unset, no required manifest cell was missing | `raw/local/kernel-check-full-followup.log` |
| Single-device transport smoke | **PASS** on exact final binary; same-context peer arm byte diff 0 and four alternating boundary slots byte diff 0 | `raw/local/pp-transport-smoke-single-device-followup.log` |
| Runtime flag inventory | **PASS**, no new drift beyond the existing frozen ledger | `raw/local/check-flags-followup.log` |

The single-device smoke checks FFI and boundary choreography only. It is not evidence that a real
PCIe peer path is clean and is not reported as such.

## Deferred native two-card cells — LANDED 2026-08-11 (box1 dualpp2 re-gate)

| Target cell | Status | Receipt |
| --- | --- | --- |
| Native peer probe on the real two-card PP-2 serve shape | **PASS — box1 2x RTX PRO 6000, 2026-08-11** | The dualpp2 re-gate's 10 production-shape boots (serial + dual arms) each ran the default-on peer probe over the native PP-2 boundary with zero mismatches and no skipped direction; every server log shows the benign `mismatches=0` telemetry and no `MISMATCH` sentinel (canary-verified grep: 0 on clean logs, 7/7 on injected faults). Raw: `research/dualpp2-20260811/raw/box1-regate/soak/`. |
| PP-2 content identity after clean native bring-up | **PASS — box1 2x RTX PRO 6000, 2026-08-11** | 929/929 golden matches per arm across 10 boots against pinned golden hash `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de`, plus a 34-point one-hash matrix at widths 1-17 and 0 slot collisions across 9123 cross-device pairs. Raw: `research/dualpp2-20260811/raw/box1-regate/` (verdict: `RESULTS-regate.md`). |

Early-merge rationale (ledger honesty, kimi `b8a2fe62`): the lane merged to main (5cf30a71)
ahead of these native receipts as a defensive fail-closed gate — a corrupt peer path refuses at
boot and can never serve garbage, so shipping the probe early only ever *adds* protection. The
native two-card window this section required is the box1 dualpp2 re-gate above; both cells are
now landed and the no-merge condition is discharged. No tag shipped between the merge and these
receipts (v0.71.0 predates the probe).

## Scope integrity

- No merge, tag, push, or release.
- No `cargo fmt`.
- No performance number, `current-board.json`, generated README block, or generated
  `docs/PERFORMANCE.md` block changed.
- Raw local receipts and `raw/SHA256SUMS` are committed beside this report; native Vast receipts
  remain pending.
