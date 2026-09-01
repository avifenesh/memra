# Dual-active PP-2 decode increment 1 pre-sec7 results — invalidated

> **Not final evidence.** The orchestrator required a rebase onto the sec7-fixed main after this
> battery completed. These source-`365e1eb7` receipts are preserved for provenance only; no gate or
> increment verdict below carries across that rebase. Final evidence must come from a fresh rebuild
> and battery on the rebased source.

Date: 2026-08-11

Lane: `lane/cx-dualpp1`

Hardened base: `main@126e6642` (contains `afb9be7b`)

Final implementation and scored source: `365e1eb71c6b635872447d4d1af1aeac4d7c087f`

## Verdict

**HOLD.** Increment 1 is complete for orchestrator review. The worker now schedules dual-active
PP-2 for arbitrary live width, and the final-source box1 battery is green. The N=5 interleaved
curve clears the frozen +15% c>=8 floor, while the ten-boot cross-device soak records equal use of
both alternating slots and zero collisions.

This is not a default flip. Dual-active decode remains default OFF and requires both
`MEMRA_DUAL_PP=1` and `MEMRA_PP_OVERLAP=1`. Promotion beyond HOLD requires the separately requested
admission-residual re-gate; it was not part of this increment. No merge, tag, push, formatting run,
or generated performance-board change was made.

## Base and evidence boundary

The initial lane fork predated the operator-metrics hardening merge. When the orchestrator's merge
gate arrived, the running pre-rebase box1 process was stopped, its partial logs were marked invalid,
and the branch was rebased onto `main@126e6642`. The complete final battery was then rebuilt and run
from `365e1eb7`; no increment-0 or pre-rebase verdict is inherited.

The invalidated partial receipt is retained only for provenance under
`raw/box1-pre-rebase-invalidated/`. All scored evidence cited below is under `raw/box1/` and names
the final source revision.

## What changed

- The worker's exact batch cap is now a per-wave cap when and only when dual PP, overlap, and a
  ready PP-2 batched path are all active. A tick may combine two exact waves; wider live sets are
  processed as ordered combined chunks plus a remainder. Width one stays on the serial path.
- Each combined chunk passes an explicit balanced midpoint to the engine. The engine requires
  `ceil(c/2)` / `floor(c/2)` and validates the larger wave against the exact cap, rather than
  treating the combined width as one unsupported numeric class.
- Stable lane-priority and request order are preserved. Focused policy tests cover c1..16, odd
  mixed-membership order, exact-tier caps, and c1 fallback; the split helper gate covers c1..32.
- PP boundary accounting now exposes completed slot pairs, per-slot uses, and rejected same-slot
  pairs. The entire `dual_pp` metrics block is operator-only; completion-domain and tenant scopes
  omit it even when the snapshot is populated.
- The final harness performs one release rebuild before any GPU gate, holds one GPU lock through
  the whole battery, and records raw stdout/stderr, exit receipts, manifests, and thermal traces.

## Final-source exactness

The detached box1 driver acquired `/tmp/memra-gpu.lock` as a PPID-1 session leader and retained the
same lock descriptor through build, correctness, soak, and performance. It rebuilt the release
server in 3m52s and the release gate binaries in 12.46s before starting the GPU battery; both build
exit receipts are zero.

| Gate | Result |
|---|---|
| `kernel-check` | ALL GREEN: 86 cells, 21 skipped |
| Direct PP-2 matrix | B=1..16 split and unsplit PASS; 140,238,848 f32 split logits, 0 differing bits |
| Dual liveness | B=2..16 PASS; overlap counter advanced by 8 in every cell |
| Fail-closed negatives | Single-slot and host-bounce dual configurations both refused before decode |
| Strict decode-batch | ALL GREEN: B=1 identity, B=4 isolated identity, device sampling/lean-logits identity |
| `run-gen` | Prefill/decode argmax 128799 MATCH; batched-prime/tokenwise argmax MATCH |
| `run-spec` | K=1..8 self-consistency PASS (8/8) |
| Final driver | `CORRECTNESS_PASS` |

The B=16 direct cell compared 16,498,688 f32 logits with zero differing bits. Widths above the
per-wave cap use the cap-bounded sequential two-wave oracle, so B=9..16 never borrow an unsupported
combined-width reference.

Primary receipts:

- `raw/box1/build/server.log`, `server.exit`, `gates.log`, and `gates.exit`
- `raw/box1/correctness/driver.log` and `SHA256SUMS`
- `raw/box1/correctness/kernel-check.log`
- `raw/box1/correctness/strict-batch.log`
- `raw/box1/correctness/run-gen.log`
- `raw/box1/correctness/run-spec.log`

## Cross-device slot-collision soak

The soak alternated ten fresh serial/dual server boots on box1's two RTX PRO 6000 Blackwell Server
Edition devices. Boots 1 and 2 covered the complete c1..17 one-hash matrix; the remaining boots
used rotated mixed widths and mixed interactive, judge, and harvest membership.

| Receipt | Result |
|---|---:|
| Points per arm | 101 |
| Requests / golden matches per arm | 929 / 929 |
| Total requests / golden matches | 1,858 / 1,858 |
| One-hash cells | 34 / 34 PASS (serial and dual, c1..17) |
| Dual boots | 5 |
| Completed dual slot pairs | 9,104 |
| Slot 0 uses | 9,104 |
| Slot 1 uses | 9,104 |
| Same-slot collisions | 0 |

The frozen completion hash was
`21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de`.
Per-dual-boot slot-pair counts were 1,474, 1,906, 1,907, 1,905, and 1,912; every boot had equal
slot use and zero collisions. Timing instrumentation was disabled, and sampled operator metrics
reported zero CUDA timing samples and zero dropped samples.

Thermal regime: N=13,990 samples at 250 ms, 28--51 C and 180--2422 MHz, with no artificial
cooldown. The summary is `raw/box1/soak/summary.json`; the raw boot, request, metrics, server, and
thermal receipts plus `SHA256SUMS` sit beside it.

## Frozen performance curve

Protocol: N=5 interleaved serial/dual arms at every c2..17, rotated width order, 512 completion
tokens per request, and one inherited GPU-lock hold. The metric is aggregate completion tokens
after first visible token divided by the decode-window duration. All 160 points and 1,520 request
rows completed without error.

| Width | Serial median tok/s | Dual median tok/s | Delta |
|---:|---:|---:|---:|
| c2 | 130.992 | 132.767 | +1.355% |
| c3 | 145.919 | 146.049 | +0.089% |
| c4 | 155.414 | 171.853 | +10.577% |
| c5 | 160.852 | 175.680 | +9.219% |
| c6 | 164.801 | 192.250 | +16.656% |
| c7 | 167.932 | 192.427 | +14.586% |
| c8 | 170.707 | 204.529 | +19.812% |
| c9 | 158.342 | 202.650 | +27.983% |
| c10 | 160.602 | 211.503 | +31.693% |
| c11 | 163.205 | 208.383 | +27.682% |
| c12 | 164.674 | 216.456 | +31.445% |
| c13 | 166.217 | 214.246 | +28.895% |
| c14 | 167.839 | 220.456 | +31.350% |
| c15 | 169.065 | 217.587 | +28.700% |
| c16 | 170.146 | 223.792 | +31.530% |
| c17 | 163.098 | 208.400 | +27.776% |

The frozen floor is >=+15% at every c>=8 median. The minimum is c8 at +19.812%, so the floor
passes. Across the 80 dual points, operator metrics record 40,956 slot pairs, 40,956 uses of each
slot, and zero collisions.

Thermal regime: N=39,804 samples at 250 ms, 30--49 C and 180--2422 MHz, with no artificial
cooldown. The reducer output is `raw/box1/perf/summary.json`; all point summaries, request rows,
operator metrics, server logs, the continuous thermal trace, and `SHA256SUMS` are retained beside
it.

## Handoff

The branch is intentionally left unmerged and untagged for orchestrator review. The implementation
and evidence support the increment-1 HOLD posture only: arbitrary-width dual-aware scheduling and
the alternating-slot collision gate are complete, while default activation remains outside this
increment.
