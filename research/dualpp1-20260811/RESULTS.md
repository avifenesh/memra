# Dual-active PP-2 decode increment 1 results

Date: 2026-08-11

Lane: `lane/cx-dualpp1`

Hardened base: `1592253ff5af5436af4c680ab614d6a9f02ff86b` (contains the required
sec7/reusepool tip `2ddb9bd2`)

Final implementation and scored source: `a8f24074331647baf79189f190a402054d1af314`

## Verdict

**HOLD.** Increment 1 is complete for orchestrator review. The worker now schedules dual-active
PP-2 for arbitrary live width, and the complete final-source box1 battery is green. The N=5
interleaved curve clears the frozen +15% c>=8 floor, while the ten-boot cross-device soak records
equal use of both alternating slots and zero collisions.

This is not a default flip. Dual-active decode remains default OFF and requires both
`MEMRA_DUAL_PP=1` and `MEMRA_PP_OVERLAP=1`. Promotion beyond HOLD requires the separately requested
admission-residual re-gate; it was not part of this increment. No merge, tag, push, formatting run,
or generated performance-board change was made.

## Base and evidence boundary

The orchestrator invalidated the earlier increment-1 battery after requiring the sec7-fixed main.
The branch was rebased onto `1592253f`, which contains `2ddb9bd2`, and the entire release build,
exactness battery, collision soak, and performance block were rerun from `a8f24074`. Main's two
`AbandonedWorkerLimit` mappings and the sec7 fail-closed-latch rearm behavior remain present; the
focused final-source rearm test passes.

No increment-0, pre-rebase, or pre-sec7 gate verdict is inherited. The two invalidated runs remain
under `raw/box1-pre-rebase-invalidated/` and `raw/box1-pre-sec7-invalidated/` for provenance only.
Every scored receipt cited below is under `raw/box1/` and names `a8f24074` as its source revision.

## What changed

- The worker's exact batch cap is now a per-wave cap when and only when dual PP, overlap, and a
  ready PP-2 batched path are all active. A tick may combine two exact waves; wider live sets are
  processed as ordered combined chunks plus a remainder. Width one stays on the serial path.
- Each combined chunk passes an explicit balanced midpoint to the engine. The engine requires
  `ceil(c/2)` / `floor(c/2)` and validates the larger wave against the exact cap, rather than
  treating the combined width as one unsupported numeric class.
- Stable lane-priority and request order are preserved. Focused policy tests cover c1..16, odd
  mixed-membership order, exact-tier caps, and c1 fallback; the split helper gate covers c1..32.
- PP boundary accounting exposes completed slot pairs, per-slot uses, and rejected same-slot pairs.
  The entire `dual_pp` metrics block is operator-only; completion-domain and tenant scopes omit it
  even when the snapshot is populated.
- The box1 harness takes one GPU lock before the fresh release build, keeps it through all GPU
  gates and measurements, and records raw stdout/stderr, exit receipts, manifests, and thermal
  traces.

## Final-source exactness

The detached `setsid` driver logged `/tmp/memra-gpu.lock` acquisition as its first action at
2026-08-11T14:08:23Z (`pid=sid=714442`). It retained that lock through build, correctness, soak,
and performance. The fresh release server build completed in 226.07 seconds and the release gate
binary build in 12.00 seconds before the GPU battery; both build exit receipts are zero.

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
combined-width reference. All seven build and correctness `.exit` receipts are zero, and the final
raw-log failure-pattern scan is empty.

Primary receipts:

- `raw/box1/launch.log` and `raw/box1/driver.log`
- `raw/box1/build/server.log`, `server.exit`, `gates.log`, `gates.exit`, and `SHA256SUMS`
- `raw/box1/correctness/driver.log` and `SHA256SUMS`
- `raw/box1/correctness/kernel-check.log`
- `raw/box1/correctness/dual-matrix.log`
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
| Completed dual slot pairs | 9,114 |
| Slot 0 uses | 9,114 |
| Slot 1 uses | 9,114 |
| Same-slot collisions | 0 |

The frozen completion hash was
`21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de`.
Per-dual-boot slot-pair counts were 1,472, 1,910, 1,910, 1,917, and 1,905; every boot had equal
slot use and zero collisions. Timing instrumentation was disabled, and sampled operator metrics
reported zero CUDA timing samples and zero dropped samples.

Thermal regime: N=14,126 samples at 250 ms, 28--51 C and 180--2422 MHz, with no artificial
cooldown. The machine-readable summary is `raw/box1/soak/summary.json`; the raw boot, request,
metrics, server, thermal, reducer, and integrity receipts sit beside it.

## Frozen performance curve

Protocol: N=5 interleaved serial/dual arms at every c2..17, rotated width order, 512 completion
tokens per request, and one inherited GPU-lock hold. The metric is aggregate completion tokens
after first visible token divided by the decode-window duration. All 160 points and 1,520 request
rows completed without error.

| Width | Serial median tok/s | Dual median tok/s | Delta |
|---:|---:|---:|---:|
| c2 | 130.955 | 132.454 | +1.144% |
| c3 | 146.366 | 145.927 | -0.300% |
| c4 | 155.663 | 171.308 | +10.050% |
| c5 | 161.633 | 175.689 | +8.696% |
| c6 | 165.593 | 191.976 | +15.932% |
| c7 | 168.484 | 192.134 | +14.037% |
| c8 | 168.785 | 203.813 | +20.753% |
| c9 | 158.622 | 202.030 | +27.366% |
| c10 | 161.099 | 210.271 | +30.523% |
| c11 | 163.247 | 209.012 | +28.034% |
| c12 | 165.459 | 216.271 | +30.710% |
| c13 | 166.826 | 213.791 | +28.152% |
| c14 | 168.811 | 219.543 | +30.053% |
| c15 | 169.714 | 217.014 | +27.870% |
| c16 | 170.048 | 223.534 | +31.454% |
| c17 | 163.929 | 208.652 | +27.282% |

The frozen floor is >=+15% at every c>=8 median. The minimum is c8 at +20.753%, so the floor
passes. Across the 80 dual points, operator metrics record 40,956 slot pairs, 40,956 uses of each
slot, and zero collisions.

Thermal regime: N=39,788 samples at 250 ms, 30--49 C and 180--2422 MHz, with no artificial
cooldown. The reducer output is `raw/box1/perf/summary.json`; all point summaries, request rows,
operator metrics, server logs, the continuous thermal trace, and `SHA256SUMS` are retained beside
it.

## Handoff

The branch is intentionally left unmerged and untagged for orchestrator review. The implementation
and evidence support the increment-1 HOLD posture only: arbitrary-width dual-aware scheduling and
the alternating-slot collision gate are complete, while default activation remains outside this
increment.
