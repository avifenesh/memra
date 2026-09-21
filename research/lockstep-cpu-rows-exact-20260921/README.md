# Lockstep CPU experts: exact cross-stream program (memra#577 follow-up)

Verdict: the companion's multi-row arm is back on by default and a stream's bytes no longer
depend on its peers. The rows kernel already produced per-row bit-identical down projections;
what broke exactness was the fold. `memra_cpu_moe_token_v2` folds each expert as
`sum = fma(y, route_weight * down_scale, sum)` in job order from zero, while the scaled rows
twin returned `y * down_scale * w` and the Rust side added tickets in expert order. The new raw
twin (`memra_cpu_expert_rows_raw_v2`) returns bare `y`, and `cpu_experts::accumulate_expert_exact`
re-folds each row in its own selection order with the one-job arithmetic (`f32::mul_add`, one
f32 scale product, zero start). Exact by construction; proved bitwise on CPU and on Hy3.

## Proof 1: CPU, `cpu_native_check` (runs in ci.yml `engine-tests`)

New arm "multi-row raw + exact accumulate == one-job program": a 3-expert one-job call with
route weights `0.37, -0.61, 0.9` and down scales `0.7, 0.3, 0.9` (non-powers of two on
purpose) against raw rows plus the exact fold, bitwise, for Q2_K, IQ3_S, Q4_K and NVFP4; plus
row 0 of an `m_r = 2` raw call against the `m_r = 1` call (amortization changes no bit).
PASS on the local rig (`raw/cpu-native-check-local.log`) and on the box. The pre-existing
scaled-rows identity arm only ever coincided because its weights (`0.5, -0.25, 0.125`) are
powers of two, which is also what hid the divergence in the lockstep harness until #577.

## Proof 2: Hy3 on a rented RTX PRO 6000 (`raw/abrows/`)

Rig: one rented RTX PRO 6000 Blackwell WS (97887 MiB), AMD EPYC 9B14, 188 GB container RAM,
direct-io disk 2.7 GB/s write and 5.1 GB/s read, 2026-09-21. (A first box with 125 GB RAM
thrashed in disk wait on the first cell and was destroyed; see the acceptance note in the
memory corpus.) `Tiyuvta/Hy3-NVFP4@0af425172b7a`, SHA256SUMS verified (README-only mismatch),
frozen residency, native companion built from this lane, greedy, 32 new tokens, stream 0 = P0.
base = origin/main `9ef2f04d6` (one job per row, #587's default), lane = this branch.
`cpu_native_check` on the box: the raw + exact accumulate arm PASS for all four formats
(`raw/cpu-native-check-box.log`).

| Cell (`raw/abrows/`) | Logits vs base M=1 |
| --- | --- |
| lane M=1 | IDENTICAL, 33 steps |
| lane M=2 mixed | IDENTICAL |
| lane M=3 mixed | IDENTICAL |
| lane M=4 mixed | IDENTICAL |
| lane M=4 same prompt | IDENTICAL |
| lane M=4 mixed, `MEMRA_LOCKSTEP_CPU_ROWS=0` (one job per row) | IDENTICAL |
| base M=4 mixed | IDENTICAL (main's default program, as #587 left it) |

Every M>1 cell reproduces the M=1 bytes with the multi-row arm on: the exact fold holds with
real routing, real sharing across streams, and the companion's amortized decode.

## Cost

THROUGHPUT

## Doors

`MEMRA_LOCKSTEP_CPU_ROWS=0` is now the rollback seam to the one-job-per-row program (also
exact, no cross-stream amortization); remove by 2026-10-05 if unused. The scaled
`memra_cpu_expert_rows_v2` stays exported for ABI compatibility; memra no longer calls it from
the lockstep path.
