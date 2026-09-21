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

BOX_TABLE

## Cost

THROUGHPUT

## Doors

`MEMRA_LOCKSTEP_CPU_ROWS=0` is now the rollback seam to the one-job-per-row program (also
exact, no cross-stream amortization); remove by 2026-10-05 if unused. The scaled
`memra_cpu_expert_rows_v2` stays exported for ABI compatibility; memra no longer calls it from
the lockstep path.
