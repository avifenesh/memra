# DSV4 m_e=1 tensor-core tail candidate

Date: 2026-09-06

This is a compile-only design receipt. It is not dispatched and has no GPU,
quality, or rate claim.

## Why this is a larger down lever

The real one-token matrix down route has one routed row per non-empty expert
group. The shipped `moe_kq_sktail_kernel<QT_NVFP4_MODELOPT>` still executes a
32-row x 64-column tail tile. For `m_e=1`, all 32 A rows alias the one input
row and warps 1/3 compute rows that are discarded. This candidate keeps the
same 128-thread launch and B tile, but for an explicit m_e=1 instantiation:

- warps 0/2 retain the valid row's existing `m16n8k16` f32 accumulation chain;
- warps 1/3 skip their invalid-row MMA chain;
- the second duplicate A stage (rows 16..31) is not loaded;
- FP8 mirror, macro2, output layout, and scatter stay on the existing caller.

This is tensor-core work elision, not the removed scalar m=1 visitor. The
valid row sees the same `kq_fetch`, `kq_store`, shared tile, MMA order, and
f32 output epilogue. The expected prize is the selected-expert down wall,
with a target of >5% end-to-end plain improvement before spending GPU time.

## Default-off gate seam

CUDA: `moe_kq_sktail_kernel<QT_NVFP4_MODELOPT, true>`

FFI: `memra_moe_kq_gemm_sk_m1`

`GroupedRoutes` dispatches it only when `MEMRA_F16G_M1_TC=1` (or the
gate-only `Dsv4Gpu::set_grouped_m1_tc_for_gate` override), on plain one-row
transactions with the deep tail enabled. The current composition win and
shipped tail remain the default rollback.

Direct CUDA compile receipt:

```text
nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 \
  --expt-relaxed-constexpr -c cu/moe_f16_grouped.cu
PASS (CUDA 13.1, sm120a)
```

The temporary object was removed after the compile. No GPU runtime arm was
launched yet.

Rust gate compile:

```text
cargo check -p memra-engine --bin dsv4_decode_rate_gate
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.09s
```

The object exports both `memra_moe_kq_gemm_sk_m1` and the
`moe_kq_sktail_kernel<108,true>` instantiation. The new gate mode is
`m1-tc-compose-ab`.

## Required exactness gate before dispatch

Use one process and the same route/input/weight buffers. For one-row cells at
256 and 8192, run the shipped tail and this candidate in alternating order,
then compare:

1. every pair-row down output before macro2;
2. the complete original-slot contribution after the existing scale/scatter;
3. full-layer logits and greedy token tape;
4. no invalid-row or stale-tail writes under route-validation ON and OFF.

The candidate must be bit-identical on the valid row, pass compute-sanitizer,
and show actual dispatch engagement. Only then price an interleaved rate cell;
the completed 1M capacity proof is not rerun.
