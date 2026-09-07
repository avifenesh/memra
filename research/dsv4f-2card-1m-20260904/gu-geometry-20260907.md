# One-token GU geometry component, 2026-09-07

The current `memra_moe_kq_gemm_sk_gu_half2` launcher is the real control.
The fixture uses a 128-expert sparse bank, hidden 4096, intermediate 2048,
and 1/3/4/6 active groups. `SK_BN=64`; a deployable one-token top-6 bound is
192 tile slots. This cap uses no host read of the actual device active count;
`total_tiles=-1` keeps the device prefix authoritative.

Two comparisons each use current/candidate/candidate/current repeated three
times. Every scored launch follows a twice-L2 flush. All concrete kernels
are warmed and checked before scoring. Every row checks finite H, bit
identity, guards and repeatability; each comparison proves six real control
FFI submissions. Both cards pass memcheck. A downstream f32-to-half/down
calculation is diagnostic only, not the production FP8-QAT transport.

| Active | GPU 0 current / capped us | GPU 0 current / M1+half2 us | GPU 1 current / M1+half2 us |
|---|---:|---:|---:|
| 1 | 112.309 / 112.544 | 112.560 / 107.851 | 114.379 / 107.963 |
| 3 | 126.731 / 125.595 | 126.656 / 120.549 | 128.272 / 121.168 |
| 4 | 135.435 / 133.248 | 134.965 / 128.256 | 136.096 / 130.155 |
| 6 | 162.981 / 162.000 | 162.469 / 152.720 | 164.288 / 153.477 |

The grid-only cap is approximately flat. Combining the existing M1 and
half2 template capabilities is bit-exact on H and 1.044-1.070 times faster
on this component. It warrants a full-model composition gate; it is not a
model throughput gain or production/default decision yet.

R7 source SHA256 `fd34be674a4a05234960c091be60417ebea09318df2b1950610c515b47bd0cfb`;
binary `e1283a29574ca4a289240fd5c4500daad8008be5129d5139672ca0e01f0f98e3`.
Included current CUDA source SHA256
`4cb4539cf3bee7260c8f85082e3fb8cfddcb34240f191d2de4d870989eb8e6f5`.
Compiled with CUDA 13.1, C++17, `-O3`, sm_120a, matching the production
MoE TU without a `-fmad=false` override. Raw namespaces:
`gu-geometry-{memcheck,rate}-20260907-r7`.

R6 stopped at a harness refusal after active-group-1 H identity: diagnostic
down passed `row_bytes=HIDDEN/2` for `in_f=INTER`. The required value is
`INTER/2`; the wrapper correctly returned 40004 before launching. R7 fixes
only that argument. R6 sanitizer timings are not performance evidence and
the incomplete campaign is not a qualified cell.
