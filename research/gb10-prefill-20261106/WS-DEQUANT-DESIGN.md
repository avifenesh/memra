# The prime's KV dequant workspace pass on GB10 (round 4 design, 2026-11-08)

Design note, then the probe. Follows `MTP-FILL-DESIGN.md` and ledger rows 46-51, which land with
memra#1108 (`perf/gb10-mtp-fill-batched`); this lever depends on that PR's round-4 receipts.

## The part

Every prime attention layer-chunk first dequantizes the whole KV prefix `[0, T_kv)` (q8_0 K,
q5_1 V; 1,856 B per token per layer) into a BF16 workspace (4 KB per token) that the h2p kernel
reads (`fa_dequant_kv_ws_bf16`). The prefix is rewritten every chunk, so the pass grows with T^2:
1.371 s at 131,051 tokens (528 calls x 2.60 ms, `profile-r4-main-131k/parts-req.txt`) and, by the
T^2 growth (not profiled), about 5 s at 259,951.

At 131k one call moves about 390 MB at the average depth (65.5k tokens x 5.95 KB) in 2.60 ms,
about 150 GB/s, against the device's 215 GB/s copy rate (`ceilings/`). The kernel runs one thread
per element: two 64-bit divisions for its token and element, the block's f16 scale (and for V the
min and the high-bit word) loaded per element, and a 2-byte store.

## The change

With both widths multiples of 8 (1,024 here), each thread converts eight consecutive elements of
one token row and writes them with one 16-byte store; the index math is 32-bit when the group count
fits (it does at every served length). The element loads stay per element (the SASS shows the
q5_1 header read per element as before); the gain is the store width, an eighth of the threads
and the index math. Each element is still `__float2bfloat16(DQ_*_ELEM(...))`
with the same arguments, so the workspace bytes cannot change; the grids are sized by the new
work-item count. Widths that are not multiples of 8 keep the per-element loop. Both entries
(`fa_dequant_kv_ws_bf16` and the `_prime_table` twin) share the body. Code:
`perf/gb10-ws-dequant8`.

Gates: `attn-r4/probe/probe_ws.cu` compiles the tree's kernels and times the engine's
`fa_dequant_kv_ws_bf16` and its `_prime_table` twin (with the engine's fixed `sm_count * 8` grid)
against main's per-element loop, copied verbatim as `ref_ws1`, comparing every bf16 of both
workspaces, in the default (q8_0 / q5_1) and the kf8vf8 (e4m3) formats at several widths.
kernel-check's ARC B cells compare the attention output of the workspace path with the
inline-dequant kernel bit for bit, for one shape (head dim 256, 4 KV heads) in the default fatbin
through the eager entry; they do not compare the workspace itself.

Expected at the copy rate: 2.60 -> about 1.8 ms per call at 131k, about 0.4 s of TTFT (0.35%); at
260k about 1.5 s (0.5%).

## Also designed, not built: the f32 norm output before a GDN mixer

The fused residual + norm kernel before each layer's mixer writes the f32 norm output `h` as well
as the seeded F8F4 v2 activation (and, before a GDN mixer, the q8_1 pair for beta and alpha). When
every weight of the mixer's input group takes a seed (`wqkv` and `wqkv_gate` on the v2 tile, beta
and alpha on the q8_1 pair: true for every 4,096-row chunk), no prime reader needs `h`, and the
84 MB write per call (0.39 ms at the copy rate) is waste: 1,536 GDN layer-chunks at 131k, about
0.6 s. Before the norm may skip it, each consumer that would fall back to reading `h` (an
unseeded group member, the f16 operand, the q8_1 quantize) must fail closed on an unwritten input,
as `matmul_prefill_ffn_f8f4v2` does with `z_written`.

## First probe (unit 2, `attn-r4/probe-runs/probe-ws-unit2.log`)

That run compiled main's `flash_attn.cu` (sha256 `e759f4ac`) with a hand copy of the new body
(`cand_ws8`: the same loop without the 2^31 guard and the fallback); the probe was then rewritten
to compile the tree's own kernels (above). 5 timed launches after a warmup, two passes, every bf16
of the candidate's K and V workspaces compared with the shipped kernel's:

| T_kv | shipped | eight per thread | change | mismatching bf16 |
|---:|---:|---:|---:|---:|
| 32,768 | 1.098 / 1.099 ms (177.6 GB/s) | 0.898 / 0.899 ms (217 GB/s) | -18.2% | 0 |
| 65,536 | 2.180 / 2.181 ms (178.9 GB/s) | 1.793 / 1.797 ms (217 GB/s) | -17.7% | 0 |
| 131,072 | 4.346 / 4.343 ms (179.6 GB/s) | 3.559 / 3.701 ms (211-219 GB/s) | -16.4% | 0 |
| 262,144 | 8.020 / 8.457 ms (184-195 GB/s) | 7.045 / 7.192 ms (217-222 GB/s) | -13.6% | 0 |

The candidate runs at the copy rate. In isolation the shipped kernel already ran at 178-195 GB/s
(150 GB/s in the served profile; the cause of that difference is not measured), so the expected
served gain is about 16% of the part: about 0.2-0.3 s at 131k and 0.7 s at 260k (about 0.2% of TTFT).
