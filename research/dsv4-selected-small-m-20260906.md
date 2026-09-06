# DSV4 selected-expert small-M candidate, 2026-09-06

## Receipt-derived shape

The plain matrix Nsight capture attributes 531,546,621 ns / 8,154 launches to
`moe_kq_sktail_kernel<108>` (39.2% of summed kernel time, 65.188 us mean,
49.0--93.1 us range). This is the ModelOpt split-plane direct visitor
(`QT_NVFP4_MODELOPT = 108`), not the scalar native decode kernel. The matrix
request program routes one token with top-8 experts; after CSR compaction the
selected expert groups are `m_e=1` (empty local groups are possible on an EP
rank). The live projections are:

| projection | in_f | out_f | NVFP4 bytes per row | visitor work |
| --- | ---: | ---: | ---: | --- |
| gate/up | 4096 | 2048 | 2048 codes + 256 E4M3 scales | `SKT_BK=64`, 32x64 output tile, 31 padded rows discarded |
| down | 2048 | 4096 | 1024 codes + 128 E4M3 scales | `SKT_BK=64`, 32x64 output tile, 31 padded rows discarded |

For `m_e=1`, `moe_kq_sktail_kernel` launches 128 threads and computes the
full 32-row MMA tile. The B tile is real work, but the A load, MMA, and f32
epilogue cover 32 logical rows to write one. This is the shape defect being
attacked, not a power or topology claim.

## Adjacent proven pattern

The selected ModelOpt readers in `wt-glm5-mlagraph` use one warp per output row
(`qmatvec_nvfp4_modelopt_sel_f32`) and a measured four-row-per-warp packing
(`..._sel_f32_v3`). Their byte contract is codes `[E,out,in/2]`, ModelOpt
E4M3 scales `[E,out,in/16]`, one warp-strided reduction, and macro fold after
the row sum. Their accumulation order is explicitly a separate class from the
cuBLAS/MMA oracle and is gated by argmax/tolerance. The DSV4 candidate follows
that work mapping but keeps the DSV4 input contract: normalized f16 activation,
f16-rounded dequantized weight, f32 FMA, then the per-pair activation row-scale
fold. It does not assume B200 instructions or geometry.

## Candidate and numeric obligations

`moe_kq_sk1_kernel<QT_NVFP4_MODELOPT>` is compiled in the grouped visitor TU
and is armed only by `MEMRA_F16G_SK_SMALL=1`. It is selected only for qtype 108
and `m_e=1`; ordinary qtypes and `m_e>1` stay on the existing visitor. The
candidate is named accumulation class `fp4-sk-small-f32acc` and is **not**
bit-identical to `moe_kq_sktail_kernel`, because the candidate sums scalar f16
products in warp order instead of issuing the m16n8k16 f32 MMA chain.

The gate must prove, on the exact matrix/EP plain recipe:

1. kernel engagement on both gate/up and down (`[moe-sk-small]` lines and
   `QT=108`, `in_f=4096/out_f=2048` plus `in_f=2048/out_f=4096`);
2. the mixed-count CSR red arm (`m_e=0`, `m_e=1`, and at least one `m_e=2`)
   leaves the latter on the shipped visitor and writes no stale rows;
3. frozen real-prompt sampled output, token identity, logits/argmax policy,
   and DSpark acceptance remain within the existing matrix gate contract;
4. timing is compared as an interleaved full-model A/B, not as a component
   launch microbenchmark alone.

The default remains OFF until that gate passes. No claim is made about watts,
SM SKU peak, or a B200 result.

## Smallest next GPU cell

After the dev-pair lock is free, build one binary from this branch and run a
single-card bounded component gate first: exact ModelOpt table, one-token
routes, `m_e=1`, gate/up `(4096,2048)` and down `(2048,4096)`, 32 warmups plus
100 interleaved timed launches, candidate vs tail, output maxdiff/argmax and
sanitizer red for `m_e=2`. If it passes, run the smallest full-model cell:
the existing matrix plain recipe, 256-token prompt, 4 warmups plus 6 measured
rows per arm, vendor-default sampled decoding, output identity receipts. Do
not spend a 1M capacity run on this arm.

## B12x comparison and exact-contract verdict

The current FlashInfer/b12x direct-micro design is a useful control-plane
reference, but it is not a drop-in replacement for this DSV4 visitor. The
official direct micro kernel consumes raw top-k ids/weights, uses 16 warps
(`_BLOCK_DIM=512`), computes paired gate/up software FP4 dot products, writes
the gated intermediate, and crosses a global epoch barrier before FC2. Its
`m==1` FC2 specialization exposes narrow row-pair tasks. The same source
supports a W4A16 mode, but that mode keeps BF16 activation operands and omits
DSV4's FP8-QAT activation scales; the W4A4/A8 paths quantize/transform the
activation in the kernel under their own scale contracts.

DSV4's matrix visitor cannot reproduce that exact B12x numeric program without
changing the model contract: DSV4 `RefFp8Round` produces per-128 activation
codes/scales, the grouped path mirrors those codes to normalized f16 plus a
row scale, and the ModelOpt expert scales are E4M3/16 plus `scale_2`. The
current visitor is f16 MMA over that mirror, while B12x direct micro uses
software FP4 dot helpers and a different intermediate quantization/barrier
contract. Reusing the B12x kernel as-is is therefore a **no-go for exact DSV4**
and must not be presented as a pure kernel swap.

The viable DSV4 adaptation is narrower: retain the existing FP8-QAT input and
ModelOpt E4M3/16 scales, fuse the two direct FC1 projections and SwiGLU into
one DSV4 CUDA-core visitor, then run the existing FP8 intermediate quantizer
and FC2 visitor. That removes one FC1 launch and reuses activation/code loads,
but it is not the full B12x FC1->quant->FC2 one-pass until a persistent
cross-CTA epoch/barrier and exact per-128 quantizer are implemented. It should
be evaluated only after the `m_e=1` row-padding candidate, because the present
39.2% wall is first a small-M tile waste.

## Device-route correction

The matrix `MEMRA_DSV4_GROUPED_ROUTE=device` arm passes `ex_off_host=null` by
design. The candidate therefore has a separate device-prefix branch: it uses
the same `ex_off_dev` pointer as the existing visitors, launches `sk1` over a
persistent bounded `(n_active * out_f)` grid, and passes `total_tiles=-1` to
the regular visitors so they derive counts from their device prefix. Their
`mlo` is raised to 2 when the candidate is armed, preventing `m_e=1` duplicate
writes. The C entry admits a null host offset only for ModelOpt qtype 108;
other direct visitors retain the old host-offset ABI.
