# Ornith's numbers depend on where its MoE experts are placed (RTX 5090 Laptop GPU)

Ornith-1.5-35B-A3B NVFP4+Q5_K MTP keeps its experts on the card only when the free memory at load
covers the experts, the trunk and a 2.0e9-byte reserve (experts 18.57 GB + trunk 1.60 GB on this
card, 22.17 GB in all against 24.52 GB free on an idle card). Otherwise the SLRU cache serves
them. `MEMRA_MOE_RESIDENT=0` forces the SLRU placement, which is what a card with another tenant
gets. Main v0.141.0 (3de59bbf80), one card, one lease per cell.

## The gate's 12-row window

`window-eager.md` and `window-batch.md` (from `../repeats/rtx5090-laptop/rep-or-*` and
`rep-or-*-slru*`, `slru_compare.py`):

- Decode is byte-identical between the placements, in eager decode and in the served batch
  program: every row's top-1 and margin match exactly.
- The prefill reference differs at all 12 rows. Prefill margins move by up to 0.562 logits (row
  2,039: 0.947 resident, 0.385 SLRU). No top-1 changes.
- The floor rule passes both placements.
- Two SLRU runs give the same table (`76dce3c92cf0ef8f`), so the SLRU placement is a second
  stable program, not noise.

## A served greedy request

`served-pair/`: the same memra-server binary (memra-0.141.0-d792eacd8d45, sm_120a) in two boots.
The first runs the default placement (experts resident), the second runs `MEMRA_MOE_RESIDENT=0`.
Each gets one `/v1/completions` request with the `board-2048` prompt (2,048 tokens), temperature
0, 128 tokens, on the default route (the MTP draft head, speculative decoding).

| placement | completion | spec acceptance | elapsed | text sha256 (first 16) |
|---|---|---|---|---|
| resident | 128 tokens, 340 characters | 54 of 148 drafted | 1.72 s | `1408f53ea2349977` |
| SLRU | 128 tokens, 345 characters | 53 of 148 drafted | 6.51 s | `a5f81b0cc03207a1` |

The texts agree for the first 95 characters, then differ. After "`bytemuck`** ", the resident run
continues "POD casts + **`raw-cuda`** FFI", the SLRU run "(zero-copy views) + **`tracing`**
(structured logs)". So a user's greedy output from the same server, model and prompt depends on
how much memory was free when the model loaded.

## Which computation differs (reading of the code and logs, not yet localized by a run)

- **Prefill (more than 16 tokens).** With the experts resident, the MoE layer takes the pairs
  route (`moe_ffn_pairs`, gated on `dev_exps`) and, for these quant types, the grouped f16 lane:
  the active experts are dequantized to f16 once per projection and run as one grouped GEMM with
  f32 accumulation (`cu/moe_f16_grouped.cu`, the f16-mirror numeric class, which that file
  notes is not byte-identical to the MMQ path). The resident boot logs the lane's `[moe-sk-form]`
  line; the SLRU boot does not. With no `dev_exps`, prefill falls through to the per-expert path
  that reads experts through the SLRU cache slots, with another dequant and accumulation order.
- **Decode (one token).** Both placements run the same per-expert kernels on the same bytes; only
  the pointer provenance differs (slab against cache slot), the bit-identity class
  `moe_ffn_dev` documents. That matches the byte-identical decode columns.
