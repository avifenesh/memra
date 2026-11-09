# Folding the boundary tail call into the trunk (GB10 prefill, round 4, lever 8)

## What the tail call is

A cold served prompt with the prefix cache on is primed in two prime calls. The first ends at the
render-stable boundary `b` (the last turn-marker token, rounded down to the GDN chunk grid, with at
least `PRIME_MIN_T` = 16 rows left after it; `plain_checkpoint_boundary`,
`grid_align_boundary_within`). There the spec prime takes the prefix-cache capture and the turn
checkpoint: a snapshot of every GDN layer's conv ring and SSM state, the full-attention KV length,
the logits of row `b - 1` and its hidden (`spec.rs`, the cold prime's stop loop). The second call
primes the last `n - b` rows. The bench prompts have 16 rows after `b` at 32k and 43 at 131k
(`[prefix-cache] stable-boundary capture armed at 131008 of 131051 prompt tokens` on main
`65fb297a59`).

The split exists only for that snapshot: the GDN states are advanced in place by the next rows, so
the call boundary is the only moment they exist.

## What it costs

Timestamped server log at the round's start (`ts-probe-main-unit1/`): the tail call takes 0.257 s
at 32,784 tokens and 0.460 s at 131,051. Its kernels at 131k, from the round's first Nsight Systems
profile (`profile-r4-main-131k/`; the raw report is in the evidence backup, snapshot `33ee8f00`),
split at the first draft-fill kernel after the call's last attention launch:

| kernel | launches | s |
|---|---:|---:|
| `fa_prefill_qw_db` (43 query rows against 131k keys, one CTA per head) | 16 | 0.163 |
| `mul_mat_q_nvfp4_f8f4` (v1 F8F4 at T 43: every weight read once) | 400 | 0.091 |
| `fa_dequant_kv_ws_bf16` (the whole prefix, once per attention layer) | 16 | 0.075 |
| the rest (matvecs, norms, GDN, quantize) | ~1,100 | 0.054 |

The same 43 rows inside a 4,096-row call cost about 1% of that call's GEMM time and their share of
h2p's key walk: about 0.04-0.05 s at 131k.

## The lever

One prime over the whole prompt. The call that contains `b` (at local row `r = b - c`, a multiple of
the GDN chunk size because `b` and the 4,096-row call start `c` both are) runs each GDN layer's conv
prep and chunked scan in two parts, rows `[0, r)` and `[r, t)`, and between them copies the conv ring
and the SSM state into a tap. Every other operation of the call (GEMMs, attention, norms, the KV
append) runs over the whole call as before. After the prime the capture is built from the tap:

- `conv`, `ssm`: the tap's copies, per GDN layer;
- `kv_len`: the full-attention length at `b` (each layer appended every row, so `len - (n - b)`);
- `pos`: `b`;
- `logits`: row `b - 1`'s trunk hidden through the final norm and LM head, the path a call ending at
  `b` takes for its last row;
- `last_h`: row `b - 1` of the prompt's hidden stack, as today.

## Why it is exact

The prime split law (`spec.rs`, `dflash.rs`, `worker.rs`: a prime split at `L` with
`L % gdn_chunk_size() == 0` is bit-identical to the monolithic prime) covers both halves:

1. The folded prime is the monolithic prime over `[c, n)`, so its rows, logits and KV equal the
   split prime's, whose tail call ran batched (the tail is at least `PRIME_MIN_T` rows, so it never
   takes the tokenwise arm).
2. Inside the call, the two-part conv and scan are the same law applied to one layer: the conv of
   rows `[r, t)` reads the ring that rows `[0, r)` left, and the scan of rows `[r, t)` starts from
   the state rows `[0, r)` produced. Every chunk-local GDN quantity (cumulative gate, `attn32`,
   `solve32`) is computed per 64-row chunk and does not see the split.
3. The tap's conv ring and SSM state are the values the split prime's first call left in the cache
   at `b`, by the same law.

Kept on the split (no fold): padded or batched primes, a call the prime graph would capture, the
pipeline-parallel and hyper walks, the dspark path, the restored-session suffix feed, and any GDN
route the tap does not cover. The tapped call runs the unfused out-norm (`gdn_scan_prefill` then
`gated_rmsnorm_f8f4v2`, bit-identical to the K4+K5 out-norm kernel by lever 4's gate) because the
fused kernel writes its activation's k-major scale plane by absolute token column.

## Expected gain

The tail call's time minus the folded rows' share and the tapped call's extra cost (48 GDN layers
without the fused out-norm at about 0.9 ms each, 48 SSM state copies of 3 MB, 96 extra launches):

| prompt tokens | tail call | fold cost | net | of TTFT (main) |
|---:|---:|---:|---:|---:|
| 32,768 | 0.257 s | ~0.06 s | ~0.20 s | ~0.9% |
| 131,072 | 0.460 s | ~0.10 s | ~0.36 s | ~0.3% |
| 260,000 | ~0.68 s | ~0.13 s | ~0.55 s | ~0.18% |

## Gates

- A bit-compare cell on the 27B (kernel-check): prime one prompt split at `b` and folded, for `b` in
  the last call, `b` on a 4,096-row call boundary, and tails of 16 and 43 rows. Compare every hidden
  row, the final logits, the logits at `b - 1`, the KV bytes of rows `[0, n)`, the capture's conv
  and SSM bytes, and its `kv_len` and `pos`.
- Served A/B B T T B B T at 32k / 131k / 260k with the text fingerprints, plus the prefix-cache
  restore: a second turn that extends the first, restored from each arm's capture, gives the same
  answer.
- The device battery and serve-smoke in the cell records.
