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
- `kv_len`: the full-attention length at `b` (each trunk layer appended every row, so
  `len - (n - b)`; a layer the trunk walk does not advance, such as the MTP block's plane, keeps
  its length);
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
   `solve32`) is computed per GDN chunk (`gdn_chunk_size()`, 32 rows) and does not see the split.
3. The tap's conv ring and SSM state are the values the split prime's first call left in the cache
   at `b`, by the same law.

Kept on the split (`HybridModel::prime_tail_fold_ok`): a stop on a call boundary (the split
already ends a call there, so a fold gains nothing), a stop with fewer than `PRIME_MIN_T` rows of its
call on either side, MoE trunks (their small-row expert program differs from the grouped one, so
the split law is not established for them), an explicit `MEMRA_PRIME_CHUNK` (operator ranges off
the grid), the gemma, hyper, sliding-gated, ring and pipeline-parallel programs, warm and restored
primes, and the dspark path. A tap a GDN route did not fill (a padded prime, a captured prime
graph) publishes no capture; the primed rows are the same either way. The tapped call runs the unfused out-norm (`gdn_scan_prefill` then
`gated_rmsnorm_f8f4v2`, bit-identical to the K4+K5 out-norm kernel by lever 4's gate) because the
fused kernel writes its activation's k-major scale plane by absolute token column.

## Expected gain

The tail call's time minus the folded rows' share and the tapped call's extra cost (48 GDN layers
without the fused out-norm at about 0.9 ms each, 48 SSM state copies of 3 MB, 96 extra launches):

| prompt tokens | tail call | fold cost | net | of TTFT (main) |
|---:|---:|---:|---:|---:|
| 32,784 (stop 32,768) | 0.257 s | | 0 | 0 |
| 131,051 (stop 131,008) | 0.460 s | ~0.10 s | ~0.36 s | ~0.3% |
| 260,000 | ~0.68 s | ~0.13 s | ~0.55 s | ~0.18% |

The 32k bench prompt's stop is 8 x 4,096, a call boundary, so it keeps the split. A stop that is the
last turn marker rounded down to the 32-row grid lands on a 4,096-row boundary about once in 128
prompts.

## Gates

- `prime-fold-gate` on the 27B: prime one prompt split at `b` and folded, for the served 131k shape
  (4,032 rows into a 4,075-row last call, 43 after), the stop between two 64-row tiles, 32 rows into
  a call, a 16-row tail, one call, and a middle call whose tail crosses into the next. Compare every
  hidden row, the final logits, the logits at `b - 1`, the KV bytes of every layer's rows, the
  capture's conv and SSM bytes, `kv_len` and `pos`, and the final recurrent state; and check that a
  stop on a call boundary is refused.
- Served A/B B T T B B T at 131k and 260k with the text fingerprints and the per-call trace; the
  signal is the trunk's end calls (TTFT alone cannot resolve 0.2-0.5 s at N=3). A restore from the
  capture reads only the fields the gate compares (conv, SSM, KV rows below `pos`, logits, last
  hidden), so the gate's equality makes it the split's restore.
- The device battery and serve-smoke in the cell records.

## Measured (ledger rows 67-67c, `tail-fold/`)

`prime-fold-gate` is bit-identical on GB10 unit 2 and on the RTX 5090 in all seven folded cases, and
refuses the call-boundary stop. Its first run caught the capture refusing on the MTP block's KV
plane, which the trunk never advances (`snapshot_from_tap` now keeps a layer's length when it is not
the cache's position).

Served on unit 2, B = main `65fb297a59`, T = the fold, 3 per arm: the trunk's end calls go from
5.283 to 5.055 s at 131,051 tokens (-0.228 s; spreads 0.011 / 0.006 s) and from 4.933 to 4.422 s at
259,951 (-0.511 s; spreads 0.005 / 0.003 s). TTFT moves -0.20% and -0.15%, inside its per-run spread.
Per call (B / T medians): at 131k the split's 4,032-row boundary call takes 4.841 s and its 43-row
tail call 0.446 s, the folded 4,075-row call 5.055 s; at 260k 4.238 s and 0.695 s against 4.422 s.
The folded call is 0.21 / 0.18 s longer than the split's boundary call: its extra rows (about 0.05 s
at 131k), the unfused out-norm in 48 layers, the copy of each part's scan output into one buffer and
the 48 state copies. Lever 8b removed the norm and the copies where the call takes the fused
route (rows 68-68d): the fused kernel writes each part's rows of one activation, and the 131k end call
fell another 0.090 s on unit 2 and 0.080 s on unit 1 (the 260k folded call, 1,903 rows, is under the
F8F4 v2 route's 2,048 rows and keeps the unfused form).

Lever 8 on unit 1 confirms unit 2: -0.235 s (spreads 0.022 / 0.016 s) and -0.514 s (0.010 / 0.007 s),
TTFT -0.28% / -0.36% with T under B in every cell.
