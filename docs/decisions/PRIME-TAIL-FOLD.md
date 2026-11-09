# The boundary tail call folds into the trunk walk (2026-11-09)

**Status:** decided for dense GDN trunks on every device the gate ran (GB10 sm_121a, RTX 5090
sm_120a). Served evidence on GB10 (both DGX Spark units). No env seam: the fold is the only path
where it applies, and `prime-fold-gate` builds its split reference from two prime calls.

## Question

A cold prime with a prefix-cache capture or a turn checkpoint at the render-stable boundary `b` ran
as two prime calls, so the GDN conv ring and SSM state could be snapshotted where the first ended.
The second call primed the 16 to 47 rows after `b` and paid a whole pass of fixed per-call costs: on
GB10 at 131k, 0.446 s for 43 rows (the round's profile: attention over the whole prefix with one CTA
per head, the KV dequant of the whole prefix, every weight read once at T 43). Can the capture be
taken inside one walk, with the same bytes?

## Decision

The walk runs unsegmented. The call that contains `b` runs each GDN layer's conv prep and chunked
scan in two parts at `b` and copies the conv ring and SSM state there into a tap
(`Cache::gdn_tap`). The capture is `Cache::snapshot_from_tap` (the tapped states, each trunk layer's
KV length less the rows after `b`, `pos = b`) with row `b - 1`'s logits through the final norm and
LM head (`HybridModel::prime_row_logits`). Both prime drivers take it: the MTP prime walker (served)
and the cold spec prime. `HybridModel::prime_tail_fold_ok` keeps the split for a stop on a call
boundary (the split already ends a call there), fewer than `PRIME_MIN_T` rows of the call on either
side, MoE trunks, an explicit `MEMRA_PRIME_CHUNK`, the gemma, hyper, sliding-gated, ring and
pipeline programs, and warm or restored primes.

Exact by the prime split law on the GDN chunk grid: the folded walk is the monolithic prime the
split already equals, and the two-part scan is that law inside one layer.

## Evidence

- `prime-fold-gate` (Qwen3.8-27B NVFP4+Q5_K, `research/gb10-prefill-20261106/tail-fold/`): seven
  folded cases bit-identical to the split on GB10 and on the RTX 5090 (every hidden row, the final and
  boundary logits, the snapshot, the final GDN state, the K/V bytes); the call-boundary stop refused.
- Served on GB10, B = main, T = the fold, B T T B B T, 3 per arm, the per-call trace (ledger rows
  67b-67c): the trunk's end calls (B: the boundary call plus the tail call; T: the folded call) fall
  by 0.228 s at 131,051 tokens and 0.511 s at 259,951 on unit 2 and by 0.235 s and 0.514 s on unit 1,
  against end-call spreads of at most 0.022 s. TTFT moves -0.20% / -0.15% on unit 2 (inside its
  per-run spread) and -0.28% / -0.36% on unit 1 (T under B in every cell): no regression. Identical
  answers; every request published its capture.

## Follow-up: the tapped call keeps the fused out-norm (lever 8b, ledger rows 68-68d)

The first form ran the tapped call's out-norm unfused and copied each part's scan output into one
buffer: about 47 ms and 40 ms of the folded 131k call on GB10. Now each part's fused K4+K5 pass writes
its rows of the call's one F8F4 activation (`gdn_chunk_state_output_norm_mma` takes the activation row
its first scanned row lands at and the activation's row count; a whole call passes 0 and T, at 128
registers and the same spill). Exact: GDN-K45-NORM with a two-part run, `prime-fold-gate`. The 131k
end call falls 0.090 s on unit 2 and 0.080 s on unit 1 (spreads under 0.01 s); the trunk's other
calls are not slower. A folded call under 2,048 rows (the 260k bench prompt's) keeps the unfused
form, as the F8F4 v2 route does.

## Rejected

- Extending the last call by the tail when the stop is on a call boundary (the 32k bench prompt):
  a call wider than 4,096 rows would re-size the resident prime slab and its admission charge.
- Folding MoE trunks: a short tail runs the per-token expert program, which is not established as
  bit-identical to the grouped one, so the split law does not cover it there.
- An env seam: the gate does not need one.
