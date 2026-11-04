# GLM-5.3-Flash B200 doors decided after the box retired (2026-11-04)

The 2x B200 pair that carried the GLM-5.3-Flash TP-2 lanes is retired, so no new cell can run
on it. Owner, 2026-11-04: decide every B200 door from the receipts already on record. A winner
with a positive, correct receipt becomes a naked default on sm_100a builds, keyed on
`BUILT_CUDA_ARCH` as the existing per-device defaults are (`MEMRA_Q8_ROW_ILP` is the
precedent), and its switch goes. A door whose receipt is negative, flat or missing is deleted
with everything only it reaches. One-rig evidence sets a one-rig default: no other build changes.
The deleted doors are in the Removed doors ledger of `docs/FLAGS.md`; this record covers the
defaults that changed and the calls left open.

## The TP split grouped prime and its host diet: sm_100a default

**Chosen:** on sm_100a builds the expert-split TP walk's prefill chunks always take the grouped
prime (both ranks run the EP grouped prime's per-rank program over all routed pairs against the
half-width slabs, the root adds the peer's partial), with the host diet inside it (the expert
CSR built once per layer-call by a counting sort and uploaded through a pinned arena, the shared
expert launched before the routing table is built). `MEMRA_GLM5_TP_SPLIT_GROUPED_PRIME` and
`MEMRA_GLM5_TP_SPLIT_PRIME_HOSTDIET` are deleted.

**Rejected:** the undieted grouped prime (a CSR built per rank, pageable uploads), measured
slower below; the host diet's root-first arm (`=2`), 9.6% slower at 256k (80.65 against 73.56 s,
byte-identical, tpwalk27), because the peer's pass then starts after root's launches and the
combine waits on it; and keeping both as launcher-set doors, since the launcher's box is
retired.

**Measurement that settled it:** the host diet's pair receipt (darklanes
`research/glm5-b200-mint-20260904/LANE.md:3612-3617`, tpwalk27/28 and box2hd, memra #322):
256k prime 74.27 -> 59.89 s (-19.4%) and 1M prime 310.1 -> 251.1 s (-19.0%) with the grouped
prime armed in both arms, decode 65-67 tok/s, tapes and ids identical to the door-off arm. The
grouped prime's own need: without it the split walk primed at about 520 tok/s flat (3,766 tokens
in 7.08 s, 29,961 tokens in 57.7 s, tpprime2 2026-09-07); with it the 256k prime took 74.27 s.
That comparison is across windows, not an interleaved A/B, and the owner promoted the pair on
these rows.

**Scope:** sm_100a builds only. On every other build the split walk keeps its per-token slot
walk at prefill. The walk itself still needs `MEMRA_GLM5_TP_EXPERT_SPLIT=1`, which is an open
call (below).

## The warp-merged top-k twin: sm_100a default

**Chosen:** on sm_100a builds `Engine::topk_rows` (the DFlash2 candidate selector) runs the
warp-merged exact twin pair (`topk_rows_wshard_k{8,16,32}_f32` + `topk_rows_wshard_merge_f32`)
for k in {8, 16, 32}, ahead of the shard split. `MEMRA_TOPK_WARP` is deleted.

**Rejected:** the shipped arms on sm_100a (lane lists in local memory, one-thread merges), and
waiting for a `topkab` cell on the TP-2 walk, which cannot run: the box is retired and the TP-2
walk serves no spec route.

**Measurement that settled it:** 2x B200 pair, GLM-5.3-Flash, `tpwalk24` (2026-09-07, PP-2 spec
route at 32k, three interleaved boots per arm): off 63.40 / 65.21 / 66.65 tok/s against
68.02 / 68.07 / 68.09, +4.4%, ranges apart; `doorscreen` one boot +1.12%
(`FLAGS-HISTORY.md#memra_topk_warp`). Output-identical by construction (a discrete selection
under a total order), gated bitwise.

**Scope:** sm_100a builds only. The selector serves every DFlash2 family, so a default on the
RTX 5090 or RTX PRO 6000 needs the same cell on that card.

## The k-pool selector's v2 body: sm_100a default

**Chosen:** on sm_100a builds the single-CTA DSA k-pool selector runs the v2 body (warp-scan
rank walk, per-warp coalesced membership count and emit; same key, ranks and emit order).
`MEMRA_KPOOL_SELECT_V2` is deleted.

**Rejected:** the v1 body on sm_100a, and a `selab2` rerun with raw output banked, which cannot
run on a retired box.

**Measurement that settled it:** `selab2` on the 2x B200 pair (256k plain, three interleaved
boots per arm, 12 rows each, same tape): off 63.71 / 63.42 / 62.45 against v2 63.86 / 64.31 /
64.89 tok/s, +1.40% on the median, every v2 boot above every off boot. `tpwalk14` (TP-2 decode at
128k, two boots per arm) read about +1% with ids identical; `tpwalk24` (PP-2 plain at 32k) read
flat with overlapping ranges, at a context where the selector is a smaller share of the token.
Prose rows in darklanes `research/glm5-b200-mint-20260904/LANE.md`; the raw outputs are not
banked. Bit-identical by construction, gated over 14 cells with ties, -inf and NaN pools.

**Scope:** sm_100a builds only; the v1 body stays the selector elsewhere.

## Open calls

### `MEMRA_GLM5_TP_EXPERT_SPLIT`: keep, promote or delete

Its 2026-09-20 decide-by passed with no split-against-whole-expert row in this repo, and the
launcher that set it is retired, so door hygiene would delete it. Deleting it deletes the
grouped prime and host diet above, which run only inside the split walk. Options: keep it as
the TP-2 expert-placement switch next to `MEMRA_GLM5_TP` (recommended: the owner promoted the
prime that needs it, and the two placements are different numeric classes an operator picks);
make the split the sm_100a TP default on the receipts its prime rows carry (no A/B against
whole-expert EP exists); or delete it with the grouped prime. Decide-by 2026-11-18.
