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
walk at prefill. The walk is the expert split, which is the sm_100a TP placement default (next
section), so a naked TP-2 command on sm_100a runs this prime.

## The expert split: sm_100a TP placement default

**Chosen:** on sm_100a builds an armed `MEMRA_GLM5_TP` walk shards every routed expert across
the ranks (each rank holds half of every expert's `gate`/`up` rows and the matching `down`
columns, and MLA `wo` turns row-parallel with it) instead of assigning whole experts.
`MEMRA_GLM5_TP_EXPERT_SPLIT` stays as the placement selector, `=1` the split and `=0` whole
experts on any build: the one-card rig gate (sm_120a) reaches the split only through `=1` (arm
S), and the gate's whole-expert arms pin `=0`. A placement map places whole experts, so a map
together with the split refuses at preflight.

**Rejected:** whole-expert EP as the sm_100a default, slower below; deleting the variable, which
would leave the split, the default program of a retired card, with no gate that can run on any
card in hand; and keeping it default OFF, which kept the promoted prime above out of every naked
command.

**Measurement that settled it:** 2x B200 pair, tpwalk2 (darklanes
`research/glm5-b200-mint-20260904/LANE.md`, 2026-09-06, same window, two reps per arm): the
split decoded at 34.84 / 39.43 tok/s against whole-expert EP's 20.29 / 20.32, with the graphed
PP-2 control at 59.81 / 59.91. The split is a different numeric class (tape `f6b309777546b67e`
against whole-expert's `2fad380bb459ff0a`): `down` and MLA `wo` split their K reduction two ways,
gated on a logits band plus tape identity and never on bits (`glm5-tp-gate` arm S).

**Scope:** sm_100a builds only; every other build keeps whole experts, since no receipt exists
there.

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

## The attention-norm q8_1 fuse: sm_100a default

**Chosen:** on sm_100a builds the attention-input norm of a plain (unsharded) KDA layer in the
glm5_next T=1 walk emits its q8_1 view (`rms_norm_zq8_f32`) and the fused six-projection
launcher consumes it instead of quantizing again. `MEMRA_GLM5_Q8_FUSE_ATTN` is deleted.

**Rejected:** deleting it as marginal. memra#886 judged +0.43% marginal as a lever on its own,
but the arms' ranges do not overlap, which is the test that confirmed the k-pool selector v2
above, and the program is bit-identical, so the promotion carries no correctness cost.

**Measurement that settled it:** 2x B200 pair, c1, three boots per arm: base 74.77 / 75.17 /
75.19 against fused 75.49 / 75.76 / 75.32 tok/s, +0.43%, every fused boot above every base boot,
tape `9437b599f6b9d2a9` on all six (darklanes verdicts ledger
`glm5-b200-attn-norm-zq8-plus-0-43`, memra PR #177).

**Scope:** sm_100a builds, plain KDA layers only. TP-sharded KDA layers and MLA layers keep the
plain norm, so TP-2 loads are unchanged.

## TP prefix entries: sm_100a default

**Chosen:** on sm_100a builds the server captures and restores every rank's state in the
prefix entries of a glm5 TP-2 session (each rank's KDA shard, each peer's latent plane, on their
own devices). `MEMRA_GLM5_TP_PREFIX` is deleted.

**Rejected:** refusing TP prefix capture by default on sm_100a, which was the door-off state.
On other builds the refusal stays: no receipt exists there.

**Measurement that settled it:** 2x B200 pair, 128k prompt cold and then again (darklanes
`research/glm5-b200-mint-20260904/LANE.md:3619-3623`, box2h2, memra #324): the second request
hit 130433 of 130433 prompt tokens from cache with ids identical over the continuation, and the
+2k extension was answered on the cached prefix. A correctness receipt, not a speed one: the
mechanism replaces a refusal with a restore of the same bytes.

**Scope:** sm_100a builds, all-layer `MEMRA_GLM5_TP` specs (the receipt's `all@0,1`).
`MEMRA_PREFIX_LATENT=1` is still required for the root planes. A partial spec refuses capture
and counts as TP-class at restore: the per-rank snapshot keys on one runtime over the whole
trunk, and the partial form used to mint an entry without shards that the restore admitted as
plain.

## Open calls

### `MEMRA_DSA_SCORE_TC`: scope of an sm_100a default

The receipt is positive and large: the TP-2 1M prime fell from 782 s (f32 scorer) to 339.8 s
with the bf16 arm, and the owner's precision call chose precision 1 over precision 2 (slower,
same flips). It is not bit-identical, and it covers the TP-2 plain prime only: spec x TP is
declined, so no verify walk ran under it. The arm keys on `t_q >= 8`, which the DFlash2 verify
walk reaches at K+1 >= 8. There bf16 scores would break the verify's rows-exact equality with
plain decode, a spec-verify against plain pair in the one-program-per-request sense, and the
retired box cannot measure it. Options: an sm_100a default limited to prefill chunks, with the
rows-exact verify keeping the f32 scorer and precision 2 deleted (recommended: it is the shape
the receipt measured); the door's own `t_q >= 8` rule as the sm_100a default; or delete the door
and both arms. Decide-by 2026-11-18; the door stays default OFF until then.

### `MEMRA_E4M3_BATCH_R2` and `MEMRA_Q80_BATCH_R2`: promote or delete

Both are bit-identical two-rows-per-warp twins of the batched verify-width matvec tier. Their
three-boot `tpwalk24` rows on the PP-2 spec route read +2.1% and +1.9% on the median, but the
lowest door boot sits under the highest off boot in both, so they are inconclusive, not
positive; the one-boot `doorscreen` rows read +2.69% and +1.40%. The missing cell named by the
lane needs the retired box. The 2026-11-04 ruling covers positive receipts (promote) and
negative, flat or missing ones (delete), not inconclusive ones, and the rows already reserve
this call for the owner by 2026-11-17. Options: promote both on sm_100a on these rows; delete
both with their kernels and gates (recommended: no cell can settle them on the card they target,
and git keeps the kernels for a later receipt on another card); or keep them default OFF for an
RTX 5090 or RTX PRO 6000 cell, which would be a new lane on a different card.
