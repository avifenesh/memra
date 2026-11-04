# GLM-5.3-Flash B200 doors decided after the box retired (2026-11-04)

The 2x B200 pair that carried the GLM-5.3-Flash TP-2 lanes is retired, so no new cell can run
on it. Owner, 2026-11-04: decide every B200 door from the receipts already on record. A winner
with a positive, correct receipt becomes a naked default on sm_100a builds, keyed on
`BUILT_CUDA_ARCH` as the existing per-device defaults are (`MEMRA_Q8_ROW_ILP` is the
precedent), and its switch goes. A door whose receipt is negative, flat or missing is deleted
with everything only it reaches. One-rig evidence sets a one-rig default: no other build changes.
The deleted doors are in the Removed doors ledger of `docs/FLAGS.md`; this record covers the
defaults that changed and the calls left open.

**The TP-2 receipts carried a posture that no longer exists.** The host-diet row below was taken
on the darklanes launcher's composed posture, and the TP-prefix row in the same window, when the
launcher armed the TP prefix together with it: `MEMRA_TP_TRANSPORT=device-push`,
`MEMRA_GLM5_TP_SYMMETRIC=1` (KDA `wo` K-sliced at load, with a row-parallel partial-sum branch
in `kda_tp_core` that prime and decode both ran), `MEMRA_TP_AR_1STAGE=1` (the one-shot
all-reduce at the MLA `wo` and split-MoE reduce sites, prime chunks included) and decode through
the symmetric graphs, which is why those rows decode at 65-67 tok/s against the eager split
walk's 35-39. fe66a24a0d deleted those programs (the open call below). So those promoted TP-2
defaults run a numeric program no receipt measured: column-parallel KDA `wo` over the gather,
return-to-root reduces, root-orchestrated eager decode. Their tapes and ids on that program are
unverified, and each on/off delta is assumed to transfer. Where a mechanism is byte-identical
by construction (the host diet: launch order and host work only), the on/off identity still
holds; what is unverified is the walk underneath. The pair cell that would confirm them is
named in each section; it needs an sm_100a pair, and none is in hand.

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

**Unverified on the promoted program:** the receipt ran on the composed posture (above). The
cell that would confirm the transfer: an sm_100a pair, `glm5-tp2-box-probe` at 256k and 1M with
`MEMRA_GLM5_TP=all@0,1` and `MEMRA_TP_TRANSPORT=device-push` on the post-deletion walk, prime
seconds against a slot-walk build of the same commit, and the tape and ids against the split
walk's own reference at its band (`glm5-tp-gate` arm S's class). No gate cell runs the grouped
prime on any other card: on sm_120a the split walk primes through the slot walk.

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
gated on a logits band plus tape identity and never on bits (`glm5-tp-gate` arm S). Unlike the
other TP-2 rows here, tpwalk2's `tp2s` arm is the root-orchestrated eager split walk, the decode
program that remains after fe66a24a0d; its prime has since gained the grouped prime above.

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

**Scope:** sm_100a builds only; the v1 body stays the selector elsewhere. The RTX 5090 rig has a
kernel-level reading for v2 (`VERDICT:kpool-select-v2-rig-ordering-wins`, all 18 pairs, 61,323
pools random 114.4 against 132.0 us) but no model-scale row; whether sm_120a takes v2 on that
reading is an open call (below).

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
mechanism replaces a refusal with a restore of the same bytes. The receipt was taken in the
composed posture's window (above). The restored planes do not depend on the walk (KDA conv/ssm
state per head and the latent rows are the same bytes either way), but ids over a restored
continuation on the promoted walk are unverified. The cell that would confirm it: box2h2 again on an sm_100a pair on
the post-deletion walk, a 128k prompt cold and then again, full hit and ids identical over the
continuation.

**Scope:** sm_100a builds, all-layer `MEMRA_GLM5_TP` specs (the receipt's `all@0,1`).
`MEMRA_PREFIX_LATENT=1` is still required for the root planes. A partial spec refuses capture
and counts as TP-class at restore: the per-rank snapshot keys on one runtime over the whole
trunk, and the partial form used to mint an entry without shards that the restore admitted as
plain.

## The tensor-core DSA prefill scorer: sm_100a default for prefill chunks

**Chosen:** on sm_100a builds the DSA indexer scores a PREFILL chunk at `t_q >= 8` on bf16
`mma.sync` tensor cores with f32 accumulation, one MMA per product (`MEMRA_DSA_SCORE_TC`'s
precision 1). The rows-exact verify walk and decode keep the f32 scorer. `MEMRA_DSA_SCORE_TC`
and precision 2 are deleted.

**Rejected:** the door's own `t_q >= 8` rule as the default, which also reaches the DFlash2
rows-exact verify walk at K+1 >= 8, where bf16 scores would break the verify's row-for-row
equality with plain decode (a spec-verify against plain pair in the one-program-per-request
sense) and which no receipt covers; precision 2 (every operand split hi/lo, three MMAs), slower
with the same flips; and deleting the door, which would give up a measured 2.3x on the 1M prime.

**Measurement that settled it:** 2x B200 pair (darklanes
`research/glm5-b200-mint-20260904/LANE.md:3372-3502`, tpwalk18/18b/18c/20/21/22, memra #309): the
TP-2 1M prime fell from 782 s (f32 scorer) to 339.8 s with precision 1, 310.1 s composed with the
indexer split. Not bit-identical: on the pair it moved one greedy token at 36/64 after a 32k
prompt, and the first-token argmax control (tpwalk22) showed the f32-scorer posture already
diverging at the same near-tie token. The owner chose precision 1 over precision 2 on these rows.

**Scope:** sm_100a builds, prefill chunks only. Other builds keep the f32 scorer. The prime rows
ran on the TP-2 walk of that window; the scorer's numerics do not depend on the walk around it,
but its share of the prime does, so the 2.3x is a statement about that walk.

## Open calls

### The symmetric TP-2 walk: deleted as unreceipted, but darklanes carries positive rows

fe66a24a0d deleted `MEMRA_GLM5_TP_SYMMETRIC`, `MEMRA_GLM5_TP_SYM_GRAPH` with its seam
`MEMRA_GLM5_TP_SYM_GRAPH_MLA`, and `MEMRA_TP_AR_1STAGE` on the premise that none had a receipt.
That premise holds for this repo and not for darklanes `research/glm5-b200-mint-20260904/LANE.md`:
symcheck2 (2026-09-06 23:40Z) ran the symmetric walk engaged (`[glm5-tp-sym] engaged`, 45.53
tok/s, then 54.06, tape `d3c3955d916b8ef6`), and tpwalk4 v2 (2026-09-07) read two reps per arm:
eager symmetric 54.46 / 54.93 tok/s, the graph door on the served posture 74.65 / 74.87, the
graphed PP-2 control 71.33, tapes equal across reps. The same tpwalk4 v2 run caught the bare
graph arm decoding garbage (fixed in 9d074abc6 by declining on the process preconditions). The
one-shot alone read flat (tpwalk2: 34.84 / 39.43 split against 34.60 / 39.10 with the one-shot),
but it is the symmetric walk's crossing primitive, so that row does not price it inside the walk.
This record accepts darklanes prose rows elsewhere (the k-pool selector, the top-k twin, the
host diet), so by the same standard these are positive rows, and the 2026-11-04 rule would make
the walk a TP-2 sm_100a default rather than delete it. The cost of the deletion: TP-2 decode on
any future pair is the root-orchestrated eager walk (tpwalk2's split, 34.84 / 39.43 tok/s)
instead of the graphed symmetric walk (74.65 / 74.87). Options: restore the four doors as TP-2
sm_100a defaults (reverting fe66a24a0d, then the composed posture is again what the receipts
measured); or confirm the deletion as a policy deletion (the box is retired and no sm_100a
pair is in hand), in which case the ledger keeps these rows and calls it that. The branch
carries the deletion until the owner rules.

### `MEMRA_KPOOL_SELECT_V2` on sm_120a

The v2 body is the sm_100a default above. The RTX 5090 rig's kernel instrument won all 18 pairs
for v2; no model-scale row exists on any sm_120a card. Options: take v2 on sm_120a on the
instrument (the two bodies are bit-identical, so only speed is at stake), or wait for a 5090
model-scale cell, which runs centrally.

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
