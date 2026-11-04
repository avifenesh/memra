# GLM-5.3-Flash B200 doors decided after the box retired (2026-11-04)

> Banner, later on 2026-11-04: B200 is not maintained (owner: "b200 dont need to be maintain, we
> dont work on those"; "the hardware i dont own we cant maintain, best effort"). The sm_100a
> defaults below stand on the receipts quoted here, and the code stays as it is. No cell this
> record names as owed on an sm_100a pair will run, and the open calls that concern only sm_100a
> builds (the five-door bundle, the TP transport split, `MEMRA_HYPER_PRIME_V2`, the batched R2
> twins) are not pursued. The calls about other cards and launchers (the plain decode graph off
> sm_100a, launchers other than the B200 one, the k-pool v2 body on sm_120a) stay open.

The 2x B200 pair that carried the GLM-5.3-Flash TP-2 lanes is retired, so no new cell can run
on it. Owner, 2026-11-04: decide every B200 door from the receipts already on record. A winner
with a positive, correct receipt becomes a naked default on sm_100a builds, keyed on
`BUILT_CUDA_ARCH` as the existing per-device defaults are (`MEMRA_Q8_ROW_ILP` is the
precedent), and its switch goes. A door whose receipt is negative, flat or missing is deleted
with everything only it reaches. One-rig evidence sets a one-rig default: no other build changes.
The deleted doors are in the Removed doors ledger of `docs/FLAGS.md`; this record covers the
defaults that changed and the calls left open.

**The TP-2 receipts carried one posture, and it is the sm_100a default again.** The host-diet
row below was taken on the darklanes launcher's composed posture, and the TP-prefix and device
sampler rows in the same window: `MEMRA_TP_TRANSPORT=device-push`, `MEMRA_GLM5_TP_SYMMETRIC=1`
(KDA `wo` K-sliced at load, with a row-parallel partial-sum branch in `kda_tp_core` that prime
and decode both ran), `MEMRA_TP_AR_1STAGE=1` (the one-shot all-reduce at the MLA `wo` and
split-MoE reduce sites, prime chunks included) and decode through the symmetric graphs, which is
why those rows decode at 65-67 tok/s against the eager split walk's 35-39. An earlier pass on
2026-11-04 deleted the symmetric walk as unreceipted, which left those promoted defaults on a
program no receipt measured. The walk is restored as the sm_100a TP-2 default (its section
below), with `device-push` armed when no transport is named, and the posture doors with
positive rows (the HtoD diet, the coalesced MLA arm, the B200 MLA split on TP shards, the fused
KDA group, the matvec occupancy arms) are sm_100a defaults too, so a naked sm_100a TP-2 command
runs the measured posture again. The launcher's other doors keep their own calls: the q8 fuse
(inconclusive, below), `MEMRA_B200_GEMV_V2` (the owner's, decide-by 2026-11-17),
`MEMRA_PREFIX_LATENT` and `MEMRA_HYPER_SUFFIX_PRIME` (serving doors other launchers set too) and
door D (verify walk only, which the TP-2 launcher never served). What no cell has done is run
that posture on today's code: the receipts are from September builds, and the pair cell that
would confirm each default is named in its section. It needs an sm_100a pair, and none is in
hand.

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

**Still owed on today's code:** the receipt ran on the composed posture (above), which is the
naked sm_100a TP-2 program again. The confirming cell: an sm_100a pair, `glm5-tp2-box-probe` at
256k and 1M with `MEMRA_GLM5_TP=all@0,1` and nothing else, prime seconds against a slot-walk build
of the same commit, and the tape and ids against the posture's own reference. No gate cell runs
the grouped prime on any other card: on sm_120a the split walk primes through the slot walk.

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
gated on a logits band plus tape identity and never on bits (`glm5-tp-gate` arm S). tpwalk2's
`tp2s` arm is the root-orchestrated eager split walk; on sm_100a the symmetric walk now runs over
the split by default (below), and `MEMRA_GLM5_TP_SYMMETRIC=0` keeps the walk this row measured.

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
composed posture's window (above), which is the naked sm_100a TP-2 program again. The restored
planes do not depend on the walk (KDA conv/ssm state per head and the latent rows are the same
bytes either way). The cell owed on today's code: box2h2 again on an sm_100a pair, a 128k prompt
cold and then again, full hit and ids identical over the continuation, once at an odd and once at
an even decoded-token count (the per-rank graphs mirror the KDA ping-pong, and a restore between
graphed tokens must replay the right phase).

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

## The symmetric TP-2 walk with its graphs: sm_100a TP-2 default

**Chosen:** on sm_100a builds an armed TP-2 walk over the expert split takes the symmetric
posture. At load every TP layer's glue (norms, hc sites, a router copy, the dense FFN, the
shared-expert halves) is replicated to the peer and the KDA `wo` turns row-parallel. At decode
both ranks run the glue themselves and the three sharded ops meet in one-shot all-reduces fused
into the hc post launch, two crossings per layer, recorded as one CUDA graph per rank over each
run of KDA layers with each MLA layer in graphed halves around an eager middle. Under the
posture the root-orchestrated reduce sites the prime chunks take use the one-shot too. With no
transport named, the posture on a real pair arms `device-push`, the only transport the one-shot's
kernel peer stores admit. `MEMRA_GLM5_TP_SYM_GRAPH`, `MEMRA_GLM5_TP_SYM_GRAPH_MLA` and
`MEMRA_TP_AR_1STAGE` are deleted as switches. `MEMRA_GLM5_TP_SYMMETRIC` stays as the posture
selector, `=1` on and `=0` off on any build, the way `MEMRA_GLM5_TP_EXPERT_SPLIT` stays: the
one-card rig gate reaches the posture only through `=1` (arm Y) and its other arms pin `=0`.

**Rejected:** the deletion an earlier pass made the same day on the premise that the walk had no
receipt (darklanes carries positive rows, below); confirming that deletion as a policy deletion,
which would leave every promoted TP-2 row above on a program none of them measured; the eager
symmetric walk as the default, slower than the graphed walk at the same tape; MLA layers whole
and eager inside the graphs (`MEMRA_GLM5_TP_SYM_GRAPH_MLA=0`), slower below; the one-shot as a
door of its own on the root-orchestrated walk, flat there; keeping `peer-pull` as the transport
under the posture, which the one-shot cannot use because its ladder proves copies, not kernel
stores; and making `device-push` the sm_100a transport for every TP walk, which no pull-vs-push
row supports (below).

**Measurement that settled it:** 2x B200 pair, darklanes
`research/glm5-b200-mint-20260904/LANE.md`. symcheck2 (2026-09-06 23:40Z) ran the walk engaged,
45.53 then 54.06 tok/s, tape `d3c3955d916b8ef6`. tpwalk4 v2 (2026-09-07, two reps per arm,
tapes equal across reps): eager walk 54.46 / 54.93 tok/s, graphed walk on the served posture
74.65 / 74.87, graphed PP-2 control 71.33. tpwalk3 v5 (two reps): eager 53.30 / 53.38, KDA runs
graphed with MLA whole 71.11 / 78.76, with the MLA halves 82.58 / 82.18, every 160-token tape
equal to the eager walk's (`VERDICT:glm5-tp-sym-graph-byte-identical-1.65x`). tpwalk11: the fused
crossing 97.90 / 98.39 against 96.43 / 96.17 with the separate reduce and post, same tape
(`VERDICT:glm5-tp-fused-reduce-post-crossing`). The root-orchestrated split walk read 34.84 /
39.43 an hour before the eager walk's first row (tpwalk2, same pair), so that comparison is
across windows; the graph and MLA-halves rows are within one run. The one-shot alone on the
root-orchestrated walk read flat, 34.60 / 39.10, with the same tape `f6b309777546b67e`
(tpwalk2). The host diet, TP prefix and device sampler receipts all ran on this posture.

**Transport:** no balanced interleaved pull-vs-push row exists. The one same-window reading is
tpwalk (2026-09-06, one round per arm on the root-orchestrated walk, quoted in the walk's
2026-09-06 FLAGS row, no raw table in either repo): host-canonical 16.59, peer-pull 19.07,
device-push 20.49 tok/s against PP-2's 59.86. `device-push` is the default under the posture
because the link needs it and every posture receipt ran on it. `peer-pull` stays the sm_100a
default for the root-orchestrated walk, where its receipt was measured
(`docs/decisions/TP-PEER-PULL-DEFAULT.md`). The TP indexer's candidate exchange keeps the same
link predicate and engages wherever `device-push` is armed, posture or not.

**Owed on today's code (an sm_100a pair):** `glm5-tp2-box-probe` at 160 greedy tokens with
`MEMRA_GLM5_TP=all@0,1` and nothing else, checking `[glm5-tp-sym] engaged`,
`[glm5-tp-sym-graph] engaged` and `[glm5-tp-transport] armed transport=device-push`, against
`MEMRA_GLM5_GRAPH_NO_CAPTURE=1` (the eager walk, same tape) and `MEMRA_GLM5_TP_SYMMETRIC=0` (the
root-orchestrated walk, the split's class), three interleaved boots per arm; `glm5-tp-gate` on two
devices; `tests/tp_ar_gpu.rs`; and the reuse-pool continuation the graph row always owed: decode
an odd and an even number of tokens, continue with the prefix and suffix-prime paths, tape equal
to a cold run.

**Scope:** sm_100a builds, TP-2 over the expert split. Every other build keeps the
root-orchestrated walk, since no receipt exists there; `=1` reaches the posture there for the
gate, and with no transport named it arms `device-push` too.

## The HtoD diet: sm_100a default with the graphs

**Chosen:** on sm_100a builds door H (`MEMRA_HTOD_DIET`, alias `MEMRA_GLM5_HTOD_DIET`) is on
with neither name set: the ungated shared-expert add reads a resident ones buffer and the
latent-plane length mirror takes the async store. Bit-identical by construction. Both names stay:
`=1`/`=0` pick explicitly on any build, the gates and the door's GPU test pin them, and a
disagreeing pair still falls closed.

**Rejected:** deleting it as unpriced, which would end every glm5 decode graph, since each one
refuses capture without it (`glm5_graph_process_refusal`: a pageable upload inside a capture
replays stale bytes, the garbage tpwalk4 v2's bare graph arm decoded); keeping it OFF, which
leaves the plain decode graph and the restored symmetric graphs inert on every naked sm_100a
command; and turning it on for every build, which would start the plain decode graph on cards
where it has no throughput receipt.

**Measurement that settled it:** its own row measures nothing (shexpab: the OFF arm already had
the diet through the alias and the graph capture). What settles it is the graphs it gates. Every
positive graph receipt on the 2x B200 pair ran with it on: the plain decode graph's graphab
(three interleaved boots per arm, 66.40 against 72.97 tok/s, +9.89%, darklanes
`research/glm5-b200-20260902/LANE.md`) and the symmetric walk's graphs (tpwalk4 v2, tpwalk3 v5,
above).

**Scope:** sm_100a builds. Other builds keep it OFF, so the plain decode graph, default ON
everywhere since 2026-09-04, still refuses capture on a naked command there: an open owner call
(below).

## MLA decode on sm_100a: the coalesced arm, and the B200 split on TP head shards only

**Chosen:** on sm_100a builds the MLA absorb and decompress launchers take the warp-per-row
coalesced twins whenever a split is in force (`mla_coalesce_on`, the named class
`mla_warp_row_reduce`), over the generic decode split on an unsharded layer. A glm5 TP head shard
refuses the generic split, and there the B200 decode arm's table split, scaled by the head ratio,
is the split, for absorb, decompress and gathered attention (`mla_b200_decode_arm_on(scope)`).
`MEMRA_MLA_COALESCE` and `MEMRA_B200_MLA_DECODE_ARM` are deleted.

**Rejected:** the B200 arm on unsharded layers, where the generic split beat it; keeping either
door default OFF, which left the TP-2 posture's MLA layers unsplit and uncoalesced on a naked
command; and the coalesced twins at split 1, which lose.

**Measurement that settled it:** 2x B200 pair, darklanes `research/glm5-b200-20260902/LANE.md`:
coalesce round 2 (two interleaved rounds, ranges clear, every tape `9437b599f6b9d2a9`) read the
generic split alone at 59.83 tok/s and coalesce on top of it at 60.93 (+1.84%), coalesce alone at
51.07 (-10.82%); `mlasplit` (three interleaved rounds, same tape) read the generic split at
58.32 against the B200 arm's 58.16, below on every round. On the TP-2 walk (darklanes
`research/glm5-b200-mint-20260904/LANE.md`, tpwalk6 v2) the posture with the arm and coalesce
read 91.55 / 91.33 against 84.20 without them (one rep).

**Owed:** an ON/OFF argmax gate for the coalesced class, which its row named; the served tapes are
the only correctness evidence. On an sm_100a build `tests/mla_gpu_forward.rs` now reaches the
coalesced class through the Engine dispatch at its CPU-oracle tolerance.

**Scope:** sm_100a builds only; other builds keep the thread-per-row kernels and the generic
split, and TP head shards there stay unsplit.

## The fused KDA six-projection group: sm_100a default

**Chosen:** on sm_100a builds the KDA stage-1 six projections run as one fused launch per layer
per token (`kda_fused_proj_on`; the e4m3 arm on the hybrid mint, the bf16 and q8 arms on other
operand classes), on unsharded layers and TP head shards alike. `MEMRA_KDA_FUSED_PROJ` stays as
an explicit `=1`/`=0` selector on any build, because its two GPU gates drive both arms through
it.

**Rejected:** keeping it default OFF, which the retired launcher alone justified, and which left
the sm_100a attention-norm q8_1 fuse (above) computing a q8_1 view that nothing read; deleting it
on a missing single-door row, which would delete the launch the TP-2 posture's best rows ran.

**Measurement that settled it:** 2x B200 pair. The five-door decode bundle it rode in (with the
matvec occupancy arms, the fused hc pre chain, the B200 MLA arm and the q8 fuse), three
interleaved fresh boots per arm (darklanes `research/glm5-b200-20260902/LANE.md`): plain 42.5 /
42.1 -> 44.9 / 44.5 tok/s, DFlash2 spec 71.4 / 49.4 -> 77.1 / 53.5. On the symmetric TP-2 walk,
adding it alone moved 82.58 to 87.90 tok/s with the tape unchanged (tpwalk3 v5, one rep; darklanes
`research/glm5-b200-mint-20260904/LANE.md`). Every TP-2 row after that carried it.

**Scope:** sm_100a builds. Other builds keep the six launches.

## The matvec occupancy arms: sm_100a default

**Chosen:** on sm_100a builds the plain-decode MoE and matvec family runs its four bit-identical
occupancy arms (the warp-packed `_w4` MoE pair, the NVFP4 rp grid fill, the `_g2` fused gate+up,
the pipelined bf16 rows kernel) with no switch. `MEMRA_B200_MATVEC_ARM` is deleted; the bench
calls both kernels of each pair directly.

**Rejected:** keeping it default OFF, which also kept the promoted `MEMRA_B200_MR1_FILL=16`
from ever running; deleting it, against a positive kernel A/B and bundle row.

**Measurement that settled it:** 2x B200 pair (darklanes `research/glm5-b200-20260902/LANE.md`):
`b200_matvec_bench`, N=5 interleaved, bit-identical, bf16 up 33.4 -> 28.5 us, down 30.65 -> 24.9,
MoE gate/up 55.0 -> 53.3, down 37.6 -> 36.0; the five-door decode bundle (above) +5.6% plain,
+8% spec, its nsys census showing the door kernels; and nvab's `MEMRA_B200_MR1_FILL=16` +2.03%,
measured with the arms on.

**Scope:** sm_100a builds. Other builds keep the shipped kernels.

## The q8_1 activation-quantize fuse: kept OFF, inconclusive

`MEMRA_GLM5_Q8_FUSE` has only the bundle row, and the bundle cannot split its gain across five
doors (one of them, the B200 MLA arm, later lost a single-door row). Its sibling
`MEMRA_HC_PRE_ZQ8` read flat alone and was deleted. It stays default OFF with decide-by
2026-11-18; the missing gate is a single-door interleaved A/B on an sm_100a pair. With no pair in
hand the date deletes it, unless the owner accepts the bundle per door (open call below).

## Open calls

### The plain decode graph off sm_100a

`MEMRA_GLM5_DECODE_GRAPH` is default ON on every build, but it refuses capture without the HtoD
diet, which stays OFF off sm_100a, so on the RTX 5090 and RTX PRO 6000 a naked glm5 command never
captures. Options: turn the diet on for every build, which starts the graph there with an
exactness gate but no throughput row (recommended only with a 5090 or PRO 6000 graph A/B); or
record the graph default as sm_100a-effective and leave both as they are (recommended until that
cell runs, since it changes nothing today). Decide-by 2026-11-18.

### What the five-door decode bundle proves per door

The 2026-09-02 bundle (`MEMRA_B200_MATVEC_ARM`, `MEMRA_HC_FUSED_PRE`, `MEMRA_B200_MLA_DECODE_ARM`,
`MEMRA_GLM5_Q8_FUSE`, `MEMRA_KDA_FUSED_PROJ`) is one positive interleaved row for five doors.
`MEMRA_HC_FUSED_PRE` was promoted on it earlier, and one of the five later lost a single-door row
(the B200 MLA arm on unsharded layers, `mlasplit`). This record promotes the doors with a second,
door-level row (the fused KDA group: tpwalk3 v5; the matvec occupancy arms: a bit-identical
kernel A/B and the `MEMRA_B200_MR1_FILL` row that ran on it) and keeps `MEMRA_GLM5_Q8_FUSE`,
which has only the bundle, default OFF with a decide-by. Options: accept the bundle per door
(promote the q8 fuse too); keep this split (recommended); or require a single-door row for all,
which no sm_100a box in hand can produce. Decide-by 2026-11-18.

### The sm_100a TP transport: device-push under the posture, peer-pull elsewhere

The owner named peer-pull among the winners to make card defaults, and it is, for the walk its
receipt measured: the root-orchestrated TP walk against the host bounce. The restored symmetric
walk cannot run on it: its one-shot crossings are kernel peer loads and stores, and only
`device-push`'s ladder proves one, so an sm_100a TP-2 load with no transport named arms
`device-push` under the posture (`glm5_tp::unset_transport_for`). No balanced pull-vs-push row
exists. Options: keep this split (recommended: each transport default sits on the walk its
receipts ran, and a naked command reaches the measured posture); make `device-push` the sm_100a
default for every TP walk (simpler, but it drops peer-pull's only receipt on the floor); or keep
`peer-pull` everywhere and add a kernel-store ladder to its arm so the walk may run on it (new
code, and the walk's fan-out hops would then run on a transport none of its rows used).
Decide-by 2026-11-18.

### `MEMRA_HYPER_PRIME_V2`: the owner's flip, or delete

The PP-2 mHC prime schedule read about 2x on TTFT on the 2x B200 pair (4.08 / 16.73 / 56.30 s to
2.09 / 9.59 / 30.63 s at 4k / 42k / 128k), one boot per arm, and its row reserved the flip for
the owner. PP-2 is no longer that model's decision shape and the pair is retired. Options: flip
it on these rows; or let its decide-by (2026-11-18) delete it (recommended, unless a PP-2 serving
shape comes back).

### Launchers other than the B200 one

The 2026-11-04 ruling voided "a launcher sets it" for the retired B200 launcher. The darklanes
glm53 launcher still sets `MEMRA_GLM5_HTOD_DIET`, `MEMRA_MOE_VROWS_DEV_TABLES` (door D),
`MEMRA_HYPER_SUFFIX_PRIME`, `MEMRA_PREFIX_LATENT`, `MEMRA_PP_BF16` and `MEMRA_BF16_MMV`, and
other retired launchers set `MEMRA_ADMIT_PREDICT_SHADOW`. Door D has no timing receipt and would
go under door hygiene; the others are serving or per-family configuration. Options: void the
clause for every retired launcher and run the same sweep on those doors (recommended, since
hosted serving is retired); or keep the clause for launchers that still exist. Decide-by
2026-11-18.

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
