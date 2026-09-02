# qwen4_exp EP2: the design, and the arithmetic that kills it as the 200+ precondition

Lane `lane/q4e-ep2-20260901`. Scope: expert-parallel execution of the qwen4_exp
(Qwen3.8-Flash-Next-NVFP4) 512-expert bank across two cards, sent as the floor-halving
mechanism behind the owner's 200+ tok/s single-request spec-decode target.

**No hardware was used for this lane.** The rig is one sm_120a laptop 5090 and is
exactness-only (`LAW:rig-gpu-exactness-only`); it cannot run any two-card cell. Every
number below is either (a) read off a banked box receipt already in this repo, or (b)
arithmetic over such rows, labelled as arithmetic. No two-card number is fabricated and
no new perf claim is made.

**Landing order (updated at the rebase).** `spec/downsel/DOWNSEL.md` (PR #27) and
`spec/ROUND-BUDGET-COMPOSITION.md` (PR #26) were open sibling lanes when this lane branched;
both are on `main` at the branch point this head sits on, so every citation below resolves in
the merge result (and this PR edits ROUND-BUDGET-COMPOSITION.md in place, which is only possible
because it is in the base). Nothing in the verdict depends on them resolving: the primary
attribution is re-derivable from a receipt that IS in this tree, namely
`spec/mtp10/nsys/win-spec_cuda_gpu_kern_sum.csv` (median 111,008 ns x 48 layers = 5.328 ms for
`qmatvec_nvfp4_modelopt_sel_gu_silu_f32` and 64,576 ns x 48 = 3.100 ms for
`qmatvec_nvfp4_modelopt_sel_f32_v3`, 2160 instances each = 48 layers x 45 verify rounds,
per-instance stddev 3.2 / 0.9 us; 8.428 ms of a 31.94 ms round = 26.39%). DOWNSEL is cited
because it derived that figure first and built its own merged ceiling on it, not because the
number needs it.

## THE VERDICT (read this alone)

**Two-card EP2 cannot reach 200 tok/s on this artifact, and the margin is a factor of ~2.
It is worth about +13% to +32%. The premise it was sent on is falsified.**

`spec/ROUND-BUDGET-COMPOSITION.md` (PR #26) reads the K-ladder as
`verify(t) = 15.1 ms flat + 2.10 ms/row`, calls the flat 15.1 ms "the single-card
weight-read floor", and concludes that 200+ needs that floor halved by "each card reading
half the expert bytes concurrently". The floor is not expert bytes, and halving the expert
work does not produce 200.

### The one inequality that decides it

Measured single-card best at K=5 is **136.2 tok/s / 31.94 ms per round**
(`devtwin/ab-devtwin-k5-dt4-raw.tsv`, reps=5, median 7.341 ms/token, accept 0.861,
mean_accept_len 4.41). 200 tok/s is a 21.75 ms round, so **10.19 ms has to come out**.

A two-card split of a fraction `f` of the round saves at most `f/2` of it. So

```
f/2 * 31.94 >= 10.19   ->   f >= 63.8%
```

**Two-card EP2 needs 63.8% of the K=5 round to be routed-expert work.** Every attribution
of that section that exists in this repo, at any scope:

| source | shape / era | routed-expert share | EP2 ceiling (perfect, free) |
|---|---|---|---|
| `spec/downsel/DOWNSEL.md` §2, from `spec/mtp10/nsys/win-spec_cuda_gpu_kern_sum.csv` | K=5, 136.2 tok/s program, real kernel medians, 2160 instances/kernel | **26.4%** (8.428 ms) | 156.9 |
| `spec/moeu/MOEUNION.md` | K=5, mtp10 thinkon attribution, scope-caveated | 22.9% | 153.8 |
| + MTP draft MoE and the shared expert, credited generously | K=5 | ~33% | 163.1 |
| `spec/mtp{4,5,6}/spec-profile-k5-*.tsv` | K=5 but a **66 tok/s** program: the per-expert dequant executor still live at 30% of attributed, eager and fully synced | **48.5%** of attributed | 179.8 |
| **needed for 200 on two cards** | | **63.8%** | 200 |

**The most favourable receipt in the repo is short by a third, and it is the stale one.**
That is why this verdict does not wait on settling the attribution dispute: the answer is
the same across a 22.9%-to-48.5% span. **EP2's two-card ceiling is 154-180 tok/s**, taken
at a perfect halving with zero exchange cost, and the honest centre of that band on the
shipped program is **~157**.

(The infinite-card asymptote, `f >= 31.9%`, is genuinely borderline between these
attributions, and it is also academic: the owner confirmed two cards.)

### Why the flat floor is not expert bytes

The routed section's cost scales with SLOT COUNT, not weight bytes
(`KNEE:q4e-sel-slots-not-bytes`; the `moeu` probe measured 10 -> 60 slots = 5.02x at 6x the
distinct bytes). So of the 8.428 ms at t=6, about 8.428/5.02 = 1.68 ms is t-independent and
(8.428-1.68)/5 = 1.35 ms/row rides the slope. **EP2 halves ~0.84 ms of the 15.1 ms flat
floor (5.6%) and ~0.67 ms of the 2.10 ms/row slope (32%): a SLOPE lever, not a floor
lever.** The round-budget doc's composition `(7.5 + 2.10x9 + 6.53)/5.02` halved the floor
and kept the slope, exactly inverted.

**What the floor actually is** (`perf/PROFILE-C0.md` §1, banked `--profile` decode sections
at 100k; absolutes are sync-bounded and inflated, shares are the signal): `qsa.sdpa`
**29.7%** (flat in depth, grid-bound at 24 CTAs on a 188-SM card),
`ple.host_ngram_gather` 14.1%, `hyper.read` 9.3%, `gdn.proj` 7.5%, `qsa.idx_host` 7.3%,
against `moe.sel_grouped` 7.7%. And in the mtp10 nsys summary the DENSE trunk matvecs
(`qmatvec_bf16w_f32` 18.8% + `qmatvec_bf16w_mt_f32` 18.3% = **37%** of kernel time) are more
than twice the sel section's 15.6%. **The floor is dense-trunk and attention work. Expert
parallelism does not touch any of it.**

Per `LAW:price-the-dispatch-first` this lane therefore stops here rather than building the
machinery a falsified premise would need. What it does instead: bank the arithmetic, record
the reuse map for whoever wants the +15%, close the coverage hole in the placement plumbing
that already shipped, and name the cheap box cell that would settle the attribution span.

## 1. Reuse map: what exists, and it is more than the lane was told

**The qwen4_exp two-card expert split is already implemented, merged, and gated.** Nothing
in deliverable "register qwen4_exp on the EP path" needs building. Verbatim reuse:

| piece | where | state |
|---|---|---|
| per-layer expert placement, 2 ranks | `qwen4exp_gpu.rs` `Tp2Placement` / `LayerPlacement` | shipped |
| the placement door | `MEMRA_Q4E_EP_MAP`, `docs/FLAGS.md:1173` | shipped, **default OFF = even split** |
| the frozen map format | `memra-ep-map-v1`, minted by `tools/build_expert_placement_map.py` | shipped, fleet-shared with glm5's `MEMRA_EP_MAP` |
| route split by placement | `qwen4exp_gpu.rs` prefill + decode MoE sites (`place.rank(eid)` / `place.local(eid)`) | shipped |
| this-card-only expert executor | `tp2_moe_rows` (grouped, SLOT_CAP 8192) | shipped |
| dispatch accounting | `tp2_count_split` -> `peer_slots` / `home_slots` / `both_card_rows` / `engaged` | shipped, and it is a NON-VACUITY bar, not a metric |
| calibrated two-regime class gate | `--tp2-prefill-gate`, bands prime 1.4e-4 / decode 1.6e-4 | shipped (PROFILE-10 section 2) |
| three deliberate red arms | `MEMRA_Q4E_TP2_GATE_RED=skip-peer-moe\|peer-local-ids\|reverse-peer-weights` | shipped, all four orders above the band |
| P2P grant + peer push | `tp2_enable_p2p`, `launch_push` (`q4e_push_f32`) | shipped |

Because it exists, it is also already MEASURED, which is why this lane could reach a
verdict without hardware. Two receipts matter more than the rest:

- **Engagement under the even split** (`round2-box-receipts/TP2-CLASS-GATE.md`):
  `peer_slots=6908 home_slots=6532 peer_slot_fraction=0.5140 layer_tokens=1344
  both_card_rows=1343 both_card_fraction=0.9993`. The dispatch really does split ~50/50,
  and essentially every layer-token pays a cross-card join.
- **Plain t=1 decode, interleaved x5, same run** (`perf/ab-tp2graphs2-nvfp4.tsv`,
  `perf/PROFILE-3.md`): single card 15.57 ms, TP2 **14.22 ms**, TP2 wins **1.095x**.
  Later box, `perf/PROFILE-5.md`: single 14.5 ms, TP2 12.6-12.9 ms.

**And that 1.095x is the whole measured two-card win, against a spec path that is already
1.55x faster than plain.** Same PROFILE-5: single-card spec K=5 is **8.37 ms/token /
119.50 tok/s**, i.e. 1.55x the plain TP2 route. Every one of those timing lines carries the
in-tree prefix declaring itself an untuned eager wall clock taken under correctness-arm
residency and explicitly not a perf claim, and they are quoted here with that scope intact.

## 2. What qwen4_exp's two-card program actually is, and the two claims it corrects

The lane brief carried a hypothesis worth stating and then correcting, because both halves
turned out to be wrong in a checkable way:

> "expert-parallel exchanges only routed activations, not full hidden allreduces"

**(a) Only the MoE bank's DISPATCH is split. Its RESIDENCY is not.** Card 0 keeps the
complete 512-expert bank and addresses it by global id (`local_of[e] = e` for rank-0
experts); card 1 holds an ADDITIONAL gathered 256-expert copy. The pair stores **1.5x** the
bank, not 1.0x. Measured: single-card post-load 89,971 MiB; TP2 post-load
`0, 92,755 MiB | 1, 40,211 MiB`: card 0 pays **+2,784 MiB** and card 1's 40 GiB is a second
copy of the upper half. So "each card reads half the expert BYTES" is not what runs. What
runs is "each card runs half the (token, expert) SLOTS", which, per
`KNEE:q4e-sel-slots-not-bytes`, is the right lever anyway, and is why the section halves
at all. **State this distinction or `VERDICT:q4e-moeu-dead` looks like it kills the lane:
it does not. moeu died because byte DEDUP does not pay at fixed slots. EP2's win is halved
per-card SLOT WORK. Different quantity, opposite direction, both true.**

**(b) The exchange is not a routed-activation scatter. It is a dense partial-sum ADD.**
`launch_push` moves a full `[t, hidden]` f32 plane per direction and the join is
`e.add(card0_partial, card1_partial)` in a fixed rank order; rows with no experts on a card
contribute **zero rows**, and the join sums the halves. There is no compaction: a card
sends a full dense plane whether it owned 1 slot or 30. That is a 2-rank allreduce in all
but name, the very shape the brief expected EP2 to avoid.

It is nevertheless the CHEAP allreduce, and this is the honest form of "why EP2's budget
survives PCIe where a dense TP2's did not". Three reasons, with the arithmetic:

| | qwen4_exp MoE-only EP2 seam | a glm5-style column-parallel dense TP2 |
|---|---|---|
| stream crossed | the mixed-down MoE stream, **hidden = 2560** | the **wide gated residual, 10240** (hc_count 4 x 2560) |
| crossings per layer | 1 join (2 pushes) | >= 2 sublayer reductions (mixer + MLP) |
| bytes per crossing at t=6 | 6 x 2560 x 4 = 61,440 B x 2 | 6 x 10240 x 4 = 245,760 B, ring both ways ~= 491,520 B |
| **per verify chunk, 48 layers** | **5.90 MB** | **~47.2 MB** |
| wire time at ~50 GB/s | **0.118 ms** vs a 31.94 ms round | ~0.94 ms, and it is on the critical path twice per layer |

**8x less traffic and a 4x narrower seam, because the MoE operates on the mixed-down 2560
stream while every dense sublayer lives on the 10240 wide one.** That is the structural
reason a MoE-only split is affordable here and a dense split is not, and it is the part of
the lane's thesis that survives. It is also why the ceiling in the verdict used zero
exchange cost as the optimistic bound: at 0.118 ms the exchange is not what stops EP2.

**The unmeasured risk that could make even 0.118 ms wrong.** `launch_push` is a UVA peer
STORE kernel, SM-issued peer access, not a copy-engine `cudaMemcpyPeerAsync`.
`tp_transport.rs` (and `research/glm53-flash-bringup-20260827/tp-transport-20260901/LANE.md`)
record that on direct-attach / `NODE`-topology hosts the driver stages SM-issued peer access
**through system memory by default**, measured **~15x slower than NCCL**, while
`nvidia-smi topo -p2p r` still reads OK and `cudaMemcpy` still looks healthy, glm5's
transport stayed copy-engine-only specifically to dodge this. At a 15x penalty the 5.90 MB
becomes ~1.79 ms per chunk, which would eat **42%** of EP2's 4.21 ms prize. So the honest
band on the ceiling is:

| EP2 K=5 ceiling | round | tok/s |
|---|---|---|
| peer path healthy (0.118 ms exchange) | 27.85 ms | **156.2** |
| peer path SysMem-staged (1.79 ms exchange) | 29.52 ms | **147.4** |

Both short of 200, which is why this risk changes the size of the prize and not the verdict.
It is still the first thing to probe before anyone trusts a byte model on this path
(`simpleP2P`-class kernel peer read, not a `cudaMemcpy` probe).

(Those two rows use the 26.4% attribution. Under the generous 48.5% reading the same
exchange costs shift a 179.8 ceiling to ~179 healthy / ~167 staged. Either way the exchange
is second-order against the attribution uncertainty, and neither reading reaches 200.)

## 3. Where the router runs, where activations cross, what stays single-card

For the record, since the design was asked for and the answer is what an EP2 arm would
inherit:

- **Router: card 0 only, on the HOST.** Card 0 runs the router GEMV, `dtoh`s the
  `[t, 512]` logits, and does softmax-top-10 as a host twin; the host then hands each card
  its filtered slot list. That is a host round trip **per MoE layer** (48 per forward).
  The single-card serving path does NOT do this, `routerdev` is default ON and the route
  never leaves the card. **A two-card spec arm either gives up the device router (48 syncs
  per verify chunk) or needs a device-side split that does not exist.** This is the largest
  piece of real engineering an EP2 spec arm would need, and it is uncosted.
- **Activations cross at the MoE join only** (in a MoE-only arm): one `[t, 2560]` f32 plane
  each way, per MoE layer. Route metadata does not cross, the host already has it.
- **Single-card resident by construction, and correctly so:** the PLE / n-gram block (its
  102 GB table is host-resident and shared, gather-only), the QSA indexer selection and its
  raw-key host cache + `pooled_dev` / `raw_dev` mirrors + `idx_audit` (structurally card-0
  only), the GDN recurrent state and the KV halves (card-local), `hyper.read`/`hyper.write`
  on the wide stream, and `lm_head` (vocab-row split, joined on the host, no P2P).
- **Consequence for depth, measured, and it is the other reason not to build this:** card 1
  is **FLAT at 43,603 MiB** from fill=16,384 onward while card 0 absorbs every growing byte,
  so TP2 OOMs during a fill below 100k where one card reaches ~731k. Card 1's ~54 GiB free
  is unusable for context. A pure expert-residency partition would free card-0 VRAM, but it
  would NOT move the growing caches (indexer/raw-key/pooled are card-0 only), so it does not
  by itself turn the depth regression around either.
- **There is no TP2 verify path at all.** `decode_step_tp2` is t==1-wired and
  `--ladder-tp2` refuses `--ladder-spec` ("spec at depth is single-card"). Every two-card
  number this family has is plain decode. **The 200+ target lives entirely on the spec path,
  and the spec path has no two-card arm**, which is the practical reason the 26.39% ceiling
  was never going to be tested cheaply.

## 4. What this lane changed in the engine

Nothing in the parallel machinery, deliberately, per the verdict. Two small things, both in
the already-shipped placement plumbing:

1. **A coverage hole closed.** `qwen4exp_gpu.rs` carried **no test module at all** when this lane branched (the selgroup lane has since landed `sel_group_tests` beside this one), so
   `Tp2Placement` / `LayerPlacement`, the bookkeeping that decides which card owns which
   expert and at which local slot, had zero coverage, while a future placement A/B depends
   on it entirely. 18 unit tests added (`mod tp2_placement_tests`), pure host logic, no GPU:
   the even-split control-arm property (card-0 slot IS the global id, card-1 gather is the
   ascending suffix, `is_even` recognises it and rejects a balanced PERMUTATION), the
   local-slot bijection over a full 512-expert non-contiguous placement, and one test per
   fail-closed clause (format, ranks != 2, expert-count, entry_rank, missing/empty `layers`,
   assignment length, rank id, unbalanced layer + its rebalance knob, uncovered MoE layer,
   layer/map geometry disagreement, unreadable path).
2. **An odd-bank geometry refused by name, scoped correctly.** The first version of this entry
   claimed a latent out-of-bounds, and review showed the tree does not have one. The correction
   is recorded rather than quietly edited, because a design doc's section 4 gets quoted back as
   fact:
   - **Production could not reach it.** `build_tp2_shard` already refuses `experts % 2 != 0`
     eleven lines before it asks for a `LayerPlacement`, and has since before this branch.
   - **There was no allocation to overflow.** The card-1 bank upload sizes on
     `place.card1.len()`, so an odd split would have produced an UNBALANCED 3-of-5 card-1 half,
     not an overflowing one.
   - **But `load()` did have a real hole**, and this is what the check is worth: `half =
     expert_count / 2` FLOORS, so a map placing exactly **2 of 5** experts on card 1 satisfied
     the balance clause and loaded clean. The balance clause caught only the *unbalanced* odd
     maps, and for those it named the wrong problem.

   So: `layer()` and `load()` are `pub`, an odd bank has no two-card answer, and the refusal now
   comes from the function whose contract it breaks instead of from an upstream check a future
   caller may not have. **Executed rather than asserted** per `LAW:loud-failures-fail-quietly`,
   and the test arm was rewritten to the case that actually bites: with the `load()` check
   neutered the balanced 2-of-5 map **loads clean** and the test fails with `odd-balanced:
   expected a refusal naming "ODD", but the map loaded`. Testing only the 3-of-5 map would have
   been nearly vacuous, since the balance clause already refused it.

qwen4_exp has 512 experts so (2) does not bite today. It is landed because the refusal is free
and because it closes the `load()` hole above on the public API.

**No new `MEMRA_*` env read, so no `docs/FLAGS.md` row is owed.** The one door this lane
touches, `MEMRA_Q4E_EP_MAP`, already has its row and already defaults OFF with its reason
written.

## 5. The cell that settles the attribution span, and the instrument already exists

The verdict survives the whole 22.9%-to-48.5% span, so no cell is owed to reach it. But that
span is a problem on its own terms: it is one quantity, on one model, read four ways, and
every future MoE lever gets sized against it. It should be closed, and closing it is one
command.

**`qwen4exp_real_gate --spec-profile <k>` already exists** ("Section-profile ONE spec run at
this K", `prof_section` timers over draft + verify). The three receipts in
`spec/mtp{4,5,6}/spec-profile-k5-nvfp4-*.tsv` are its output, and they are all from a
**66 tok/s** program (963-982 ms for 64 tokens, i.e. ~15.05 ms/token against today's 7.341),
with `moe.dequant` + `moe.expert_gemms` + `moe.idx_gather` at **30% of attributed** on
177.2 calls/round. That is the per-expert grouped executor, which the merged verify path
(`t > 1 && verify_mt_on() && sel_gufuse_on()`, 2 sel launches + t combines per layer)
replaced. **Nobody re-ran it after the program doubled.** So the 48.5% row above is a
measurement of a program that no longer exists, and its 30% per-expert term is most of the
difference from the 26.4% nsys reading.

Cell **A** in `q4e-ep2-cell.sh` is therefore: `--spec-profile 5` on the CURRENT serving
binary, ship seams, under the standard `flock -x` + capacity guard. One short exclusive
hold, no checkpoint mint beyond the one already on the box, no code.

**Falsifiable prediction, so the cell can fail:** the routed-expert sections
(`moe.sel_grouped` + `moe.sel_bf16` + `mtp.moe` + whatever `moe.dequant`/`moe.expert_gemms`
survive) come out at **25-35%** of attributed, with the per-expert terms collapsed from 30%
to near zero, reconciling with the nsys 26.4%. If instead they come out **above 63.8%**, the
verdict is wrong and this lane reopens. If they land in between (35-63%) the verdict still
holds but the EP2 ceiling moves up inside the 154-180 band, which is the number a funding
decision would want.

Note also `qsa.sdpa` at **1.2%** in those shallow spec profiles against **29.7%** in
PROFILE-C0's 100k decode: not a contradiction, it is the QSA selection budget (512 blocks x 4
= 2,048 tokens) being unsaturated at a 64-token context and saturated from ~8k on. Any
section share quoted for this model has to name its DEPTH as well as its `t`, and cell A's
receipt does.

**And a gap found while writing that cell, stated rather than papered over: there is no
instrument in this repo that section-profiles a spec round AT DEPTH.** `--spec-profile` runs a
fixed 64-token spec loop near the top of `main`, BEFORE the `--ladder` block, so
`--ladder 32768 --spec-profile 5` emits a shallow profile and then a separate ladder , 
silently answering a different question than the flags read like. There is `--ladder-spec` and
`--ladder-spec-shape` but no `--ladder-spec-profile`. So **every routed-expert share in this
lane, including the verdict's, is a shallow-to-mid figure**, and the deep composition of the
spec round is unmeasured. The fix is small (run the existing `prof_section` timers over the
rung's spec loop rather than the shallow one) and it is the prerequisite for any deep MoE-share
number. It does not move the verdict, the section would have to more than double its share at
depth to reach 63.8%, and the depth-sensitive sections that grow (`qsa.idx_host`,
`ple.host_ngram_gather`) are NOT expert work, so growing them moves `f` DOWN, not up.

## 6. What to fund instead, from the same receipts

Ranked by measured share of the round, since that is the only ranking that survived this
lane:

1. **The dense trunk matvecs, 37% of GPU kernel time** (`qmatvec_bf16w_f32` 18.8% +
   `qmatvec_bf16w_mt_f32` 18.3%), against the sel section's 15.6% in the same nsys summary.
   Untouched by any lane. DOWNSEL section 7 already points here: "If the 200 target is the
   goal, that is where the next lane should price a lever, not here."
2. **`qsa.sdpa`'s grid, 29.7% of the decode token, and bit-identically fixable.** It
   launches `grid = (n_head, T) = (24, 1)`, `block = 128`: **24 CTAs on a 188-SM card**,
   with phase 2 running on `tid == 0` while 127 threads idle, and the binding constraint
   measured from the cubin is the grid, not registers or smem. `kvsplit3` (PROFILE-C0
   section 5) is fully specified and takes phase 1 to 408 CTAs with every accumulation order
   unchanged, so the bar is byte-identity rather than a band. This is the single largest
   receipted single-card lever in the lane and it is designed but not built.
3. **`ple.host_ngram_gather`, 14.1% and GROWING** (+0.048 ms/k, ~12.7 ms extrapolated at
   262k). A host gather on the critical path.
4. **`selgroup`** (`spec/downsel`, PR #27), 6.42% ceiling, +3.0-4.0% measured on the box (K=5, serving caches), default ON
   since 2026-09-02 (PR #56). Smaller than EP2, and it stacks with it.
5. **EP2 itself, honestly sized: a 154-180 tok/s ceiling (centre ~157) against today's
   136.2**, i.e. +13% to +32% at a perfect free split, and needing a t-generic TP2 verify
   path plus a device-side route split before it can touch the spec path at all, neither of
   which exists. Worth doing when two cards are already committed for another reason. Not
   worth doing to chase 200.

## Corpus rows (for `agent-knowledge/gpu/`, darklanes side)

```
VERDICT:q4e-ep2-cannot-reach-200 | scope: qwen4_exp (Qwen3.8-Flash-Next-NVFP4) NVFP4 spec decode at K=5 on TWO cards, RTX PRO 6000 Blackwell SE | two-card expert parallelism CANNOT reach the 200 tok/s target and the margin is ~2x: from the measured 136.2 tok/s / 31.94 ms round, 200 needs 10.19 ms out, so a 2-card split of fraction f saves f/2 and needs f >= 63.8% of the round to be routed-expert work. Every attribution in the repo is far below that - 22.9% (MOEUNION), 26.4% (DOWNSEL from mtp10 nsys, 8.428 ms = sel_gu_silu 5.328 + sel_v3 3.100, 2160 instances each), ~33% crediting MTP MoE + shared expert, 48.5% from the STALE mtp4/5/6 spec-profiles (a 66 tok/s program whose per-expert dequant executor was 30% of attributed and has since been replaced). EP2's two-card ceiling is therefore 154-180 tok/s at a perfect free halving, centre ~157. The verdict does NOT depend on settling that span. The floor is dense-trunk + attention work (dense matvecs 37% of kernel time vs the sel section's 15.6%; qsa.sdpa 29.7% of the 100k decode token, grid-bound at 24 CTAs). Corrects ROUND-BUDGET-COMPOSITION.md, which called the 15.1 ms flat term a "weight-read floor" and sized EP2 as halving it: the section scales with SLOT COUNT, so EP2 halves ~5.6% of the flat term and ~32% of the 2.10 ms/row slope - a SLOPE lever, not a floor lever | keywords: EP2, expert parallel, spec decode, round budget, floor, ceiling, 200 target, Amdahl, sel matvec | src: memra research/qwen4exp-bringup-20260829/ep2/EP2-DESIGN.md | since: 2026-09-01

TRAP:section-shares-need-t-and-depth | scope: any quoted section share for qwen4_exp (and any spec-decoding model with a bounded attention selection budget) | a section share is meaningless without BOTH its `t` and its DEPTH, and this model spans two orders on one section between them: qsa.sdpa is 1.2% of the K=5 spec profile at a 64-token context and 29.7% of the t=1 decode token at 100k, because the QSA selection budget (512 blocks x 4 = 2,048 tokens) is unsaturated below ~8k and flat above it. The same model's routed-expert share reads 22.9% / 26.4% / 48.5% across four in-repo attributions differing by t, depth, program era and instrument. Re-run the section profile after any program-level speedup: the mtp4/5/6 spec-profiles were never re-cut after the program went 66 -> 136 tok/s and their 30% per-expert-dequant term describes an executor that no longer runs | keywords: profile, section share, attribution, depth, t, selection saturation, stale receipt | src: memra research/qwen4exp-bringup-20260829/ep2/EP2-DESIGN.md | since: 2026-09-01

TRAP:ep-split-dispatch-not-residency | scope: qwen4_exp TP2/EP2 and any "split the expert bank across cards" reading of it | the shipped two-card split partitions DISPATCH, not RESIDENCY: card 0 keeps the complete 512-expert bank and addresses it by global id while card 1 holds an ADDITIONAL gathered half, so the pair stores 1.5x the bank (single-card post-load 89,971 MiB; TP2 post-load 0, 92,755 | 1, 40,211, card 0 pays +2,784 MiB). "Each card reads half the expert bytes" is not what runs; "each card runs half the (token, expert) slots" is, which is the lever that works per KNEE:q4e-sel-slots-not-bytes. Consequences: no VRAM relief on the binding card, card 1 FLAT at 43,603 MiB while card 0 absorbs all depth growth, and TP2 OOMs below 100k where one card reaches ~731k | keywords: EP, expert bank, residency, dispatch, VRAM, depth regression, TP2 | src: memra research/qwen4exp-bringup-20260829/ep2/EP2-DESIGN.md | since: 2026-09-01

LAW:moe-seam-is-the-narrow-stream | scope: any MoE-parallel vs dense-parallel sizing on a gated-residual / hyper-connection architecture (qwen4_exp hc_count 4, wide stream 10240, MoE stream 2560) | price a cross-card seam by WHICH STREAM it crosses, not by the parallel mode's name: a MoE-only split crosses the mixed-down stream once per layer (48 x 2 x t x 2560 x 4 B = 5.90 MB per t=6 chunk) while a dense column-parallel split crosses the WIDE residual at every sublayer (~47.2 MB, ~8x, and twice per layer on the critical path). That 4x width plus 2x crossing-count ratio is why a MoE-only two-card seam is affordable on PCIe where a dense one is not, the argument is the stream width, never "EP avoids allreduce": qwen4_exp's own MoE join IS a dense zero-padded partial-sum ADD with no compaction | keywords: EP, TP, PCIe, seam width, wide residual, hyper connections, allreduce, partial sum, join | src: memra research/qwen4exp-bringup-20260829/ep2/EP2-DESIGN.md | since: 2026-09-01

TRAP:sm-issued-peer-access-sysmem-staged | scope: any cross-card byte budget on a peer path built from UVA peer load/store KERNELS rather than copy-engine cudaMemcpyPeerAsync (qwen4_exp launch_push / q4e_push_f32; the tp.rs NVFP4 EP staging kernels) | on direct-attach / NODE-topology hosts the driver stages SM-issued peer access THROUGH SYSTEM MEMORY by default, measured ~15x slower than NCCL, while nvidia-smi topo -p2p r reads OK and cudaMemcpy looks healthy, so neither of the usual probes detects it and a byte budget computed at link bandwidth can be off by 15x (5.90 MB -> 0.118 ms healthy vs ~1.79 ms staged). Only a simpleP2P-class KERNEL peer read detects it; glm5's transport stayed copy-engine-only specifically to dodge this | keywords: P2P, peer access, UVA, SM-issued, SysMem staging, PCIe, NODE topology, byte budget | src: memra research/qwen4exp-bringup-20260829/ep2/EP2-DESIGN.md, crates/memra-engine/src/tp_transport.rs, research/glm53-flash-bringup-20260827/tp-transport-20260901/LANE.md | since: 2026-09-01
```
