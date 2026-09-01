# glm5 MoE-VERIFY LOCALITY — the 36.4%-of-GPU pair, reconciled against its byte floor
### lane/glm5-moe-loc, 2026-08-31

Base: `origin/lane/glm53-flash-bringup` @ **9158ea5d5** (the matvec-doors flip: T/X/K/W default
ON, M default OFF-refuted). Worktree `~/projects/wt-glm5-moe-loc`, branch `lane/glm5-moe-loc`.
A predecessor lane died to provider errors before any commit; nothing to take over, opened fresh.

Charter: the vrows MoE pair (`moe_gate_up_preclamp8_q8_rows` + `moe_down8_fma_q8_rows`) is
**9.86 ms/round = 36.4% of winner-shape GPU** and is NOT occupancy-bound (door M refuted at
0.9959x). It is the biggest sized item on the census; the 100 tok/s bar needs 1.42x on the
70.458 tok/s ship winner.

Rig law: every number measured here is a 5090 exactness/counter receipt (`flock
/tmp/memra-5090.lock`, `NVIDIA_TF32_OVERRIDE=0`); every ms is arithmetic against banked BOX
constants and labelled as such. A separate box window prices every door.

---

## 1. STEP 0 — THE RECONCILIATION (written before any code)

### 1.1 The geometry, from the artifact

`glm-config.json` text_config: hidden 4096, `moe_intermediate_size` 2048, 288 experts, top-8
routed + 1 shared, 42 MoE layers. NVFP4 = 0.5625 B/element.

    one expert projection slab = 4096 x 2048 x 0.5625 B = 4,718,592 B = 4.5 MiB = 4.7186 MB

This reproduces the banked MEASURED figure exactly — the decode-gap attribution's
`ATTRIBUTION.md` §2 row "routed experts, NVFP4 | 1008 blocks x 4.72 MB | MEASURED
accesses/token = 1008.0 exactly = 42 x 8 x 3" — so the slab size is a receipt, not a derivation.

Per (row, expert) VISIT: `gate_up` reads gate + up = **9 MiB = 9.4372 MB**; `down` reads
down = **4.5 MiB = 4.7186 MB**.

### 1.2 The measured pair, and the row count

Source: `../mv-battery-20260831/receipts/c5/` (the WINNER census: 239p + 192c, **73 spec
rounds**, acc-rate 0.696, duration-bounded nsys).

| kernel | inst | total ms | avg us | inst/round | ms/round |
|---|---|---|---|---|---|
| `moe_gate_up_preclamp8_q8_rows` | 3066 | 479.1 | 156.3 | 42.0 | 6.563 |
| `moe_down8_fma_q8_rows` | 3066 | 240.3 | 78.4 | 42.0 | 3.293 |
| **pair** | | | | 84.0 | **9.856** |

3066/73 = 42.0 = one launch pair per MoE layer-call per round, as designed.

Verify rows per round, by the census's own stated method (the c8-ship census derives
"drafted/round ~= 2.5 ⇒ t = K+1 ~= 3.5" the same way, and this formula reproduces its numbers):

    tokens/round      = 192 / 73        = 2.6301
    accepted drafts   = 2.6301 - 1      = 1.6301      (one bonus token per round)
    drafted/round     = 1.6301 / 0.696  = 2.3422      (acc-rate = accepted/drafted)
    verify rows t     = 2.3422 + 1      = 3.3422

    visits per layer-call = 8 x t = 26.738   (near-disjoint = no dedup, the receipts' assumption)

### 1.3 The floor, three ways — and the pair is ON it

**(a) Achieved bandwidth per kernel.**

| kernel | bytes/call | measured us | achieved | vs 1.79 TB/s peak | vs the 87% bound (1.5573) |
|---|---|---|---|---|---|
| `gate_up` | 26.738 x 9.4372 MB = 252.33 MB | 156.3 | **1.6144 TB/s** | **90.2%** | **103.7%** |
| `down` | 26.738 x 4.7186 MB = 126.16 MB | 78.4 | **1.6092 TB/s** | **89.9%** | **103.3%** |

The bound constant is this lane family's own: 1.79 TB/s GDDR7 spec peak, and **87% = 1.5573
TB/s** as the achievable-efficiency bound, the number q38 proved reachable on this card class
(`QUIRK:qwen38:decode-weight-bound`, the lm_head receipt every matvec lane cites as its target).
The pair is not 57-64% of that bound. It is **above it, by 3.3-3.7%.**

**(b) Per round, against both bounds.**

    pair bytes/round = 42 x (252.33 + 126.16) MB = 15,896 MB = 15.896 GB
    at 1.79   TB/s (spec peak, never observed on this card class) =  8.881 ms   -> measured is 111.0% of it
    at 1.5573 TB/s (the banked 87% achievable bound)              = 10.207 ms   -> measured is 0.351 ms FASTER

**By the lane family's own bound constant there is NEGATIVE headroom in this pair.** The
remaining 0.975 ms/round to spec peak is the 10% no kernel on this card class has ever
recovered on any shape.

**(c) The independent proof that time IS bytes** (no bound constant needed):

    avg_us ratio   down / gate_up = 78.4 / 156.3   = 0.5016
    byte ratio     down / gate_up = 4.7186/9.4372  = 0.5000
    agreement                                        0.3%

The two kernels have the SAME block count structure, the SAME one-warp `warp_reduce_sum`
butterfly (verified in source: neither has `__syncthreads` or shared memory), and the SAME
per-pair epilogue cost — but `down` reads half the bytes. Any fixed per-block, launch, or
occupancy overhead would inflate `down`'s share and pull the ratio ABOVE 0.5. It sits at 0.5016.
**Time is bytes to within 0.3%; there is no non-byte term left to remove.**

### 1.4 Why door M was refuted, stated as a mechanism

Door M raised warp occupancy from <=67% to 100% and read 4 adjacent expert rows per block. Both
are real improvements to a *latency-bound or issue-bound* kernel. A kernel already at 90% of
theoretical DRAM peak has no idle bandwidth for extra resident warps to fill, and its reads were
already fully coalesced, so the packing bought nothing and paid the extra register/index
arithmetic: **0.9959x is the pack's cost with no bandwidth to reclaim.** The prediction (-1.5 to
-3.1 ms) was computed against the 57-64%-of-bound figure — which was measured on a DIFFERENT
program (see next).

### 1.5 The 57-64% figure was measured on the program the vrest lane already deleted

The decode-gap `ATTRIBUTION.md` §3a row is `moe-epilogue ... 4701 us/tok ... expert-read bytes
roofline 3.0 ms, so ~57-64% efficiency`. That is the **t=1 plain** shape, which ran 49 launches
per token-layer (2,058/token) with ~8 KB of work per launch: latency-bound, 4.76 GB / 4.70 ms =
**1.013 TB/s = 57% of peak**.

The vrest lane's vrows port collapsed that to 2 launches per layer-call over 26.7 pairs of
parallel work, and that is what took the pair from 1.01 TB/s to **1.61 TB/s**. Nobody priced it,
because door M's prediction and the WINDOW's "36.4% of GPU" headline both inherited the plain
arm's efficiency number. **The vrest port already banked this pair's efficiency win.** 36.4% of
GPU is a share-of-a-shrunken-denominator artifact (the total fell when doors T/X/K landed), not
a slack signal.

### 1.6 VERDICT: measured IS the floor. Pivoting to the tails.

Per the charter's own instruction ("If measured IS the floor, say so and pivot ... Do not chase
bytes that must be read"), the efficiency class on this pair is CLOSED. What remains:

**The only surviving byte lever is cross-row expert dedup, and its size is unmeasured.** With
the pair bandwidth-saturated, saving = the share of the 26.738 visits/layer that re-read a slab
another verify row already read. Independent-routing bound:

    P(expert e unselected by one row) = 1 - 8/288 = 0.972222
    E[distinct over t rows]           = 288 x (1 - 0.972222^t)
    at t = 3.3422                     = 288 x (1 - 0.910136) = 25.881
    repeat fraction                   = 1 - 25.881/26.738    = 3.21%
                                      = 0.316 ms/round       = +0.86% ship tok/s

Sensitivity (round wall on the winner = 2.6301 tok / 70.458 tok/s = **37.328 ms/round**):

| repeat fraction across the t~3.34 verify rows | Δms/round | ship tok/s | ratio |
|---|---|---|---|
| 3.2% — independent routing (the bound above) | -0.32 | 71.06 | 1.009x |
| 10% | -0.99 | 72.34 | 1.027x |
| 20% | -1.97 | 74.38 | 1.056x |
| 33% | -3.25 | 77.18 | 1.095x |
| 70.1% — structural ceiling (all rows route identically: distinct = 8) | -6.91 | 86.46 | 1.227x |

The lever spans **+0.9% to +23%** depending on one routing property nobody has measured. That is
not a kernel campaign to start on speculation — it is a counter to add. **LANDED: the
`MEMRA_MOE_VROWS_DEDUP_STAT` instrument** (§3), which reports `visits` and `distinct` per
layer-call so the next box window prices the lever for the cost of a host bitset.

(The existing `moesd` capture is NOT this measurement: it unions experts across a whole capture
window and across rounds, so its `union_size/assignments` answers "what is the resident working
set", not "what do the t rows of ONE layer-call share". Different question, different number.)

Also note the mechanism constraint, for whoever builds the dedup kernel: the per-layer-call
working set is 26.738 x 14.156 MB = **378 MB against ~128 MB of L2**, so a repeat visit is only
cache-served if it is SCHEDULED within about a third of the call. Expert-major pair ordering
(the pairs are `p = tok*n_used + j`, dense slot-major today) is therefore NECESSARY for dedup and
worth exactly what the overlap measurement says — no more.

### 1.7 REFUTED ON GEOMETRY, before code: gate_up+down fusion

The charter's hypothesis was "gate_up+down fusion per expert-visit (halves slab re-reads if down
re-reads gate_up's slabs)". **`down` does not re-read `gate_up`'s slabs.** gate, up and down are
three DISTINCT 4096x2048 weight matrices: `gate_up` reads gate+up (9 MiB), `down` reads down
(4.5 MiB). Byte overlap is exactly zero, so fusion halves nothing.

What fusion would actually save: the intermediate activation round trip (26.738 x 2048 f32 =
219 KB written + read per layer-call = 18.4 MB/round both directions, **~0.02 ms** at the pair's
own 1.61 TB/s), one `quantize_q8_1` launch and one kernel launch per layer-call (42 x 2 x 2.049
us box launch constant = **~0.17 ms**, partially hidden). Sub-0.2 ms class — and it forfeits the
`down` kernel's slot-ordered `__fmaf_rn` chain isolation, which is the standing vrest gate-4 bit
bar. **REFUSED.**

---

## 2. THE TAILS, SIZED — and the HtoD tail fully attributed

Winner round budget: wall **37.328 ms/round**, GPU-busy 27.09 ms (9.856 / 0.364), so
**10.24 ms/round is non-GPU wall**. To reach 100 tok/s the round must fall to 26.30 ms:
**-11.03 ms/round needed, more than the entire MoE pair.** This lane is not the 1.42x and no
single lane is; the arithmetic is stated so nobody re-derives it optimistically.

### 2.1 Tail 1 — HtoD 71.6 calls/token was UNATTRIBUTED. It is now named, all 156/round.

A full source census of every HtoD site reachable in a glm5 spec decode round on the serving
recipe (PP-3, VERIFY_BATCH+HYPER_BATCH ON, FUSED_EPI off, peer probe PASS so PP boundaries take
`memcpy_peer_async`, not the host bounce). There is no memra HtoD wrapper; everything bottoms
out in cudarc `memcpy_htod` / `clone_htod`.

| subsystem | site | what | bytes | calls/round |
|---|---|---|---|---|
| **MoE vrows staging** | `hybrid_forward.rs` `moe_vrows_pairs_q8` | plane-major expert POINTER table | 768 B | **42** |
| **MoE vrows staging** | same | matching SCALE table (gate/up macro, `w*down` macro) | 384 B | **42** |
| **MoE shared expert** | `hybrid_forward.rs` `moe_shexp_add` | `vec![1.0f32; t]` — a CONSTANT, re-uploaded | 16 B | **42** |
| MLA latent plane | `hybrid_forward.rs` `mla_attn_cached_pre_wo` | `len_d` i32 mirror | 4 B | 11 |
| MLA rollback | `glm_spec.rs` `glm5_rollback_layer` | `len_d` reset (fires even when `keep == rows`) | 4 B | 11 |
| verify walk | `glm_spec.rs` `Glm5VerifyPos::new` | `[t]` position vector, once per PP stage | 16 B | 3 |
| verify walk | `glm_spec.rs` `glm5_verify_rows_ppn` | HOST embed gather of the t rows | 64 KB | 1 |
| drafter | `glm_spec.rs` `glm5_dflash_round_drafts` | tap-feature rows -> drafter ctx | ~210 KB | 1 |
| drafter | same | HOST embed gather `[anchor, MASK x15]` | 256 KB | 1 |
| drafter | `dflash.rs` `ingest_ctx` / `forward_round` | ctx + block positions | 12 B / 64 B | 2 |
| KDA (34 layers) | `kda.rs` | — | — | **0** |
| hyper-connections glue | `hyper.rs` | — | — | **0** |
| head / topk / greedy accept | | — | — | **0** |
| **total** | | | ~531 KB | **156** |

`156 / 2.6301 tok/round = 59.3 HtoD/token`; the sampled route adds 3-4 (`dflash.rs` accept
rows/ids/residual, all <1 KB). The gap to the measured **71.6/token** closes with two named
contributors, exactly as the census note said it folds prefill:

1. **PMIN0.7-truncated rounds.** When the confidence gate emits zero drafts, `t = 1`, so both
   `glm5_verify_batch_on() && t > 1` and `vrows_fires` (`t >= 2`) are FALSE. That round costs
   ~75 HtoD for ONE token (2 pos x 3 stages + embed + 11 MLA-walk + 11 MLA-rollback + 42 shexp
   + 4 drafter) and pulls the per-token mean up.
2. **Prefill staging**, ~420 calls over the 239-token prime (grouped-prefill sigmoid MoE is 9
   HtoD per MoE layer x 42 = 378, plus pos uploads and the shared-expert add), amortized over
   192 completion tokens = **+2.2/token**.

**150 of the 156 calls are under 1 KB, all pageable (plain `Vec`), totalling ~49 KB.** This is
a pure driver-call-count and pageable-copy tail, not a bandwidth tail.

**The killable share, and what landed:**

| class | calls/round | share | disposition |
|---|---|---|---|
| vrows ptrs+scl — a device->host->device ROUND TRIP | 84 | 53.8% | **door D, LANDED (§3)** |
| shexp `g = 1.0` — a re-uploaded CONSTANT | 42 | 26.9% | **door H, LANDED (§3)** |
| `len_d` 4-byte scalars — `Engine::i32_set_k` already exists and its own doc calls `set_i32_one` "poison mid-round" | 22 | 14.1% | **door H, LANDED (§3)** |
| host embed gathers (verify + drafter) — `HybridModel::embed`'s device gather already exists and the PRIME path uses it | 2 | 1.3% | named follow-up 1 (320 KB -> 80 B) |
| drafter tap round trip (D2H then H2D of the same features) | 1 | 0.6% | named follow-up 2 (largest bytes, hardest: the drain hops PP stage engines) |
| genuinely host-origin (positions, accept ids) | 5 | 3.2% | irreducible |

**Doors D + H remove 148 of 156 HtoD calls per round = 94.9%**, plus 42 full
`cuStreamSynchronize` and 84 DtoH.

### 2.2 Tail 2 — bf16-mmv, 9.522 ms/round at 69% of peak (the biggest GPU-ms lever left)

Census: `matvec_bf16_f32acc_x1_tcols` 10,877 inst / 695.1 ms / avg 63.9 us = **149.0 calls and
9.522 ms per round, 24.7% of GPU.** Composition per round (matvec LANE §1 + the door T/X census
receipt that x4_rows is gone and all tcols traffic runs the x1 form):

    136 kda proj   x 67.1  MB = 9,125.6 MB
     11 indexer qb x 12.6  MB =   138.6 MB
      1 verify head (t~3.34) = 1,269.0 MB   (weight read ONCE, tcols)
      1 drafter head (t=15)  = 1,269.0 MB   (door T's win: read once, not 15x)
    total                      11,802.2 MB = 11.80 GB / round

    achieved = 11.802 GB / 9.522 ms = 1.2394 TB/s = 69.2% of peak, 79.6% of the 87% bound
    at the 87% bound                = 7.578 ms  ->  -1.94 ms/round available

Attributing the two head calls at their own demonstrated 1.43 TB/s (1.775 ms) and the indexer at
~1.0 (0.139 ms), the residual kda trunk is **7.608 ms for 9,125.6 MB = 1.1995 TB/s = 67.0% of
peak**; at the bound, 5.860 ms — **-1.75 ms/round from the kda trunk alone.**

**WHY it is still at 67% after door X — a source finding, not a guess.** At the kda shape
(`in_f` 4096, `out_f` 8192, `mmv_block()` = 128 threads):

    main loop trips per thread = in_f / (blockDim.x * 8) = 4096 / 1024 = 4
    reduce tail per block      = t SEPARATE 128-thread strided trees, 7 levels each, one
                                 __syncthreads per level plus a leading and a trailing one
                               = 9 barriers per token column, ~30 barriers at t = 3.34

**~30 block-wide barriers against a 4-iteration main loop.** The kernel is barrier/tail-bound,
not DRAM-bound, which is why raising DRAM efficiency looks stuck. Door X's own comment named the
symptom ("DRAM idles in the reduce phases") and fixed the WAVE COUNT (1.0265x) — but the tail is
per-block work, so 4x the blocks pays it 4x more often.

**Door R, designed and sized, NOT built here** (named follow-up 3, the next lane):
`MEMRA_BF16_TCOLS_RED_FUSED` — (a) one shared region per token column (`red[t*blockDim]`) so the
t trees share ONE barrier sequence instead of t; (b) levels `s <= 16` are intra-warp (after
`s = 32` only lanes 0..31 hold live partials) so they become `__shfl_down_sync` with the SAME
pairing and the SAME addition order, zero barriers. Barriers per block **9t (~30) -> 3**.
Bit-identical by pairing preservation — the same bar doors T and X already carry, and it applies
to all three tcols twins (x1, x4, tcols16), so the drafter head's extreme case (t=15 = **135
barriers** against 4 loop trips) gets it too. Predicted **-1.0 to -2.0 ms/round**, capped by the
87% bound at -1.94.

Why it is not in this PR: it is a new kernel in the family whose doors flipped to default ON
hours ago (the ship winner's 1.1288x rides them), so it needs its own bit gate across t=2..8 and
t=9..16 on both grid forms plus a re-run of the standing tcols gates. That is a lane, not a
rider, and this lane's own doors already needed the full battery.

### 2.3 Tail 3 — ~3,849 allocs/round (door-W extensions)

Doors D and H take a bite here too: door D removes 2 host `Vec` allocations per MoE layer-call
(84/round) and 2 device allocs from the router readback path; door H removes 42 `vec![1.0; t]`
host allocations plus their device `clone_htod` allocations. The remaining ~83% of the churn is
the matvec lane's own named list (hyper glue at m=t, MLA rows internals, the f32 per-column
linear class, the drafter forward, accept/rollback replays) — mechanical extensions of the same
pool, unchanged here.

### 2.4 cublas-f32 (7.6%) — left alone, as instructed

The `lt_ndep` law refuses batching the mixes GEMM: cuBLASLt's reduction split is n-dependent and
plain decode runs m=1, so any m=t batching breaks the byte bar against plain decode by
construction. Re-read for a seam the law does not cover: there is none at this site. The
`batch_size=t` (n=1 per entry) probe the vrest lane named remains refused for the same reason it
was refused there — per-entry equality would be a library-VERSION property, not a structural one.

---

## 3. WHAT LANDED

| door | flag (default) | mechanism | bit bar | receipt |
|---|---|---|---|---|
| **D** device vrows tables | `MEMRA_MOE_VROWS_DEV_TABLES` (**OFF**) | new kernel `moe_vrows_tables_from_sel` evaluates `base + ex*stride` and the three macro lookups where the router's `sel_idx`/`sel_w` already live; the layer routes through `moe_router_sigmoid_topk` (device) instead of `..._host`, so the 2 DtoH + full `cuStreamSynchronize` + 2 host Vecs + 2 pageable HtoD per MoE layer-call all disappear. Macro planes get a resident device mirror keyed `(layer, plane)`, uploaded ONCE. | integer pointer arithmetic is exact; macro lookups are the same f32 table at the same index; `selw[p] * md` is ONE IEEE-754 single multiply of the same two operands in the same order as the host's `w * macro_scale(ex)` (no FMA contraction is possible in a bare product) | §4 gate table |
| **H** htod diet | `MEMRA_GLM5_HTOD_DIET` (**OFF**) | the shexp `g = 1.0` vector becomes a resident ones buffer sliced to t (a CONSTANT stops being re-uploaded 42x/round); the two `len_d` 4-byte scalar stores take `Engine::i32_set_k`, the existing async value-rides-the-kernel-arg twin, instead of `set_i32_one`'s synchronizing pageable copy | identical values written to the identical buffers; `i32_set_k` is stream-ordered exactly like `memcpy_htod` | §4 gate table |
| **S** dedup instrument | `MEMRA_MOE_VROWS_DEDUP_STAT` (**OFF**) | on the host table-build arm, count the pair union's VISITS and DISTINCT experts per layer-call into two counters. NOT a serving door — the measurement that decides whether the dedup lever of §1.6 is 0.9% or 23%. Requires `MEMRA_MOE_VROWS_DEV_TABLES=0` (door D removes the host selection); stated in the FLAGS row. | counters only, no dispatch change | §4 |

**Every default is OFF BY DESIGN**, and the reason is the same one the matvec lane's five doors
carried: no box timing receipt exists yet and the rig is exactness-only. Door D additionally
changes the round's SYNC STRUCTURE (the host stops draining the device 42 times per round and
runs further ahead), which is precisely the class the diet window warned does not always transfer
from counts to wall — so it ships default OFF with a count receipt and the box prices the wall.
Doors D and H each carry a FLAGS.md row in this PR with both arms, the rollback seam and the
receipts pointer.

Fail-closed structure (door D): the arm is pre-decided at the router because it changes HOW the
layer routes, and its predicate is `vrows_fires`' conjuncts plus (a) `sigmoid_router_enabled()`
— `MEMRA_SIG_ROUTER=0` is a full-logit HOST oracle with no device selection to read — and (b)
every host-visible consumer of the selection disarmed: `moesd::capture_active()`,
`hidden_trace`, `MEMRA_MOE_TRACE` / `MEMRA_MOE_STATS` / the other `observe_routes` modes. Each of
those reads `sel_all` and would silently read an EMPTY slice. `promote_worker_h2d` needs no
conjunct: it requires `t == 1` and this arm requires `t >= 2`. Any miss falls closed to the host
readback, and a dispatch-time check errors LOUDLY if door D routed device-only while the vrows
arm did not fire (the silent-wrong-answer class this seam could produce).

One launch path, two table provenances — the fused-epilogue arm's own discipline in this file:
only the table build differs, so the macro fold, the clamp and the kernel pair cannot drift apart
between the arms.

### Predicted ship arithmetic (nothing here is a claim; the box window prices it)

    winner today: 70.458 tok/s, 2.6301 tok/round, 37.328 ms/round, GPU-busy 27.09, non-GPU 10.24

| door | Δ/round (counts, MEASURED-derived) | Δms/round (predicted) | basis |
|---|---|---|---|
| D | -42 `cuStreamSynchronize`, -84 DtoH, -84 pageable HtoD, -84 host Vec, -~2 device allocs/layer-call | **-0.4 to -2.6 (UNPRICED, host class)** | 42 device-wide drains at the box's own drain-refill structure; the launch-diet census attributes the whole wall-minus-GPU gap to launch submit "serialized into ~43 drain-refill cycles by the syncs", and this lane removes 42 of them from the verify walk. Wall transfer of a sync removal is NOT arithmetic — stated as a band, box prices it |
| H | -64 pageable HtoD (42 shexp + 22 `len_d`), -42 host Vec | **-0.1 to -0.4 (UNPRICED, host class)** | 64 pageable driver calls; the two `len_d` stores are SYNCHRONIZING pageable copies mid-round by their own doc, so their removal may exceed their call count |
| S | 0 | 0 | measurement only |
| composed D+H | **-148 of 156 HtoD (94.9%), -42 syncs, -84 DtoH** | **-0.5 to -3.0** | non-GPU wall is 10.24 ms/round, so the band is 5-29% of it |

    if -0.5: 36.83 ms -> 71.4 tok/s (1.013x)
    if -1.5: 35.83 ms -> 73.4 tok/s (1.042x)
    if -3.0: 34.33 ms -> 76.6 tok/s (1.087x)

Against the 100 bar: **1.31x to 1.40x still needed after this lane.** With door R (§2.2,
-1.0 to -2.0 GPU ms) the band moves to ~74-81 tok/s and the bar still needs ~1.23-1.35x, which
lives in a mature TP composition (bare TP-2 does NOT pay — 22.65 tok/s, tp2-battery same day),
the remaining door-W extensions, and the dedup lever IF the §1.6 instrument says it is real.

---

## 4. GATE TABLE

Rig 5090, `flock /tmp/memra-5090.lock`, `NVIDIA_TF32_OVERRIDE=0`, exactness and counters ONLY —
no timing row exists in this lane by the rig law. Runners: `receipts/run-battery.sh`,
`receipts/run-matrices.sh`; per-suite logs in `receipts/`.

### 4.1 The new doors gate — `glm5_moe_loc_doors_gpu` 4/4 PASS

| gate | result |
|---|---|
| door D tables: device build vs the HOST loop, t=2..=8, with LIVE macro planes AND with none — `3*n_pairs` pointers compared as exact `u64` and `3*n_pairs` scales compared BITWISE (`to_bits`), 7 shapes x 2 macro arms | PASS, 0 diffs in all 14 arms. Strides deliberately unequal across planes and non-power-of-two (1408/1408/2816) so a plane mix-up cannot pass by symmetry and a shift-vs-multiply slip shows; the selection includes REPEATED experts across rows and expert 0 so the `base + 0*stride` term is exercised |
| door D reds | BOTH BITE. RED 1 wrong DOWN stride (`sd + 64`) moves the pointer table (without it the equality could hold for a kernel that ignored `sd`); RED 2 dropped macro planes move the scale table (the fold is real) |
| door D pair: the FULL vrows pair driven by device-built tables vs host-built tables, t=2..=8 on minted NVFP4 slabs with a LIVE macro plane and a biting PRE clamp (the vrest gate-4 shape class) | PASS, 0 diffs in all 7 arms; the dropped-macro RED moves the pair output, so the identity is not vacuous |
| door H shexp: `add_scaled_rows_ones` (resident buffer) vs the shipped freshly-uploaded host ones vector through the SAME `add_scaled_rows_f32` kernel, nrows 1/2/4/8/15 | PASS, 0 diffs; the non-ones-scale RED bites at every nrows, so "identical" is not proving the scale went unread |
| door H `len_d`: both arms land the value; counter anchored | PASS — `memcpy_htod` arm lands 4321, `i32_set_k` arm lands 9876, `GLM5_HTOD_DIET_AVOIDED` moves ON and is FLAT OFF |
| instrument S: `vrows_overlap_counts` on planted overlaps | PASS — disjoint 16 visits/16 distinct, two rows sharing 3 experts 16/13, identically-routed rows 32/8. The counting is the whole instrument, so it is gated on plants, not inferred from a live tape |

### 4.2 Standing batteries — ALL SUITES PASS, three arms

Run with `--include-ignored`, not `--ignored`. That is a deliberate change from the sibling
lanes' runners and it MATTERED: `--ignored` silently filtered out every non-ignored test in a
suite, so `kda_fixture_gpu` reported "ok. 0 passed; 3 filtered out" — a suite that ran NOTHING,
recorded as green. Recovered coverage: kda_fixture **0 -> 3**, kpool_indexer 14 -> 18,
moe_epilogue 9 -> 13, mtp_head 5 -> 6, hyper_connections 6 -> 7.

| arm | suites |
|---|---|
| DEFAULT (this lane's doors OFF; the matvec doors T/X/K/W at their shipped default ON — so this is the SHIP arm, the correct control) | glm5_moe_loc_doors 4/4, glm5_matvec_doors 4/4, verify_batch 4/4, tparallel_verify 9/9, spec_session 10/10, dflash_session 10/10, moe_epilogue 13/13, mtp_head 6/6, kpool_indexer 18/18, hyper_connections 7/7, hc_fused_pre 4/4, hc_decode_ws 2/2, mla_decode_split 3/3, kda_fixture 3/3, kda_fused_proj 5/5, kda_fused_proj_bf16 5/5, kda_quant_operand 4/4, mla_gpu_forward 5/5 |
| COMPOSE, doors **D + H ON** (door M pinned `=0`, never unset) | verify_batch 4/4, tparallel_verify 9/9, spec_session 10/10, dflash_session 10/10, moe_epilogue 13/13 |
| STAT, instrument S ON with door D pinned `=0` | verify_batch 4/4, tparallel_verify 9/9, spec_session 10/10, dflash_session 10/10, moe_epilogue 13/13 |

### 4.3 Split matrices — ALL 32 ARMS PASS

| matrix | arms |
|---|---|
| `glm5-spec-ppn-gate` | 8 banked arms (n2 even/split1/split3/streams0/overlap0, n3 even/asym/streams0) + **1 COMPOSE arm with doors D+H ON** = 9/9 |
| `glm5-hyper-ppn-gate` | 10 banked arms + 1 compose = 11/11 |
| `glm5-hyper-batch-gate` | 10 banked arms + 1 compose = 11/11 |
| instrument S arm (`stat-gate/dedup-n3-even.log`) | 1/1 |

**Door D's walk-level engagement receipt** (`ppn-gate/compose-n3-even-doors-DH.log`): the announce
`[moe-vrows-dev-tables] engaged: pointer/scale tables built on device from the router's own
sel/w; the per-layer pinned readback and its cuStreamSynchronize are skipped` appears immediately
before `[glm5-vrows] verify MoE batched across rows: pairs=16 (t=8 x 2)`, and the very next line
is `glm5-spec-ppn gate PASS [W1 verify-walk]: 8 verify rows bit-identical to plain decode under
the split`. The announce is ABSENT on all eight default arms.

**Engagement SCOPE, stated before it can be mistaken for a gap:** door D rides the vrows arm, so
it engages only in the SPEC verify walk. It is structurally silent in `glm5-hyper-ppn-gate` and
`glm5-hyper-batch-gate`, which exercise the plain and B-batched DECODE walks — those compose arms
prove the door does not perturb a walk it must not touch, not that it ran there. Door H's two
substitutions have no announce (they are counter-anchored in §4.1) and DO run in every arm.

### 4.4 Other gates

| gate | result |
|---|---|
| `memra-server` suite | **492/492** |
| `tools/check-flags.sh` | 745 runtime literal reads, no uncovered names, no grandfather list — the three new flags resolve against `docs/FLAGS.md` |
| `cargo clippy --all-targets` | ZERO lints |
| `cargo fmt --all --check` | clean |
| `tools/local-ci.sh --perf` | **exit 0 TWICE** on the final tree (`receipts/local-ci-perf-run{1,2}.log`): correctness ALL GREEN, perf stage **0 fail 0 warn**, qwen9b-plain-short 137.69 / 134.93 tok/s both `[OK]` vs the rolling median. The eight absent-model cells SKIP — the rig's standing shape, stated |

### 4.5 A LOUD FAILURE AND TWO SILENT ONES, found and fixed in-lane

The battery's first pass FAILED `glm5_matvec_doors_gpu::gpu_tcols16_...` on
`flag-off arm moved the wide-tcols dispatch counter — left: 1, right: 0`. Cause: doors T/X/K/W
flipped to **default ON** hours before this lane opened, and those A/B arms leave the variable
UNSET rather than pinning `=0`, so the "OFF" arm now runs the door. Door W's gate was pinned `=0`
in the flip commit; **T, X and K were missed.** Door T failed loudly. Doors X and K did not:

- door X's `y_x4` REFERENCE launch was unset, so it ran the x1 kernel — the gate compared **x1
  against x1** and passed.
- door K's "standing topk" REFERENCE was unset, so it ran the shard split — the gate compared
  **the sharded kernel against itself** and passed.

Two vacuous greens on the exactness gates of two doors that are default ON in the ship config.
Fixed here with a `with_flag_off` helper that pins `=0` and restores the prior value; all three
OFF arms now pin, and the suite is 4/4 with the comparisons live. This is the
exception-lists-need-expiry / loud-failures-fail-quietly shape: the flip changed what "unset"
means and the A/B arms were not swept with it.

---

## 4.6 The S instrument speaks — and what its rig numbers do NOT mean

An instrument that only moves an atomic is useless to a box window, which greps a server log. So
S reports on the first vrows layer-call and every 42 after (42 = the MoE layer count, about one
line per decode round). Receipt, `receipts/stat-gate/dedup-n3-even.log`:

    [moe-vrows-dedup] layer-calls=1   visits=16  distinct=4   repeat=75.00% ...
    [moe-vrows-dedup] layer-calls=43  visits=496 distinct=163 repeat=67.14% ...
    [moe-vrows-dedup] layer-calls=85  visits=666 distinct=298 repeat=55.26% ...
    [moe-vrows-dedup] layer-calls=127 visits=918 distinct=440 repeat=52.07% ...

Absent on every arm without the door.

**These percentages are FIXTURE ARTEFACTS and must not be read as the dedup lever.** The ppn-gate
fixture has `n_active = 4` experts against `n_used = 8`, so `distinct` is CAPPED at 4 while
`visits` is 16 — a 75% "repeat" that is arithmetic on a 4-expert bank, not routing behaviour. The
real bank is 288 experts top-8, where the independent-routing bound is **3.2%** (§1.6). What this
receipt proves is exactly what a rig gate can prove: the counting is correct, the line is emitted,
the door is silent when off, and the arm still passes its exactness battery with the instrument on.

**The number that decides the dedup lever needs ONE box boot** on the real artifact and the ship
recipe with `MEMRA_MOE_VROWS_DEDUP_STAT=1 MEMRA_MOE_VROWS_DEV_TABLES=0`, greppping the
`[moe-vrows-dedup]` lines from the server log. That is a minutes-long instrument boot with no
timing claim attached, and it converts §1.6's +0.9%-to-+23% span into a single decided number. It
is the cheapest high-information cell left on this census.

---

## 5. NAMED FOLLOW-UPS (not built here)

1. **Host embed gathers -> the device gather that already exists.** `glm_spec.rs`
   `glm5_verify_rows_ppn` (64 KB/round) and `glm5_dflash_round_drafts` (256 KB/round) both do a
   HOST embed gather then upload `t*4096*4` bytes. `HybridModel::embed` implements the device
   alternative (resident quantized table + a `t*4`-byte token upload + `embed_gather_device_td`)
   and the PRIME path already uses it. 320 KB/round -> 80 B/round, 2 HtoD -> 2 HtoD but tiny.
2. **The drafter tap round trip** (`glm5_hc_tap` writes features to device, `glm5_tap_drain`
   reads them back, the next round re-uploads them as drafter ctx: ~210 KB down + ~210 KB up per
   round). Largest bytes of any single site. The host hop exists because the drain crosses PP
   stage engines; a stage-aware device path is the fix and it is the hardest one here.
3. **Door R, the tcols reduce-tail fusion** (§2.2) — the biggest PRICED GPU-ms lever left,
   -1.0 to -2.0 ms/round, full design and bit argument written above.
4. **Cross-row expert dedup**, gated on the §1.6 instrument reading materially above its 3.2%
   independence bound. If it does: expert-major pair ordering FIRST (necessary for L2 reuse at a
   378 MB working set against ~128 MB of L2), then a tcols-style twin that reads a shared slab
   once and carries one accumulator per sharing row — the same weight-once pattern doors T and X
   proved on the trunk, applied to the expert planes.
5. `moe_vrows_pack_on()` is an UNCACHED `std::env::var` read per launcher call = 84 environ scans
   plus String allocations per round on the exact path this lane de-hosts, for a door that is
   default OFF and refuted. Caching it in a `OnceLock` would change its documented per-call
   rollback seam, so it needs a FLAGS amendment of its own — named, not taken here.
6. `glm5_rollback_layer`'s MLA arm uploads `len_d` even when `keep == rows` (the KDA arm
   short-circuits). 11 of the 22 scalar stores are unconditional; door H makes them cheap but not
   absent.
7. **Sweep the other lanes' A/B arms for the same default-ON mis-set §4.5 found.** This lane fixed
   doors T/X/K in `glm5_matvec_doors_gpu`, but the flip-to-default-ON class ("unset no longer
   means off") applies to every A/B arm of every door that has ever been flipped. A one-pass audit
   is: for each `MEMRA_*` whose predicate is `!= Ok("0")`, grep its gates for an arm that neither
   sets `=1` nor pins `=0`. Two of the three cases here were SILENT.
8. **The box window should carry a `MEMRA_MOE_VROWS_DEDUP_STAT=1` instrument boot** (§4.6): one
   short boot, no timing claim, and it collapses the dedup lever's +0.9%-to-+23% span to a number.

---

## 6. STATUS LOG

- Lane open 2026-08-31. No predecessor commits at `~/projects/wt-glm5-moe-loc` — opened fresh
  from `origin/lane/glm53-flash-bringup` @ 9158ea5d5 (fetched first).
- §1 reconciliation written from the banked mv-battery c5 census + the artifact geometry BEFORE
  any code change, per the charter's STEP 0. Verdict: the pair is AT its byte floor (90.2% /
  89.9% of theoretical peak, ABOVE the banked 87% achievable bound), so the efficiency class is
  closed; door M's refutation explained as a mechanism; gate_up+down fusion refuted on geometry.
- §2 tails: a full source census named all 156 HtoD calls/round and closed the measured
  71.6/token with two contributors, replacing the census's UNATTRIBUTED line.
- BUILT: door D (kernel `moe_vrows_tables_from_sel` in qmatvec.cu; launcher + resident macro
  mirror + counters in lib.rs; routing arm, `VrowsSel` provenance enum and the fail-closed
  predicate in hybrid_forward.rs; `moesd::capture_active`), door H (`add_scaled_rows_ones`,
  `i32_mirror_store`, the resident `shexp_ones` buffer; two `len_d` sites in hybrid_forward.rs and
  glm_spec.rs), instrument S (`vrows_overlap_counts` + the self-reporting
  `moe_vrows_dedup_report`). FLAGS.md 3 rows + KERNELS.md row and symbol recount in the same
  commit. New gate `glm5_moe_loc_doors_gpu`.
- Gate battery §4 all green same day, including the three mis-set OFF arms fixed in §4.5 (one
  loud failure, two vacuous greens on default-ON doors) and the `--include-ignored` coverage
  recovery in §4.2 (`kda_fixture_gpu` had been running ZERO tests and reporting ok).
- `tools/local-ci.sh --perf` run TWICE on the final tree, both exit 0 (§4.4). An EARLIER run was
  killed and its log DELETED rather than banked: source was edited while it was in flight, so its
  receipt would have straddled two tree states (the rebuild-after-checkout-attribution law).
- Receipt hygiene: `run-matrices.sh` used `tee -a` for its per-matrix summary, so a stale `FAIL`
  line from an earlier typo'd invocation (wrong bin names — the bins are hyphenated) sat in a
  banked `matrix.out` reading exactly like a live failure. The runner now truncates each matrix's
  summary on its first arm, and the banked summaries were rebuilt from the current per-arm logs.
- PUSHED to `origin/lane/glm5-moe-loc`; no self-merge. The box window prices doors D and H, runs
  the §4.6 instrument boot, and owns every flip decision.
