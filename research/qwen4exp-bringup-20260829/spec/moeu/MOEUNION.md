# mtp13: the MoE routed-union gather, priced (`moeu`)

Lane `lane/q4e-moe-union-20260901`. Box sbox-eval (RTX PRO 6000 Blackwell Server Edition,
188 SM, 97,887 MiB, **L2 = 128 MiB**), rig sm_120a laptop 5090 for exactness and harness
validation only. Owner target: 200+ tok/s single-request spec decode; current best
**136.2 tok/s** raw shape, K=5 (`devtwin/ab-devtwin-k5-dt4-raw.tsv`: reps=5,
median 7.341 ms/token, accept 0.861, mean_accept_len 4.41, draft 5.31 ms, verify 26.63 ms).

## The lever, and where it came from

VFUSE.md (mtp12, merged PR #85) killed the fused-verify lever ON PREMISE and handed this
lane the section it proved it could not touch:

> the MoE routed union (the largest GPU section, and a *union-of-experts gather* would read
> each routed expert's bytes once per chunk instead of once per routing token)

Stated precisely: at K=5 the verify chunk is t=6, and with top-10 routing the chunk
dispatches `t*selected` = **60 (token, expert) slots**. If the kernels read each slot's
expert bytes independently, an expert named by two columns is read twice, and a
union-ordered gather would turn `|pairs| x bytes` into `|union| x bytes`.

## 1. Dispatch audit — the premise SURVIVES (unlike vfuse's)

Per LAW:price-the-dispatch-first, read which flag gates each section before building
anything. Here the answer is the opposite of vfuse's: the redundancy is real and the code
says so in as many words.

The merged verify path (`qwen4exp_gpu.rs`, `t > 1 && verify_mt_on() && sel_gufuse_on()`)
builds `sel`/`tok` arrays of `t*selected` entries — one slot per (token, expert) pair,
token-major — and dispatches:

| launch | kernel | grid | block |
|---|---|---|---|
| gate+up+silu | `qmatvec_nvfp4_modelopt_sel_gu_silu_f32` | `(ff/4, slots)` = (160, 60) | 32 |
| down | `qmatvec_nvfp4_modelopt_sel_f32_v3` | `(hidden/4, slots)` = (640, 60) | 32 |
| combine | `launch_axpy_rows_seq_at` x t | per token | — |

Both kernels put **slots on `grid.y`** and each block resolves its weights through
`e = sel[slot]`, addressing `codes + (e*out_f + o)*row_codes`. Nothing groups slots by
expert. The kernel's own doc comment states the consequence without hedging
(`kernels.cu:4663-4667`):

> tok_map (mtp-spec verify): a nonzero pointer maps slot -> token ... ONE launch covers
> every verify column's routed experts (per-slot row program unchanged => bit-identical to
> the per-token launches it replaces; **the weight banks are read once per selected slot
> either way — the launch count is what drops**).

So `verify_mt` + `gufuse` collapsed the LAUNCH count (3t -> 2 + t combines) and left the
per-slot weight reads exactly as they were. The lever has a real surface at the dispatch
level. That is where the good news ends.

## 2. The byte arithmetic, from the kernels' own row layout

A row is `in_f/2` code bytes + `in_f/16` ue4m3 scale bytes (both kernels, verbatim).
qwen4_exp MoE geometry is `experts=512`, `hidden=2560`, expert `ff=640`, `selected=10`,
gate_up `[512, 1280, 2560]`, down `[512, 2560, 640]` (SEMANTICS.md "MoE (L510-527)"):

| bank | rows/expert | row bytes | bytes/expert |
|---|---|---|---|
| gate | 640 | 1,440 | 921,600 |
| up | 640 | 1,440 | 921,600 |
| down | 2,560 | 360 | 921,600 |
| **total** | | | **2,764,800 = 2.637 MiB** |

This reproduces PROFILE-C0's independently derived "10 selected experts x ~2.76 MB =
~27.6 MB per layer, x 48 layers = ~1.33 GB per decode token", so the geometry and the MoE
layer count (48) are cross-checked against a prior lane rather than re-derived here.

Per MoE layer at t=6, nominal (per-slot) reads: gate+up **105.47 MiB**, down 52.73 MiB.

**And that is the first problem: 105.47 MiB fits inside this card's 128 MiB L2.** The whole
chunk's routed gate+up working set is L2-resident at the widest possible union, so the
hardware may already be deduplicating exactly what the kernel re-reads. A traffic argument
made against nominal bytes on this card is not automatically an argument about DRAM.

## 3. The union size — bounded, but NOT safely bounded by geometry alone

The prize is `1 - union/pairs`. With 512 experts and top-10 over 6 columns, **independent**
routing gives

```
E[union] = 512 * (1 - (1 - 10/512)^6) = 57.0 of 60   ->   ratio 0.950, prize 5.0%
```

`moe-union.py` reproduces this exactly on a synthetic independent-routing trace
(9,600 chunks): **union 57.14 / 60, ratio 0.9524, traffic_saved 0.0476**; fresh experts per
column 10.00, 9.80, 9.61, 9.42, 9.24, 9.07.

**That number is NOT the kill, and saying it was would have been this lane's own version of
the mistake it was sent to avoid.** "512 experts top-10 cannot collide" is a statement about
a *uniform* router, and the union ratio turns out to be highly sensitive to hotness and
temporal locality — both of which real routers have. Simulated over the same geometry
(4,000 / 800 / 2,000 chunks per model):

| router model | union of 60 | ratio | prize |
|---|---|---|---|
| independent uniform top-10 | 57.16 | 0.953 | 4.7% |
| Zipf hotness a=0.5 | 55.36 | 0.923 | 7.7% |
| Zipf hotness a=1.0 | 43.99 | 0.733 | **26.7%** |
| Zipf hotness a=1.5 | 31.98 | 0.533 | 46.7% |
| temporal: re-pick 20% of previous column | 48.15 | 0.802 | 19.8% |
| temporal: re-pick 40% of previous column | 38.98 | 0.650 | **35.0%** |
| temporal: re-pick 60% of previous column | 29.52 | 0.492 | 50.8% |

So a real router with moderate hotness or a 40% column-to-column re-pick would put the
traffic prize at 27-35%, not 5%. The geometry argument alone does **not** carry the verdict;
section 4 does, and it holds at every row of this table.

What the prior evidence does say, honestly scoped: PROFILE-10 measured **99.93% of
layer-tokens touching BOTH cards** under an even 256/256 split with the peer taking
**51.40%** of dispatched expert slots, against 99.80% / 50.0% predicted by independence.
That is evidence the router is diffuse *within a token* (which is what a card split sees). It
says nothing about correlation *across* tokens, which is exactly the axis the union depends
on — so it is corroboration, not proof.

**The real union for this artifact is UNMEASURED, and this lane did not spend box time to
measure it.** The sibling `qA5` queue was already collecting exactly the right input —
`moe-{thinkon,thinkoff,raw}-32768.trace` with `MEMRA_Q4E_ROUTER_AUDIT=1 MEMRA_MOE_TRACE=...`
under `--spec-k 5`, whose t=6 lines are 60 ids token-major — so the correct move was to read
its output rather than duplicate the run. The box was terminated before that queue reached its
trace cells (see 4b), so the number is owed, not gathered.

`moe-union.py` is the reader when a trace exists (it takes the t>1 half of the trace that
`moe-hit-rate.py` deliberately skips), and the number is worth banking then — it is the
expert-placement lane's input too. The verdict below does not wait on it, because section 4
makes every row of the table above land in the same place.

## 4. The payoff curve — measured on the shipped kernels, no kernel written

Rather than argue, price it. A union gather changes exactly one quantity: distinct experts
touched. Per-slot arithmetic, slot count and launch geometry all stay. So run the SHIPPED
kernels at a fixed 60 slots and sweep only how many distinct experts those slots name —
the `slots=60, union=U` row **is** the idealised union gather's cost, measured, with no
rewrite (`moe_union_cost_probe` + `moe_union_probe`, synthetic banks of the serving
geometry, no checkpoint, ~1.3 GiB).

Probe honesty, stated because a timing arm that looks like a correctness arm is how a wrong
number gets quoted later:

- **Synthetic banks, and no correctness claim anywhere.** Sound for a traffic/latency probe
  and nothing else: the NVFP4 lane program is branch-free and data-independent (LUT
  extract, fixed shfl tree), so bytes decide addresses and never control flow. Scale bytes
  are confined to a mid ue4m3 range so the f32 chain stays in normal range. No output is
  compared to an oracle, and no oracle arm is added to the tiny fixture gate — because this
  lane adds no kernel and changes no serving path. If the verdict had been GO, that gate arm
  would be the first thing built.
- Gate and up get **separate** allocations. Aliasing them would halve the distinct bytes and
  silently turn the sweep into a cache-hit measurement.
- Expert ids are spread across the bank by a fixed stride, so the sweep measures
  distinct-byte count and not address clustering.
- **Arms are interleaved rep by rep**, not swept as contiguous blocks, and every arm reports
  its own spread (LAW:interleaved-ab). Rep 0 of every arm is a warmed throwaway (the
  `scan_warm` lesson); per-arm statistic is the median rep. See the pass table below for what
  this fix was worth — it was not cosmetic.
- Box rows are taken under `flock -x /tmp/q48fn-measure.lock` held around the whole
  invocation, with the capacity+idle guard INSIDE the hold (the vfuse lesson: checking cards
  then blocking on flock is a race, and nvidia-smi free is not driver free).

### 4a. Rig rows — WITHIN-RUN RATIOS WITH SPREAD, never absolute numbers

**This section carries the verdict, so its relationship to LAW:rig-gpu-exactness-only has to
be stated exactly rather than waved at.** That law exists because the 5090 laptop throttles,
which makes (a) absolute figures and (b) cross-box or cross-run comparisons invalid. What is
quoted here is neither:

- **Not quoted:** any absolute us/launch, any tok/s, any comparison against a box number, any
  claim about how fast this section runs. The us columns below are printed only so the ratios
  can be checked.
- **Quoted:** the RATIO between arms measured *in one process, in one residency, with the arms
  interleaved rep by rep*, each arm reporting its own spread. Throttling that afflicts one arm
  afflicts all of them at the same point of the drift curve — that is precisely what
  interleaving buys, and it is why passes 1 and 2 (swept) were thrown away.
- **And the spread is reported BECAUSE it is large.** At 8.6-11.7% per arm it is the honest
  bound on what this card can resolve, and the verdict is built on the effect being SMALLER
  than that bound, not on any particular value inside it. A noisy instrument can support
  "this effect is below X"; it cannot support "this effect is exactly Y", and nothing below
  claims the latter.

The box cell (4b) is owed to tighten this. It is not owed to decide it — see 4b for why the
direction of the remaining uncertainty is already known.

They matter for one structural reason beyond validation. **The rig is the lever's BEST CASE,
and it can be said in one line of hardware:**

| | SMs | L2 | t=6 gate+up working set (105.5 MiB) |
|---|---|---|---|
| rig, RTX 5090 Laptop | 82 | **64 MB** | does NOT fit — duplicate slot reads must go to DRAM |
| box, RTX PRO 6000 Blackwell SE | 188 | **128 MB** | **FITS** — the L2 can already dedup them |

So the rig is where a union gather has the most to win, because the cache cannot do the
lever's job for it. **Prediction the box cell tests: the box shows LESS union sensitivity
than the rig, not more.** If the box curve is flatter, the lever is dead on both cards; the
only way the lever survives is if the box is somehow MORE byte-sensitive than a card whose
L2 cannot hold the working set at all.

#### The measurement had to be fixed before it could be trusted — and the fix shrank the lever

Three passes were taken and only the third is quoted. The progression is recorded because it
is the methodological point of this lane, not throat-clearing:

| pass | build | reps | arm order | union 60 -> 10 |
|---|---|---|---|---|
| 1 | debug | 3 | swept (contiguous per arm) | 1.115x |
| 2 | release | 25 | swept (contiguous per arm) | 1.273x |
| 3 | release | 25 | **INTERLEAVED rep by rep** | **1.101x** |

Passes 1 and 2 violated LAW:interleaved-ab: each union size ran as a contiguous block, in
monotone order, smallest union first. In that order any clock or thermal drift over the run
accumulates onto the LARGER unions and reads as a union effect — inflating the payoff, which
is the direction that makes a dead lever look alive. Pass 2's 1.273x is that artifact;
interleaving the arms rep by rep put every arm at every point of the drift curve and the
effect fell back to 1.101x. **Passes 1 and 2 are superseded and are quoted nowhere else in
this document.** The probe now interleaves by construction.

#### The quoted rows (`rig-moeu-t6-interleaved.tsv`, `--release`, reps=25, arms interleaved)

| slots | union | gu us | down us | section us | gu spread | vs union=60 |
|---|---|---|---|---|---|---|
| 10 (t=1) | 10 | 53.7 | 33.9 | 87.6 | 14.7% | 4.545x |
| 60 | 10 | 220.4 | 141.5 | 361.9 | 8.6% | **1.101x** |
| 60 | 55 | 242.4 | 152.0 | 394.4 | 11.7% | **1.010x** |
| 60 | 60 | 245.2 | 153.1 | 398.3 | 11.5% | 1.000x |

Three readings, each hostile to the lever, on its best-case card:

1. **A 6x traffic cut (union 60 -> 10) buys 10.1%, not 6x.** The `union=10` row touches
   17.6 MiB of gate+up where `union=60` touches 105.5 MiB.
2. **At the union an independent router produces (~57), the effect is ~1%.**
3. **The whole effect is the size of one arm's own noise.** Per-arm spread is 8.6-11.7%; the
   entire union axis, end to end across a 6x byte change, is 10.1%. At the realistic operating
   point the ~1% delta is an order of magnitude BELOW the spread — a companion interleaved
   pass measured union=55 at **0.9941x**, i.e. nominally slower than union=60. The lever is not
   merely small at realistic unions; it is unresolvable.

And the reason is a controlled 2x2 the sweep already contains: two cells hold the distinct
bytes fixed and change the slots, two hold the slots fixed and change the bytes, both by 6x.

| | 10 distinct experts (17.6 MiB) | 60 distinct experts (105.5 MiB) |
|---|---|---|
| **10 slots** | 87.6 us | n/a (10 slots cannot name 60 experts) |
| **60 slots** | 361.9 us | 398.3 us |

- Hold the BYTES at 10 experts, take slots 10 -> 60 (6x): **87.6 -> 361.9 us = 4.13x.**
- Hold the SLOTS at 60, take distinct bytes 10 -> 60 experts (6x): **361.9 -> 398.3 us =
  1.101x.**

Same 6x on each axis; the slot axis costs 4.13x and the byte axis 1.10x.

`t=9` cross-check (`rig-moeu-t9-interleaved.tsv`), so none of this is a t=6 artifact: 90
slots, a **9x** distinct-byte cut (union 90 -> 10) buys **1.106x**, and at t=9's independent
union of 82 (`512*(1-(1-10/512)^9)`) the curve reads **1.007x** against a per-arm spread of
8.6-13.1%. Identical conclusion at a wider chunk.

The section is per-slot-work bound, and a union gather moves only the byte axis.

### 4b. Box rows — OWED, and the verdict does not wait on them

**Status: the lane box was reclaimed by a fleet-level sweep before the cell got the lock.**
(Provider, instance identity and fleet state are darklanes-side by the public-boundary rule and
are deliberately not recorded here.)

The cell was built, installed and parked correctly, and it never ran. Timeline, UTC
2026-09-01:

| time | event |
|---|---|
| 10:07 | `moe_union_probe.moeu` installed (sha `bbedcc3c...`, src `c0acf3d91`), cell parked on `flock -x` behind the sibling 262k queue's `spec262kv1-thinkon` |
| 10:25 | interleaving fix built on the box; the parked cell STOPPED and relaunched rather than swapping the binary under it (its receipt had already logged the old sha — a receipt naming a binary that did not run is the rebuild-attribution trap) |
| 10:27 | re-parked on `flock -x`, sha `a8cbb7a21bf5b48cbd1cdce2d7f62b04352c5f5442a94b22d7c77a0854bb7fb1`, src `c3f57b289`, marker string verified present in the installed binary |
| ~10:30 | the lane box stops answering (fleet reclaim, reported upstream) |

The sibling queue's cell and both sibling queue scripts (`q4e-qA4v2.sh`, `q4e-qA5.sh`) were
verified alive and untouched after this lane's own processes were stopped; nothing was killed
or reordered to make room, which is also why the cell never won the lock.

**The cell that is OWED, verbatim, for whoever has hardware next.** It needs no checkpoint
and ~1.3 GiB, so it interleaves in ~30 s between any other lane's cells:

```
# build:  cargo build --release -p memra-engine --bin moe_union_probe
# install as a DISTINCT basename (pkill trap): ~/realgate/bin/moe_union_probe.moeu
bash ~/q4e-moeu-cell.sh      # research/qwen4exp-bringup-20260829/spec/moeu/q4e-moeu-cell.sh
# = 3 interleaved t=6 passes + 1 t=9 pass, each:
#   flock -x /tmp/q48fn-measure.lock env MEMRA_MOEU_T=6 MEMRA_MOEU_REPS=25 \
#     ~/realgate/bin/moe_union_probe.moeu
```

| row it would fill | value |
|---|---|
| t=6 slots=60 union=60, section us/launch-pair | _owed_ |
| t=6 slots=60 union=55, and union=10 | _owed_ |
| per-arm spread on a quiet 188-SM card | _owed_ |
| implied section ms/chunk (`us x 48` MoE layers) vs the 7.3 ms mtp10 attribution | _owed_ |

**Why the NO-GO does not wait for it, stated as a falsifiable claim rather than an
assumption.** The box row would tighten the number and replace the 7.3 ms attribution with a
same-tip one. It cannot flip the verdict, for two reasons that are both about direction:

1. **The rig is the lever's BEST CASE on cache grounds.** The t=6 routed gate+up working set
   is 105.5 MiB; the rig's L2 is 64 MB (does not fit, so duplicate slot reads must reach
   DRAM) and the box's is 128 MB (fits, so the hardware can already dedup them). A union
   gather has strictly more to win where the cache cannot do its job. For the box to rescue
   the lever it would have to be MORE byte-sensitive than a card whose L2 cannot hold the
   working set at all.
2. **The residual uncertainty is a noise floor, and it is already smaller than the target.**
   The rig's per-arm spread (8.6-11.7%) exceeds the whole byte axis (10.1%), so a quieter
   card can only resolve the realistic-union delta somewhere between 0% and ~1% of the
   section — i.e. between 0% and 0.23% of the round. Both ends are a rounding error against
   the 176.6 tok/s that DELETING the entire section would produce.

So the box cell is owed as a tightening receipt, not as the go/no-go. If it ever runs and
shows the realistic-union delta ABOVE the section's 10% end-to-end axis, that would
contradict the arithmetic above and this verdict should be reopened on it.

## 5. The arithmetic ceiling — this lever was never the 200 lever

Independent of union sizes and caches, from the current-best receipt (round 31.94 ms =
draft 5.31 + verify 26.63, delivering 4.41 committed tokens):

| | round ms | tok/s |
|---|---|---|
| today | 31.94 | 136.2 |
| **MoE routed union section DELETED ENTIRELY** (7.3 ms, mtp10 attribution) | 24.64 | **176.6** |
| 200 tok/s target | 22.05 | 200 |

**Deleting the whole section does not reach the target.** A union gather cannot delete the
section — it can only remove the duplicated fraction of its weight traffic, and the measured
payoff curve converts even a 6x traffic cut into 10.1% of the section.

Scope note on the 7.3 ms: it is mtp10's attribution, taken on the THINKON shape with a
pre-devtwin binary whose verify chunk was 36.5 ms, while the round arithmetic above is the
RAW shape at 26.63 ms verify. It is used here only to size a share, and the box probe's
own `us/layer x 48` figure is the same-tip replacement for it (section 4b).

Composed honestly, and this is the form that survives the section-3 sensitivity table,
because it reads the payoff curve at the union instead of assuming one. Taking the section's
round share as 22.9% (7.3 ms of 31.94, mtp10 attribution) and reading the measured curve:

| if the real union is | interleaved rig curve (section) | round gain | tok/s |
|---|---|---|---|
| 57 (independent uniform) | ~1.006x | 0.14% | 136.4 |
| 44 (Zipf a=1.0-class) | ~1.040x | 0.88% | 137.4 |
| 39 (40%-sticky-class) | ~1.050x | 1.09% | 137.7 |
| 30 (Zipf a=1.5-class) | ~1.065x | 1.40% | 138.1 |
| 10 (unreachable at top-10 of 512) | 1.101x | 2.16% | **139.2** |

**Every row is a rounding error against a 200 target, including the unreachable one** — and
every row is measured on the card where the L2 CANNOT hide the duplication, i.e. the lever's
best case. The box, whose L2 holds the whole working set, should be flatter still. Every row
above the last also sits below the instrument's 8.6-11.7% per-arm spread.

## Verdict

**NO-GO. The dispatch premise is CORRECT — the kernels really do read each routed expert's
bytes once per slot — and the lever is still not worth building, because the section is not
weight-traffic bound. The prize converts to ~0 at any union size a router can produce.**

Three legs, in order of how much weight they carry:

1. **PRIMARY, and measured: the section is not byte-bound.** On the shipped kernels at a
   fixed 60 slots with the arms interleaved, a **6x** cut in distinct bytes (union 60 -> 10)
   buys **10.1%**, and the 60 -> 55 step buys 1.010x — against a per-arm spread of 8.6-11.7%,
   so the entire byte axis end to end is the size of one arm's noise. The controlled 2x2
   separates the axes: the same 6x factor costs **4.13x on slots** and **1.10x on distinct
   bytes**. This leg does not depend on the union size at all, which is why it is the kill:
   even at an unreachable union of 10 the round gains 2.2% (139.2 tok/s). And it is measured
   on the card whose 64 MB L2 cannot hold the 105.5 MiB working set — the lever's best case.
   `t=9` reproduces it (9x cut -> 1.106x).
2. **Ceiling.** Deleting the entire MoE routed union section yields 176.6 tok/s against the
   200 target, so this was never the lever that closes the owner's gap even at infinite
   success.
3. **Weakest, and explicitly NOT relied on: the union ratio.** Independent routing gives 57
   of 60 (prize 4.7%), but section 3 shows a Zipf-hot or 40%-re-picking router would give
   26-35%, so this leg is only decisive if the real trace comes back near independence. It is
   listed third on purpose. An earlier draft of this document led with it; the sensitivity
   sweep in section 3 refuted that framing before it was banked.

The box's 128 MiB L2 is a fourth leg, and it is used only as a DIRECTION rather than a
number: the chunk's routed gate+up working set (105.5 MiB) fits in it, so the box should
already deduplicate what the kernel re-reads, which is why the rig (64 MB L2, working set
does not fit) is quoted as the lever's upper bound. The measured payoff curve makes the leg
unnecessary either way.

**Kill criteria, stated so a revival has a bar to clear.** The two conditions below are
**AND, not OR**, and that is the load-bearing part: leg 1 kills this lever at *every* union
size, so a low union on its own is NOT grounds to reopen. Someone who measures a clustered
router and reopens on that alone will rebuild a lever that the payoff curve already priced at
~1% of a 23%-of-round section. Reopen only when BOTH hold, and never on re-derivation:

1. **The sel kernels have become byte-bound** — e.g. after the occupancy/reduce work named
   below makes per-slot work cheap enough that weight traffic binds. Test by re-running
   `moe_union_probe`: the condition is that the `union=60 -> union=10` span at fixed slots
   grows well beyond its per-arm spread (it is 10.1% against 8.6-11.7% today). This is the
   necessary condition, and it is the one that fails hardest right now.
2. **AND the real routed union is low** — `union/pairs <= 0.70` on a real t=K+1 trace, which
   for 512 experts needs consecutive columns to re-pick about half their experts. Measure it
   with `moe-union.py`; do not re-argue it. Note the trace that would have answered this was
   owed from the sibling `qA5` queue and died with the box (see 4b), so this number is
   genuinely unmeasured for this artifact — which is exactly why it is condition 2 behind a
   condition that does not need it.

A geometry change (smaller expert count, larger top-k) moves condition 2 only, and so is not
by itself a reopen. K growing past what the L2 can hold moves conditions 1 and 2 together,
which is the one realistic joint path.

## Where the section's time actually is, for the next lane

Not in duplicated expert bytes. The probe says the section scales with SLOTS, and the kernel
sources say why. Named, not built, and deliberately not started inside this lane:

- **The down kernel is lane-starved by construction.** `qmatvec_nvfp4_modelopt_sel_f32_v3`
  loops `for (p = lane; p < pairs; p += 32)` with `pairs = in_f/32`, and the down projection
  has `in_f = ff = 640`, so **`pairs = 20`: lanes 20-31 do nothing for the entire kernel**,
  each active lane executes exactly ONE iteration (zero ILP, pure latency exposure), and the
  block then pays a full 5-step shfl tree over 32 lanes plus 4 accumulators. 37.5% of every
  warp is idle in a launch that moves a third of the MoE bytes. The gate+up launch has
  `pairs = 80` = 2.5 iterations per lane, so it carries a 3-vs-2 tail imbalance instead.
- **One warp per block, by a reverted decision.** Both launchers use `block_dim = (32,1,1)`
  with a comment recording that warp packing was tried and reverted as negative on plain
  decode and flat on verify sel (mtp6). That measurement was taken at t=1/verify-sel of the
  time; the probe's slots-linear scaling is the signal that the geometry deserves a re-price
  at the t=6 chunk shape specifically, where 9,600 one-warp blocks are launched per layer.
- The `_gu_wpr` kernel in `qmatvec.cu` already carries this diagnosis in prose for the dp4a
  family — "nsys: 23.6 MB per call in 29.8us = 792 GB/s, 44% of this pair's GDDR7 peak ...
  **The reduce, not the load, is the cost.**" — and is default OFF and UNPRICED. The same
  structural claim now has a measured twin on the modelopt family.

The TP2-route verify, the second lever VFUSE.md named, is untouched by this lane and remains
out of scope (PROFILE-C0 records TP2 as a depth regression that cannot reach the 262k
window at all).

## Verdicts-ledger rows (BANKED in darklanes PR #49)

```
VERDICT:q4e-moeu-dead | scope: qwen4_exp NVFP4 spec verify K=5 (t=6), 512 experts top-10, memra tip 2026-09-01 | MoE routed-UNION gather DEAD BECAUSE THE SECTION IS NOT BYTE-BOUND, and NOT on the dispatch (which the premise gets right): both sel kernels put slots on grid.y and resolve weights per slot, and `qmatvec_nvfp4_modelopt_sel_gu_silu_f32`'s own doc says "the weight banks are read once per selected slot either way - the launch count is what drops". Priced WITHOUT writing the kernel by holding the shipped kernels at a fixed 60 slots and sweeping only the distinct experts those slots name (that row IS the idealised union gather). Controlled 2x2 with arms INTERLEAVED, same 6x on each axis: slots 10->60 at fixed bytes costs 4.13x, distinct bytes 10->60 experts at fixed 60 slots costs 1.101x - slot-work bound, and a union gather moves only the byte axis. The whole byte axis (10.1%) is the size of one arm's own spread (8.6-11.7%), so at realistic unions the lever is UNRESOLVABLE, not merely small. Round arithmetic: +0.1% at an independent union of 57, +1.1% at a 40%-sticky union of 39, +2.2% (139.2 tok/s) at an unreachable union of 10; deleting the ENTIRE section gives 176.6 tok/s against a 200 target. Measured on the RIG deliberately - its 64 MB L2 cannot hold the 105.5 MiB t=6 working set, so it is the lever's UPPER bound; the box's 128 MB L2 holds it and should dedup the re-reads in hardware already. Do NOT reuse the tempting "512 experts top-10 cannot collide" argument as the kill - that holds only for a UNIFORM router (union 57/60, prize 4.7%); a Zipf a=1.0 or 40%-re-picking router unions to 39-44 (prize 27-35%), and it still dies on the payoff curve. The real union for this artifact is UNMEASURED (the sibling qA5 trace queue died with the box), and that is fine BECAUSE the payoff curve is union-independent. Reopen only on BOTH (AND, not OR): the sel kernels became byte-bound (the union 60->10 span at fixed slots grows well past its per-arm spread; re-run moe_union_probe) AND a real t=K+1 trace shows union/pairs <= 0.70 (moe-union.py) - a low union alone is NOT grounds. Rows measured on the RIG as interleaved within-run RATIOS with per-arm spread, no absolutes; box tightening cell owed and scripted | keywords: moe union, routed union, expert gather, union-of-experts, sel matvec, verify chunk, mtp13, moeu | src: memra research/qwen4exp-bringup-20260829/spec/moeu/MOEUNION.md | since: 2026-09-01 | rev: 2026-12-01
```

```
LAW:price-a-dedup-lever-on-the-fixed-work-arm | scope: any dedup/reuse/caching lever (union gather, weight sharing, cache-hoisting) over a routed or sampled population | a dedup lever changes ONE quantity - distinct items touched - and leaves the work, the launch geometry and the arithmetic alone. So price it WITHOUT building it: run the SHIPPED kernel at fixed work and sweep only the distinct-item count. That row IS the idealised lever, measured. Two traps this closes: (1) the population's collision rate is NOT the prize - qwen4_exp's 512-expert top-10 router unions to 57/60 under independence (prize 4.7%) but 39-44/60 under Zipf/temporal locality (prize 27-35%), so a closed-form uniform-router bound is not a verdict; (2) the prize is not the GAIN - the qwen4_exp sel section converts a 6x distinct-byte cut into 10.1% (against a per-arm spread of 8.6-11.7%, so the whole axis is one arm of noise), because it is slot-work bound, so even the unreachable best case was +2.2% tok/s | keywords: dedup, union, reuse, population, ceiling, top-k, sizing, prize, fixed-work arm, collision rate | src: memra research/qwen4exp-bringup-20260829/spec/moeu/MOEUNION.md | since: 2026-09-01
```

```
TRAP:monotone-sweep-inflates-the-lever | scope: any parameter SWEEP used as a perf A/B (union size, K, chunk width, batch, cache size) | a sweep is an A/B with its arms run as contiguous blocks in monotone parameter order, which is exactly the arrangement LAW:interleaved-ab forbids: clock/thermal drift accumulates onto the arms visited LAST and reads as a parameter effect. Direction matters and is the trap - a sweep ordered "cheap arm first" INFLATES the apparent payoff, making a dead lever look alive. Measured on this lane's own instrument: the moeu union sweep read 1.273x swept vs 1.101x with the same arms interleaved rep by rep, same build, same reps, same card - the extra 17 points were drift. Interleave the arms INSIDE the rep loop and have every arm report its own spread; a delta smaller than the spread is unresolvable, not small | keywords: sweep, interleave, A/B, drift, thermal, monotone, parameter scan, spread | src: memra research/qwen4exp-bringup-20260829/spec/moeu/MOEUNION.md | since: 2026-09-01
```

```
TRAP:nominal-bytes-are-not-dram-bytes | scope: any weight-traffic argument on a card with a large L2 | a per-slot/per-pair NOMINAL byte count is not DRAM traffic. The qwen4_exp t=6 verify chunk's routed gate+up working set is 105.5 MiB and the RTX PRO 6000 Blackwell Server Edition L2 is 128 MiB, so the whole chunk is L2-resident and the hardware already deduplicates re-reads the kernel issues; an achieved-bandwidth figure computed against nominal bytes then overstates how traffic-bound the section is. Measure the payoff by holding the WORK fixed and varying only the distinct bytes on the shipped kernel - that isolates the traffic axis without writing the kernel | keywords: L2, nominal bytes, DRAM traffic, achieved bandwidth, roofline, working set | src: memra research/qwen4exp-bringup-20260829/spec/moeu/MOEUNION.md | since: 2026-09-01
```

```
KNEE:q4e-sel-slots-not-bytes | scope: qwen4_exp NVFP4 MoE sel matvecs (modelopt_sel_gu_silu_f32, modelopt_sel_f32_v3) at verify-chunk shapes | the sel section's cost scales with SLOT COUNT, not weight bytes: 10 slots -> 60 slots is 5.02x at 6x the distinct bytes, while holding 60 slots and cutting distinct bytes 6x moves only 1.115x. Structural cause in the sources: block_dim is (32,1,1) (one warp per block, warp packing reverted on a t=1-era measurement) and the DOWN projection has in_f=640 so pairs=20 - lanes 20-31 idle for the whole kernel, one loop iteration per active lane, then a full 5-step shfl tree over 4 accumulators. Occupancy/reduce structure is the priced-next lever in this section; weight traffic is not | keywords: sel matvec, occupancy, lane starvation, shfl reduce, warp per block, down projection, pairs, verify chunk | src: memra research/qwen4exp-bringup-20260829/spec/moeu/MOEUNION.md | since: 2026-09-01
```
