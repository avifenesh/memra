# lane/b200-dsa-select-20260903: MEMRA_B200_DSA_SELECT, the exact multi-CTA k-pool selector

Follows `lane/b200-dsa-decode-20260902` (PRs #104/#110). That lane took `attn_gathered` and
`kpool_score` down ~10x and 4-7.6x on the pair and named this kernel as what the scorer had been
hiding. No B200 in this worktree; the local RTX 5090 was used for EXACTNESS and direction only.

## 1. What was wrong

`memra_mla_kpool_select_kernel` grids `t_q` blocks, so plain decode runs the whole selection on
**ONE CTA** - 0.68% of a 148-SM die - sweeping `n_pools` up to ten times (8 MSB-first radix
passes, an optional unique-resolution scan, the membership count, the emit). Depth-LINEAR in
`n_pools = t_kv / pool`. Measured on the 5090 at t_q=1: 82.7 us/layer at 131_072 tokens, 149.4 at
262_144, **551.4 at 1_048_576** - against a byte floor of `n_pools * 4 B` = 1 MB per sweep, ~0.13 us at 8 TB/s.
The gap is parallelism, nothing else.

## 2. Exact, not banded - and why that is a construction

The emitted plane is a pure function of ONE 64-bit number: `thr`, the `select_k`-th smallest
order key `(desc32(score) << 32) | pool_index`. The shipped kernel picks `thr` and then emits
every pool with `key(p) <= thr`. `desc32` is a strictly decreasing injection from finite f32 to
u32 and the low word is a unique pool index, so **keys are distinct** and "the k-th smallest" is
unambiguous. Reproducing `thr` bit-for-bit therefore reproduces the selection bit-for-bit. The
parallel pipeline computes the same key and runs the same membership test: this is a
launch-geometry change with an exact answer, not a tolerance. No banded class was needed and none
is claimed.

## 3. The pipeline: six launches, no grid barrier

1. `clear` - zero the histogram and the control words (the workspace is an uninitialised
   allocation, so this is what makes pass 0 well-defined).
2. `hist` pass 0 - 65536-bin histogram of the key's high word, top half, plus the finite count;
   the last-arriving CTA runs the descent and leaves the bins zeroed for pass 1.
3. `hist` pass 1 - the low half within the resolved prefix. `thr_s` is now exact.
4. `tie` - per-CTA count of the tie group, then the last CTA locates the rank-r smallest index
   among the tying pools. `thr_p` is now exact.
5. `count` - per-CTA membership counts; the last CTA exclusive-scans them.
6. `emit` - ascending, two levels of contiguous range, pool expansion, tail, -1 pad.

**Only 32 bits take a radix descent, not 64.** The key's low word is the pool index, which is
unique, so it never needs a histogram: once the score is resolved the threshold's index is simply
the rank-r smallest index among the pools that TIE at that score, and a rank selection over an
ascending contiguous range is a count plus a scan. Ties are the common case, not a corner - ReLU
zeroes every head whose query-pool dot is non-positive, so exact 0.0 ties are ORDINARY (the same
fact that makes the scorer's bit-identity load-bearing).

**No grid barrier and no spin.** Each multi-CTA pass ends with the last CTA to arrive running the
epilogue (`atomicAdd` on a done counter plus `__threadfence()`), the standard two-phase-reduction
idiom. Deliberately not a cooperative launch or a persistent-grid residency assumption: a
spin-wait that deadlocks if occupancy ever changes is not something to put on a serving path,
even behind a default-OFF door.

The 65536-bin descent is run by the last CTA with **all 256 threads** (each sums a 256-bin slice,
thread 0 scans the slice sums, the owning thread re-walks its slice). A single-threaded walk of
65536 bins would have cost ~131 us on its own - more than the kernel being replaced.

## 4. The gate, and its red arm

`crates/memra-engine/src/bin/dsa_select_gate.rs`, three bars:

- **EXACTNESS** - byte-identical `idx` at every score shape x context x width. The shapes hit the
  paths that decide the answer: heavy exact-0.0 ties, a plane where EVERY finite score ties (the
  threshold is then decided entirely by pool index), a sparse plane with fewer finite pools than
  the budget (the rank clamp), and an all-`-INFINITY` plane (nothing selected, tail only).
- **RED ARM** - the same pipeline with the resolved threshold deliberately lowered by one must
  produce a DIFFERENT plane; if it matches, the gate exits 1 however green everything else is.
- **ANCHOR** - the shipped radix kernel vs the in-tree reference selector, so the chain the door
  is compared against is pinned to the order definition rather than to itself.

**The red arm earned its keep on the first run.** The initial version bumped the threshold by
`+1` and silently MATCHED: raising the threshold admits pool `thr_p + 1` only if that pool TIES
at the threshold score, and otherwise the set is unchanged. Lowering by one always drops the
threshold pool itself, because `key(thr_p) == thr` by construction. A red arm that can no-op is
not a red arm, and for one run every exactness cell below it was vacuous.

## 4b. The race review caught, and why no gate could have

`memra_sel_last_arrival` originally read `__threadfence(); if (tid == 0) atomicAdd(&done, 1);`
with **no block barrier first**. That is wrong, and it was wrong in exactly the pass that matters:

`__threadfence()` orders only the CALLING THREAD's prior global writes. The histogram pass's data
is written by all 256 threads (`atomicAdd(&hist[..])`) and by eight warp leaders (the finite
count), with only warp-scoped `__shfl_down_sync` between those writes and the arrival call. So a
CTA's thread 0 could publish that CTA's arrival while its own warp 7 had not yet issued its
histogram atomics. Whichever CTA then observed the full count -- on another SM, synchronised only
through the counter -- would run the descent on a histogram missing those bins and an
undercounted `n_fin`, producing a wrong `ctrl[HI]` and a wrong rank clamp. A **silent,
non-deterministic break of the exact byte-identity this whole door exists to provide.**

The sibling `tie` and `count` passes were correct by accident of shape: they reduce through
shared memory, `__syncthreads()`, and let thread 0 alone write `cta[blockIdx.x]`, so thread 0's
own fence covered the only write that mattered. The histogram pass had no such funnel.

Fixed by putting the `__syncthreads()` inside the helper, ahead of the fence, so every call site
gets the CUDA Programming Guide's `threadFenceReduction` idiom in full rather than three call
sites each getting it right or wrong on their own.

**Nothing in the tree could have caught this.** `compute-sanitizer racecheck` is shared-memory
only; `synccheck` looks for barrier divergence; and this gate runs on an idle device where the
window is vanishingly small. The original 40/40 EXACT receipt did not speak to it, and the
re-run after the fix does not either -- **the ordering argument is the evidence, and the gate
only shows the fix costs nothing.** The exactness cells now repeat 20x each (800 comparisons) as
a net rather than a proof, and the reasoning is written on the helper so it is not re-derived.

Found in review on PR #115, not by a gate. Worth remembering when the next lane reaches for a
last-CTA epilogue: the idiom is `__syncthreads()` THEN `__threadfence()` THEN the counter, and
the leading barrier is the half that is easy to drop.

## 5. Receipts, ORIGINAL 5-rung run (RTX 5090, N=5 interleaved, 2026-09-03)

**This is the FIRST log, `gate-5090-20260903.txt`, 40 exactness cells.** The 7-rung ladder that
supersedes it for every timing figure is section 7c (`gate-5090-band-20260903.txt`, 56 cells).
Do not mix them: overlapping cells moved on the re-run (t_q=4 at 65536 pools 1.10x -> 1.07x,
t_q=1 at 262144 pools 3.17x -> 3.13x), which is ordinary run-to-run spread on a laptop part.

`research/b200-dsa-select-20260903/gate-5090-20260903.txt`. **PASS**: red arm DIFFERS, anchor
IDENTICAL, **40/40 exactness cells EXACT at 20 repeats each (800 comparisons)**.

Timing, direction only under the rig law (the 5090 throttles and this door is sm_100a-gated, so
it cannot even engage there - the gate reaches the kernels through raw FFI):

TRANSCRIBED FROM `gate-5090-20260903.txt` AS IT STANDS. An earlier version of this table was
built from a PREVIOUS run of the gate whose log this file later overwrote, so it disagreed with
its own citation (t_q=4 at 65536 read 1.20x against the log's 1.10x, and the whole t_q=4 column
was off). Numbers here are now read out of the file:

| n_pools | context (tokens) | shipped t_q=1 | parallel | ratio | t_q=4 ratio |
|---|---|---|---|---|---|
| 4096 | 16_384 | 20.3 us | 101.7 | 0.20x | 0.19x |
| 8192 | 32_768 | 25.7 us | 101.6 | 0.25x | 0.21x |
| 32768 | 131_072 | 82.7 us | 92.1 | 0.90x | 0.49x |
| **65536** | **262_144** | **149.4 us** | **97.3** | **1.53x** | **1.10x** |
| 262144 | 1_048_576 | **551.4 us** | **174.0** | **3.17x** | **1.97x** |

Six launches against one kernel is a real fixed cost, so this is a DEPTH door: on this rig it
loses below 131_072 tokens and pays from 262_144 up. The constant was initially 4096, copied from
the scorer's `MLA_DSA_SCORE_MIN_POOLS`; that would have shipped a measured **4.6x regression at
32_768 tokens**, and the gate's own regression bar is what caught it. **The floor this section
argued for was later overturned on the target** -- see section 7d, where t_q=4 at 65536 pools is
0.94x on the pair against the 1.10x above, and the floor became keyed on `t_q`.

## 6. Predictions, SUPERSEDED -- kept only to show what transferred and what did not

Both rows below have since been measured end to end on the pair. Kept because the comparison is
the useful part, not the numbers.

Per token = per layer x 11 MLA/DSA layers at t_q=1, applying 5090 ratios:

| context (tokens) | predicted saving | what the pair actually measured |
|---|---|---|
| 262_144 | 0.55 ms/token (>= 1.5x) | **about +2% end to end** (section 7e) -- the prediction was wrong IN KIND: 1.5x was a kernel ratio, and at that depth the selector is a small share of the token |
| 1_048_576 | 4.19 ms/token (>= 3.2x) | **4.26 ms/token, +17.5%** (section 7b) -- transferred to within 1.7% |

Note the row labels: these are 262_144 and 1_048_576 TOKENS, i.e. the 65536- and 262144-pool
cells. A rung called "256k" is 64_189 pools and does not engage at all (section 7c).

The lesson worth keeping: a kernel ratio becomes a serving number only after weighting by that
kernel's share of the token. At 1M the selector's share is large, so the ratio carried almost
exactly; just above the floor its share is small, so a 1.5x kernel ratio was worth ~2%.

## 7. Composition audit (2026-09-03, static)

Written out because a sibling lane's default flip was withdrawn (#114) for a compose-time defect
that no tape-identity gate catches, and because this lane's own `memra_sel_last_arrival` bug was
the same class: correct in isolation, wrong in combination, invisible to every gate in the tree.

**Buffers this path touches.** `score` (read-only; allocated and filled immediately before the
call, hybrid_forward.rs:8658), `idx` (written here, allocated at :8673, consumed later by
`mla_attn_gathered`), `ws` (allocated per call inside the wrapper, touched only by the six
kernels, never escaping).

| conjunct | verdict |
|---|---|
| `MEMRA_B200_DSA_DECODE=1` / `=2` | The ONLY door that meets these buffers: `=1` may replace the producer of `score`, `=2` the consumer of `idx`. Same stream, strict producer -> select -> consumer order, no shared scratch. Composes. |
| `MEMRA_B200_MATVEC_ARM`, `MEMRA_GLM5_W8`, `MEMRA_B200_GEMV_V2` | Change the `mm()` producing `q_index`/`head_weights`, i.e. the VALUES in `score`. Input data, not a shared buffer; the exactness claim is conditional on the score plane handed in, not on how it was computed. Composes. |
| `MEMRA_HC_FUSED_PRE`, `MEMRA_KDA_FUSED_PROJ`, `MEMRA_GLM5_Q8_FUSE` | Do not appear anywhere in the kpool call chain (hybrid_forward.rs:8439-8690). Disjoint. |
| `MEMRA_B200_PRIME_V2` arms 1+2 | The mHC PRIME schedule (hybrid_forward.rs:678-1405), a prefill mechanism; this door is `t_q <= 8` only. Disjoint by width AND by code path. |
| `MEMRA_GLM5_DECODE_GRAPH` | `ws` goes through `uninit_i32` -> `alloc_uninit` -> stream-ordered `alloc` + `keep_if_capturing`, the identical contract `score`, `idx` and the sibling lane's `part_*` buffers already use here. Composes for the same reason they do. |
| streams | The engine has two: `stream()` and `copy_stream` (MoeSlotCache weight prefetch only). All six kernels launch on `stream()`; none of these buffers is touched by `copy_stream`. |
| slice arithmetic (#114's failure mode) | Kernels index only through `memra_mla_kpool_select_ctas`/`_ws_ints`, the same entry points the host sizes from; verified numerically over n_pools 0/1/255/4096..1M including the 1024-CTA clamp, emit worst case (pool slot 2047, tail slot 2050) under `width` 2051. **No Rust-side slicing of a device buffer anywhere in the path, so there is no `slice_mut` to panic.** |

**What the audit still does NOT cover, stated rather than assumed:** it is a STATIC audit, and
no conjunct has been run in isolation against this door. The door itself HAS now been exercised
end to end on sm_100a -- the three-pair serving receipt in section 7b is that exercise, and its
`mla_dsa_select_announce` line only fires when `mla_dsa_select_on() && mla_dsa_select_engages(..)`
-- but it ran in ONE posture, the full best-posture stack. So what is measured is "the door
composes with that whole stack", not "the door composes with each of these doors independently".

## 7b. sm_100a serving receipt (2026-09-03, three pairs, MEASURED)

Main `3908a431` built fresh on the B200 pair, best posture, vendor sampling, 1_027_024-token
prompt (256_756 pools), measured rep only, fresh boot per arm. Receipts:
`darklanes:research/glm5-b200-20260902/box/dsasel2/`, driver `box/dsasel2.sh`.

| pair | 1M decode OFF | 1M decode ON | delta |
|---|---|---|---|
| 1 | 35.17 | 41.00 | +16.6% |
| 2 | 34.95 | 41.07 | +17.5% |
| 3 | 34.72 | 41.09 | +18.3% |
| **median** | **34.95** | **41.07** | **+17.5%** |

**The OFF and ON spreads do not overlap** (off 34.72-35.17, on 41.00-41.09), which is this lane's
bar for a real win rather than a boot difference.

In per-token terms: **28.61 -> 24.35 ms, 4.26 ms saved**, against the rig-derived prediction of
4.19 ms over 11 MLA/DSA layers. **The ratio transferred to within 1.7%** -- a stronger statement
about deriving a serving prediction from a synthetic kernel gate than about the number itself.

Controls all behaved:

```
[mla-b200-dsa-select] engaged kpool_select t=1 pools=256756 ctas=1003 class=exact
  (sm_100a; MEMRA_B200_DSA_SELECT=1)
```

`select=1 pools=256756 ctas=1003` on all three ON arms and `select=0` on all three OFF arms; TTFT
flat at 488-496 s on both arms, which is what a decode-only door requires; and the 66-token
control unchanged across all six boots because it sits far below the pool floor.

**Where this lands on the lane's ladder:** 1M decode went 22.7 base -> 34.9 with
`MEMRA_B200_DSA_DECODE` -> **41.07 with this door on top, 1.81x over where the lane started**.

### 7b.1 The `RemoteDisconnected` from the first pass was harness fratricide, not this door

Recorded so it is not later misread as a stability question. A follow-up knee cell waited on a
`grep -q "ALL DONE"` marker that an earlier aborted pass had already written to the same log, so
its wait loop fell through immediately and it `pkill`ed the live server mid-request while booting
its own beside it. The engine behaved correctly throughout (`SIGTERM: draining (1 in flight,
deadline 30s)`); the client simply saw the socket close. No panic, no defect in this path.

## 7c. The floor is 262_144 tokens, and a "256k" rung is under it

`MLA_DSA_SELECT_MIN_POOLS` is 65536 pools, which at `pool = 4` is **262_144 tokens exactly**. A
serving rung called "256k" is usually ~256_756 tokens = **64_189 pools, 2.06% under the floor**,
so it engages nothing. That cost a real B200 rung on 2026-09-03, sized from an earlier note of
mine that read "n_pools >= 65536 (256k context at pool=4)" -- true only if "256k" means the
binary 262_144. Every context figure in this lane is now an exact token count.

Full rig band (RTX 5090, N=5 interleaved, `gate-5090-band-20260903.txt`), t_q=1 / t_q=4:

| n_pools | tokens | t_q=1 | t_q=4 |
|---|---|---|---|
| 4096 | 16_384 | 0.20x | 0.19x |
| 8192 | 32_768 | 0.25x | 0.25x |
| 32768 | 131_072 | 0.90x | 0.51x |
| 49152 | 196_608 | **1.13x** | 0.64x |
| 64189 | 256_756 | **1.44x** | 1.02x |
| 65536 | 262_144 | **1.53x** | **1.07x** |
| 262144 | 1_048_576 | **3.13x** | **1.95x** |

**`t_q=4` is what holds the floor up.** Plain decode crosses into profit around 49152 pools and
is a clear 1.44x by 64189; the DFlash2 spec-verify width is still 0.64x at 49152 and only reaches
parity near 64189. The door engages uniformly across `t_q`, so the floor is set by the worse
width, and 65536 is the first cell that wins at both.

So the door refuses a 256k rung the rig measures at **1.44x on the plain route**. Closing that
needs the floor KEYED ON `t_q` -- a lower floor for plain decode, this one kept for spec-verify --
the way the sibling `MLA_DSA_ATTN_ARM` table is keyed. Deliberately NOT done here: a default-ON PR
for this door is in flight, and widening which shapes engage underneath it is the wrong order of
operations. It wants its own PR and a B200 gate run.

## 7d. Kernel gate ON THE TARGET, and the floor decision it forced (2026-09-03)

`dsa-select-gate`, 2x B200 SXM (sm_100a), dev 0, N=5 interleaved, binary built from main
`3908a431`. Receipts: `darklanes:research/glm5-b200-20260902/box/selgate/gate-b200.{txt,full}`,
driver `box/selgate.sh`.

**Exactness is perfect on sm_100a: 40/40 cells EXACT** across mixed, all-ties, sparse and empty by
t_q 1 and 4, anchor against the in-tree reference selector IDENTICAL, and **the RED ARM DIFFERS**.
The gate is non-vacuous on the target, not only on the rig -- which is what the whole
exact-not-banded argument rested on.

| pools | tokens | t_q=1 | t_q=4 |
|---|---|---|---|
| 4_096 | 16_384 | 0.20x | 0.20x |
| 8_192 | 32_768 | 0.25x | 0.28x |
| 32_768 | 131_072 | 0.67x | 0.35x |
| **65_536** | **262_144** | **1.31x** | **0.94x** |
| 262_144 | 1_048_576 | **2.81x** | **2.06x** |

**That run FAILED its own regression bar**, and the failure is the finding:

```
REGRESSION kpool_select n_pools=65536 t_q=4: 311.6 us vs shipped 292.7 us (1.064x, margin 1.05x)
```

The uniform floor of 65536 was chosen from RTX 5090 data, where `t_q=4` at that point measured
1.07x -- a small win. On the silicon this door is gated to it is **0.94x, a 6.5% loss**. So the
floor **admitted a shape that regresses on the target**, and the door is sm_100a-only: its floor
had been set by evidence from a card it never runs on. That is a policy-CORRECTNESS defect, not a
missed optimisation.

### The decision

**The floor is now keyed on `t_q`** (`mla_dsa_select_floor`):

| width | floor | why |
|---|---|---|
| `t_q == 1` (plain decode) | 65536 pools = 262_144 tokens | B200-measured **1.31x** |
| `t_q >= 2` (spec verify) | 262144 pools = 1_048_576 tokens | B200-measured **2.06x**, the only pool count where this width was measured to win |

Both are measured cells with **no interpolation**; everything between them at `t_q >= 2` is
unswept, and unswept shapes do not engage.

This lands together with the evidence rather than waiting for a separate PR, which is the opposite
of the sequencing argued in section 8 for the default flip -- and deliberately so. That argument
was about WIDENING engagement while flipping a default. This **narrows** engagement: it removes a
measured regression and adds nothing. Narrowing to delete a known-bad shape is strictly
derisking, and holding it back would mean knowingly leaving a floor that fails its own gate.

Nothing here touches the 1M serving result: that cell ran `t_q=1`, and at 262144 pools the kernel
is 2.81x at t_q=1 and 2.06x at t_q=4, so the door is good for both routes there. The +17.5%
end-to-end stands.

### The 5090 read this corrects

Section 7c called 65536 "the first cell that wins at BOTH measured widths". True on the rig, false
on the target: the rig put t_q=4 at 1.07x there, the pair puts it at 0.94x. The rig band table in
7c is kept as-is because it is an accurate record of that rig -- but **quote section 7d for any
sm_100a decision**, and treat the 5090 crossover as a direction that did not transfer.

## 7e. The last prediction, now measured

The floor cell (`box/selfloor.sh`, two pairs, door OFF vs ON) measured the rung just above the
plain-decode floor end to end:

| rung | pools | result |
|---|---|---|
| 264_290 tokens (just OVER the floor) | 66_072 | **about +2%** -- deltas +2.7% and +1.2%, arms do not overlap across the two pairs |
| 249_251 tokens (just UNDER) | 62_312 | arms overlap, the two pairs disagree in sign -- what no effect looks like, the negative control behaving correctly with the door armed |

That **replaces** the old "~0.55 ms/token, >= 1.5x at 262_144" prediction, which was wrong in KIND
rather than degree: 1.5x was a KERNEL ratio, and at that depth the selector is a small share of
the token, so it converts to a couple of percent end to end. The lesson is worth keeping: a kernel
ratio only becomes a serving number after it is weighted by that kernel's share of the token, and
at 1M the selector's share is large (hence +17.5%) while just above the floor it is not.

## 8. Open

The 1M serving A/B is done (7b), the kernel gate has run on the target (7d), the floor is keyed
from target evidence (7d), and the last prediction is measured (7e). What remains:

1. **A default-ON PR.** Now unblocked. The case: the class is exact by construction and the gate
   proved it EXACT on sm_100a itself (40/40, anchor identical, red arm fires); +17.5% median at 1M
   with non-overlapping spreads; and the floor no longer admits the shape that regressed. Still a
   SEPARATE PR from this one -- a default flip deserves its own review, and this PR is already
   carrying a policy change plus its evidence.
2. **The `t_q >= 2` band between 65536 and 262144 pools is unswept.** The spec-verify floor sits
   at 262144 because that is the only measured win at those widths, not because 262144 is where
   the crossover is. If the spec route matters between 262_144 and 1_048_576 tokens, sweeping that band would
   likely lower it. Cheap: it is a `dsa-select-gate` ladder edit and one box run.
3. **The composition audit (7) is still static.** No conjunct has been run in isolation; the
   end-to-end evidence is one posture, the full best-posture stack.

Historical note, kept because it explains why this doc was written prediction-first: for part of
2026-09-03 the vast.ai account was out of credit and every instance was stopped - the B200 pair,
the glm5 prod box, q38 and ornith - so no sm_100a cell could be scheduled and this section read
"OWED". That is resolved.
