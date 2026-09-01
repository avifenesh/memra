# glm5 EXPERT-SLAB DEDUP — spending the 21.96% the struct battery measured
### lane/glm5-dedup, 2026-08-31

Base: `origin/lane/glm53-flash-bringup` @ **b55536ab1** (fetched first — the moe-loc merge plus its
perf-ci row; doors D/H and the dedup instrument in tree). Worktree `~/projects/wt-glm5-dedup`,
branch `lane/glm5-dedup`. A predecessor lane died to provider errors before any commit and left no
worktree; opened fresh.

Charter: the vrows MoE pair (`moe_gate_up_preclamp8_q8_rows` + `moe_down8_fma_q8_rows`) is at
**90.2% / 89.9% of this card class's theoretical DRAM peak** (moe-loc LANE.md §1.3) and therefore
ABOVE the banked 87% achievable bound — the efficiency class is closed and the only surviving lever
is **reading less**. The struct-battery box window then measured how much less is available:

> **21.96% cumulative repeat fraction** over 99,751 vrows layer-calls / 2.55M expert visits on the
> real artifact and the ship recipe — greedy pools 22.27% vs vendor-default sampled 21.53%, so it
> is a ROUTING property and not a decoding artifact. That is **6.9x the 3.21% independent-routing
> bound**. (`../struct-battery-20260831/WINDOW.md` cell 2, receipts `ce9b57bb5`.)

About a fifth of the (verify row, expert) visits in ONE layer-call re-read a slab a sibling verify
row already read. moe-loc sized the lever at ~-2.1 ms/round / +5.6-6% ship and named the design
order in its follow-up #4: **expert-major ordering FIRST, then the shared-slab twin.** This lane
lands the ordering, and refuses the twin on receipted arithmetic (§5).

Rig law: every number produced here is an exactness or counter receipt (`flock
/tmp/memra-5090.lock`, `NVIDIA_TF32_OVERRIDE=0`); every ms is arithmetic against banked BOX
constants and labelled as such. **A rig-hold order landed mid-lane (§7) and the standing batteries
are DEFERRED — read §6 before treating anything here as a final-tree receipt.**

---

## 1. THE MECHANISM — and why the charter's own framing was too weak

### 1.1 The reuse-window problem, stated with the numbers

moe-loc §1.6 named the constraint: the per-layer-call working set is 26.738 visits x 14.156 MB =
**378 MB against ~128 MB of L2**, so a repeat visit is cache-served only if it is SCHEDULED within
about a third of the call. The charter's framing was slab residency: a 4.72 MB slab against ~128 MB
of L2 means several can be resident, so put same-expert visits near each other.

That framing is true and it is also the WEAK version. Look at what the shipped grid actually does:

    moe_gate_up_preclamp8_q8_rows   grid = (n_ff, n_pairs)   blockIdx.x = o (FASTEST)
    moe_down8_fma_q8_rows           grid = (out_f, t)        blockIdx.x = o (FASTEST)

CUDA dispatches blocks in increasing linear id with x fastest, so the shipped schedule walks ONE
pair's whole slab pair (2048 blocks, 9.4372 MB) before it touches the next pair. Two pairs sharing
an expert are therefore separated by **a full slab pass at best, and by up to 26 slab passes
(252 MB) at worst** — which is why moe-loc could only offer "worth exactly what the overlap
measurement says". Making the pair order expert-major shortens that to one slab pass: 9.44 MB has
to survive in L2. That works, but it is a residency bet.

**Transposing the grid removes the bet.** Put the deduplicating index in the FASTEST dimension:

    moe_gate_up_preclamp8_q8_rows_ord   grid = (n_pairs, n_ff)   blockIdx.x = pr (FASTEST)
    moe_down8_fma_q8_rows_tmaj          grid = (t, out_f)        blockIdx.x = tok (FASTEST)

Now the entire pair union for ONE expert-FFN row `o` is in flight together, and two pairs sharing
an expert read the **identical gate row and up row** — 2 x 2304 B at the serving geometry
(`in_f` 4096 x 0.5625 B/element) — from ADJACENT blocks. Reuse distance falls from 2048 blocks over
9.44 MB to ~1 block over ~4.6 KB: an L1-class hit rather than an L2 gamble. Composed with the
expert-major order plane, same-expert pairs are literally consecutive blocks.

### 1.2 The L2 footprint does not change, which is the point

The obvious objection to transposing is that ~27 concurrent slab streams replace one. The per-wave
L2 working set says otherwise. On the serving card class (188 SMs; a one-warp block is capped by
the 32-blocks/SM limit, so ~6,016 blocks are resident):

| schedule | resident blocks cover | bytes in flight |
|---|---|---|
| shipped (o fastest) | 6,016 consecutive `o` of ONE pair | 6,016 x 4,608 B = **27.7 MB**, 2 streams |
| `_ord` (pr fastest) | 225 consecutive `o` of EACH of ~26.7 pairs | 26.7 x 225 x 4,608 B = **27.7 MB**, ~53 streams |

**Identical footprint** — 27.7 MB against ~128 MB of L2 (and against a 96 MB L2 too, so the
argument does not depend on which figure for this card family is right). Only the DISTRIBUTION
across slabs changes, and each of the ~53 streams still advances sequentially through its own
slab's rows, which is what DRAM row-buffer locality actually wants. The risk is real but bounded
and it is named in §4.3, not waved away.

`_tmaj` is the milder case: a down block ALREADY reads 8 slabs' row `o` (its `j` loop), so the door
takes the stream count from 8 to ~26.7, not from 1.

### 1.3 What the door may NOT touch, and how it is guaranteed

The down chain's slot-ordered `__fmaf_rn` accumulation is the standing vrest gate-4 bit bar. It
lives INSIDE a block (`for j in 0..n_used`, `pr = tok*n_used + j`) and it keeps its **original slot
order** in `_tmaj` — verbatim. The `_tmaj` door therefore takes NO order plane at all: permuting
the down visit list would permute the accumulation, and that is refused, not gated.

The gate/up side has no cross-pair accumulation, so it takes the permutation. **No reorder buffer
is needed anywhere in this lane** (the charter's conditional: the kernel does not accumulate
visit-ordered, so the explicit reorder buffer it asked for is not required — the shared-slab twin
of §5 is where one WOULD be, and that is exactly one of the reasons the twin loses).

Bit identity is structural rather than empirical: in both kernels every output is a pure function
of its coordinate — `ptrs[pr]`, `ptrs[n_pairs+pr]`, `scl[pr]`, `scl[n_pairs+pr]`, `tok = pr/n_used`,
`act[pr*n_ff+o]` / `dst[tok*out_f+o]` — and neither kernel has a `__syncthreads`, shared memory, or
any cross-block communication. Re-indexing WHICH BLOCK COMPUTES WHICH OUTPUT moves no bits, and the
order plane is a permutation, so every pair is still computed exactly once.

The bodies are their twins' character for character: same `expert_dot_g` g-strided chain (==
`qmatvec_expert_q8`'s), same `warp_reduce_sum`, same `swiglu_preclamped_mul_scaled_f32` expression
on the exact dot values, same slot-ordered `__fmaf_rn` chain.

### 1.4 The order plane rides the table that already exists

The pointer table grows a FOURTH plane: `ptrs[3*n_pairs .. 4*n_pairs)` (planes gate | up | down |
order). That choice is load-bearing on cost:

- **host-tables arm (door D off):** the plane is appended to the `Vec` the host already uploads, so
  it rides the SAME `htod_u64_into`. **Zero extra transfers, zero extra allocations, no new
  workspace pool** (it draws from door W's existing u64 pool at a new size key).
- **device-tables arm (door D on):** `moe_vrows_order_from_sel` writes the plane where the
  selection already lives. One extra launch per MoE layer-call = 42/round = **~0.093 ms** at the
  box's 2.216 us eager-launch constant. Named follow-up 1 folds it into
  `moe_vrows_tables_from_sel` (same inputs, same one-thread-per-pair grid) and recovers it.

The device build is a **stable counting rank** — thread `p` counts how many pairs sort strictly
before it on `(expert id, pair index)` and stores itself at that rank. Chosen because it needs no
scratch, no scan and no order-of-execution dependence, so it is bit-identical to the host arm's
stable sort with nothing to prove about scheduling. O(n_pairs^2) with `n_pairs = t*n_used <= 64` on
every serving shape = 4,096 comparisons in one 128-thread block.

---

## 2. WHAT LANDED

| door | flag (default) | mechanism | bit bar |
|---|---|---|---|
| **E** expert-major gate/up | `MEMRA_MOE_VROWS_DEDUP_ORDER` (**OFF**) | new kernel `moe_gate_up_preclamp8_q8_rows_ord`: grid `(n_pairs, n_ff)`, pair from the order plane. Plane built by `moe_vrows_order_from_sel` (device arm) or `vrows_expert_major_order` (host arm). | every output a pure function of `(o, pr)`, no block communication, plane is a permutation |
| **E-down** token-major down | `MEMRA_MOE_VROWS_DOWN_TMAJ` (**OFF**) | new kernel `moe_down8_fma_q8_rows_tmaj`: grid `(t, out_f)`. Slot-ordered `__fmaf_rn` chain verbatim in its ORIGINAL order. | same, plus the accumulation order is untouched by construction |

Split into two flags deliberately: gate/up is 2/3 of the pair's 15.896 GB/round and down 1/3, the
two halves have different risk profiles (§4.3), and the box should be able to attribute the win —
or the loss — to one of them. Both carry a FLAGS.md row in this PR with both arms, the refusals,
the rollback seam and this doc as the receipts pointer; both kernels carry a KERNELS.md row.

**Both defaults are OFF BY DESIGN.** The identity is structural, but the WIN is a block-dispatch-
order property and the rig is exactness-only by law — there is no timing row this lane could
legitimately produce. They ship with COUNT receipts and a predicted band; the box prices the flip.

### Refusals, by name

| refusal | behaviour |
|---|---|
| door M (`MEMRA_MOE_VROWS_PACK`, refuted at 0.9959x) armed | BOTH doors fall closed to the shipped grid; the two schedule families are never crossed, so no `_ord_w4`/`_tmaj_w4` cross product exists. Gated. |
| order plane absent (`ptrs.len() < 4*n_pairs`) | door E falls closed. This is what keeps every standing gate's 3-plane direct launcher call on the shipped program instead of reading past the table. Gated. |
| `n_ff > 65535` (door E) / `out_f > 65535` (door E-down) | falls closed on the grid.y bound. Not a serving shape (2048 / 4096), stated so it is not a silent cliff. |
| **BOTH table arms supported** | the charter's requirement. Door E works with host-built tables AND with door D's device-built tables, and both provenances are gated bitwise (§3). Nothing is refused by table arm. |

The avoided-slab-read counter (`MOE_VROWS_SLAB_READS_AVOIDED`) is the one thing that is HOST-ARM
ONLY, and it is a measurement, not the door: with door D on there is no host-side selection to
count and a 4-byte readback would reintroduce the very `cuStreamSynchronize` door D removed. The
counting boot is `MEMRA_MOE_VROWS_DEV_TABLES=0`, exactly like the dedup instrument. The engagement
counters (`MOE_VROWS_DEDUP_ORDER_DISPATCHES`, `MOE_VROWS_DOWN_TMAJ_DISPATCHES`) and both announce
lines move in BOTH arms.

---

## 3. GATE TABLE

Rig 5090, `flock /tmp/memra-5090.lock`, `NVIDIA_TF32_OVERRIDE=0`, exactness and counters only.
**Read §6 for what ran and what is deferred.** New suite `glm5_dedup_sched_gpu`, **6/6 PASS**:

| gate | result |
|---|---|
| 1. order plane: device build vs the HOST stable sort, t=2..=8 — exact `u64` equality, plus three properties asserted rather than assumed: it IS a permutation of `0..n_pairs`, it IS stably expert-major, and its RUN COUNT equals `distinct` (which is what makes `visits - distinct` the avoidable-read count) | PASS 7/7 shapes. Selections plant repeated experts across rows, two identically-routed rows, and expert 0 |
| 1R. changed-selection RED | BITES at every t — the plane moves when `sel` moves, so `sel` is read |
| 2. `_ord` gate/up bitwise vs the shipped schedule: t=2..=8 x {live macros, none} x {host tables, device tables} | PASS, **28/28 arms**, 0 bit diffs |
| 3. `_tmaj` down bitwise vs the shipped schedule: t=2..=8 x {live macros, none} x {down-only, composed host tables, composed device tables} | PASS, **42/42 arms**, 0 bit diffs. Routing weights deliberately span five decades so a reordered `__fmaf_rn` chain could not pass as identical (a flat weight vector would have let one through) |
| 4A. VALID SHUFFLES ARE BIT-INERT — 3 seeded permutations of the visit order | PASS, 0 bit diffs each. This is the lane's arithmetic claim, stated as a positive invariance rather than a red |
| 4B. NON-PERMUTATION REDS BITE — one duplicated entry (one pair dropped) and a degenerate all-zero plane | BOTH BITE (128 and 512 outputs differ). This is the ONLY arm that proves the plane is READ: a kernel ignoring it would compute `pr = blockIdx.x`, itself a valid permutation, and would pass gate 2, gate 3 and 4A |
| 4C. dropped-macro RED, re-bitten THROUGH the dedup schedule | BITES, 512 outputs differ — the identity is not vacuous about the scale planes either |
| 5. counters + refusals: both counters move ON; FLAT with both doors pinned `=0`; door E FLAT on a 3-plane table while door E-down still engages; BOTH FLAT under door M | PASS |
| 6. avoided-slab-read arithmetic on planted overlaps (CPU): disjoint 8/8 avoided 0, two rows sharing 3 experts 8/5 avoided 3, identically-routed 6/2 avoided 4, one-expert-everywhere 12/1 avoided 11 — and the order plane's run count == `distinct` in every plant | PASS |

### 3.1 A FALSE NEGATIVE IN A RED ARM, found and fixed in-lane (twice)

Gate 4B's first pass **did not bite**, and the gate said so loudly. Cause: `act` is an uninit draw
(`vws_uninit`, and with door W off that is `alloc_uninit`), the red's reference had been computed
EARLIER FROM THE SAME `z`, and the async allocator handed the red back the very block the reference
run had just freed — so the pair the non-permutation DROPS read as exactly correct and the red
passed as inert.

Fixed by making the red run on a `z` no earlier arm uses AND run BEFORE its own reference, so no
freed block can hold the correct row for the dropped pair (and a fresh zeroed page cannot match a
live-clamp epilogue's nonzero row either). The second pass then reproduced the SAME trap one level
deeper: the two red arms shared one `z`, so the first arm's reference run freed a correct block of
exactly the right size for the second arm. Each red arm now carries its own `z`.

This is the loud-failures-fail-quietly / pin-against-truth-not-siblings shape, and it is worth
recording as a general trap for this family: **a red arm that compares against a reference computed
from the same inputs on an uninit-draw buffer can be silently vacuous.** Two arms of it here; both
were caught only because the red's outcome was asserted rather than assumed.

---

## 4. PREDICTED SHIP ARITHMETIC (nothing here is a claim; the box prices it)

Constants, all banked and cited: pair bytes/layer-call **gate_up 252.33 MB @ 156.3 us =
1.6144 TB/s achieved** and **down 126.16 MB @ 78.4 us = 1.6092 TB/s** (moe-loc §1.3, from the
mv-battery c5 winner census); **42 MoE layer-calls/round**; repeat fraction **21.96%**
(struct-battery cell 2); baseline **71.489 tok/s = the current best single stream** (struct-battery
cell 1 dhon, D+H ON) at **2.6301 tok/round = 36.791 ms/round**; eager launch **2.216 us**.

### 4.1 Bytes avoided -> ms/round

    gate/up   0.2196 x 252.33 MB = 55.41 MB/call  x42 = 2,327.3 MB/round  / 1.6144 TB/s = 1.442 ms
    down      0.2196 x 126.16 MB = 27.70 MB/call  x42 = 1,163.6 MB/round  / 1.6092 TB/s = 0.723 ms
    composed                                        3,490.9 MB/round      =              2.165 ms
    device-tables arm only: +42 order launches x 2.216 us                 =            + 0.093 ms

Pair bytes/round fall 15,896 -> 12,405 MB, so the pair stays bandwidth-bound after the removal
(12,405 MB at 1.61 TB/s = 7.69 ms vs 9.856 measured) — the door does not walk the pair off its own
floor.

### 4.2 ms/round -> ship tok/s

| arm | Δms/round | ms/round | ship tok/s | ratio | still needed for 100 |
|---|---|---|---|---|---|
| composed, host tables (door D off) | **-2.165** | 34.626 | **75.96** | **1.0625x** | 1.317x |
| composed, door D ON (device tables) | -2.072 | 34.719 | 75.75 | 1.0596x | 1.320x |
| door E only (gate/up half) | -1.442 | 35.349 | 74.40 | 1.0407x | 1.344x |
| door E-down only | -0.723 | 36.068 | 72.92 | 1.0200x | 1.371x |
| 50% of the schedule transfers | -1.083 | 35.708 | 73.65 | 1.0302x | 1.358x |
| 0% transfers | 0 | 36.791 | 71.49 | 1.0000x | 1.399x |

This reproduces moe-loc's sizing of the lever (~-2.1 ms/round, +5.6-6% ship) from its own constants
rather than restating it, and it lands inside the §1.6 sensitivity table's 20%-row (-1.97 ms,
1.056x) as it should.

### 4.3 THE BYTE ARITHMETIC IS A CEILING — two named risks that could eat it

Stated before the box runs, so nobody reads the table above as a prediction of the wall:

1. **An issue/ALU ceiling underneath the byte floor.** All 26.738 dots are still COMPUTED; only
   their DRAM traffic is deduplicated. If the kernel hits an issue-rate ceiling once the byte
   stream is 22% lighter, the win is less than the arithmetic. Evidence that headroom exists: door
   M raised warp occupancy from <=67% to 100% and bought nothing (0.9959x), which says the kernel
   is not issue-bound AT THE SHIPPED byte rate — it says nothing about a 22%-lighter one. This is
   the honest floor of the band and it is why the 0% row is in the table.
2. **DRAM stream count.** Gate/up goes from ~2 sequential streams to ~53 (§1.2). The per-wave L2
   footprint is unchanged at 27.7 MB and every stream still advances sequentially, but if DRAM
   efficiency drops by more than the 21.96% of bytes removed, **door E loses**. That is precisely
   why the two halves are separate flags: `_tmaj` carries far less of this risk (8 -> ~26.7 streams,
   from a block that already interleaved), so a split verdict is expressible.

The interleaved-A/B protocol law applies to whatever the box measures here: x3 interleaved minimum,
x5 on anomaly, cross-arm tape identity as a STOP bar (both doors carry rig bit gates, so any
divergence on the served path is a bug, not a tolerance).

---

## 5. THE SHARED-SLAB TWIN — DESIGNED, SIZED, AND REFUSED ON ARITHMETIC

The charter's work item 2, conditional on item 1 "under-delivering on the rig's L2". Two things to
say plainly. First, **that condition cannot be evaluated on this rig**: the rig is exactness-only by
law and cannot produce an L2 or a timing row, so "under-delivers" is a BOX measurement — and door
E's box row IS that measurement. Second, the twin was designed out in full anyway, and the
arithmetic refuses it.

**The design.** One pass per DISTINCT expert covering its rows (the read-once shape, doors T/X's
weight-once pattern applied to the expert planes). Groups as a canonical-representative linked list
(`grp_head[p]`, `grp_next[p]`, buildable device-side by the same O(n^2) one-block kernel), so the
grid stays `(n_ff, n_pairs)` and no host-visible `nd` is needed — a device-dependent grid extent
would have forced a readback and undone door D. A representative block reads its expert's row once
and runs `cnt <= t` member accumulators; each member's accumulator sees the identical sequence of
`expert_dot_g` terms in the identical g order, so it is bit-identical for the same reason door E is.

**Why it loses, term by term:**

| | door E (landed) | shared-slab twin |
|---|---|---|
| gate/up bytes avoided | 55.41 MB/call | 55.41 MB/call — **identical ceiling** |
| gate/up cost | none (grid + one plane) | linked-list group tables, `2*t` live accumulators per warp (register pressure on a kernel whose whole margin is residency) |
| down bytes avoided | 27.70 MB/call | 27.70 - 0.875 = **26.83 MB/call — strictly WORSE** |
| down extra traffic | none | the explicit reorder buffer the accumulation forces: `part[n_pairs, out_f]` f32 = 437.6 KB written + read per layer-call (+0.69% on down's bytes), plus a combine kernel replaying the slot-ordered `__fmaf_rn` chain |
| down block count | 4,096 x 3.34 = 13.7k | 4,096 x 26.7 = **109.4k launched** (85.4k doing work) + 13.7k combine |
| what it buys | — | removes the block-dispatch-order assumption; makes the repeat register-local inside one warp |

So the twin is a **wash on gate/up's ceiling and strictly worse on down's**, for materially more
machinery — and its one genuine advantage is insurance against a risk that door E's own box row
measures directly. Building both now would be paying for the insurance before reading the premium.

**The trigger, written so it is not a judgement call later:** if the box shows door E transferring
materially below its §4.2 row while `MOE_VROWS_DEDUP_ORDER_DISPATCHES` confirms engagement — i.e.
the schedule is running and the reuse is NOT landing — then build the twin's **gate/up half only**
(register-local reuse, no reorder buffer, ceiling unchanged) and leave `_tmaj` as the down answer.
That is named follow-up 3.

## 5.1 Also refused here, on geometry (so it is not re-derived)

- **A permutation of the down visit list.** It would permute the accumulation. The down chain's slot
  order is the vrest gate-4 bit bar; `_tmaj` moves the grid and nothing else.
- **Deriving `nd` (distinct count) host-side to size a grid.** One 4-byte readback per MoE
  layer-call = 42 `cuStreamSynchronize` per round, exactly what door D removed for a 1.0154x win.
  Any dedup shape that needs a device-computed extent is refused at this seam.
- **Counting avoided reads in the device-tables arm.** Same reason; the counter is host-arm-only by
  design and the instrument boot pins `MEMRA_MOE_VROWS_DEV_TABLES=0` (§2).

---

## 6. WHAT RAN, WHAT IS DEFERRED, AND WHY — read before citing anything above

An **owner rig-hold order landed mid-lane** (2026-09-01, relayed by the coordinator): the local rig
is occupied, all local GPU gates and heavy builds stop, and the lane completes CPU-light. A
follow-up authorized `MEMRA_SKIP_PERF_CI=1` for the push, with the end-of-day batch measurement
owned by the coordinator, not this lane.

> **DEBT PAID 2026-08-31, owner released the rig.** Every row below that read DEFERRED has now RUN
> on the final tree (5848b3d0c). The table is rewritten in place with the measured verdict; the
> deferral text is kept in each cell so the reason it existed stays readable. The two doors STILL
> stay default OFF — that was never contingent on these gates (the identity is structural and the
> box prices the flip); green here clears the bringup merge, nothing more.
>
> One arm did NOT come back clean on the first pass and it is the most important line in this
> lane: `glm5-spec-ppn-gate [E forced-rejection sweep K=7]` is **load-sensitively
> nondeterministic, and NOT because of this lane's doors** — see the flake block under the table.

| item | state |
|---|---|
| `glm5_dedup_sched_gpu` 6/6 | **GREEN ON THE FINAL TREE 2026-08-31** (5848b3d0c). Re-run paid: `flock /tmp/memra-5090.lock env NVIDIA_TF32_OVERRIDE=0 timeout 2400 nice -n 5 cargo test -p memra-engine --test glm5_dedup_sched_gpu -- --include-ignored --test-threads=1 --nocapture`, `exit=0`, `6 passed; 0 failed; 0 filtered out`, and every PASS line byte-for-byte the pre-hold transcript (28-arm gate/up, 42-arm pair, 3 inert shuffles, both non-permutation reds, the dropped-macro red, counters flat on all three refusals). The two mechanical edits (`bool::then` -> `then_some`, one `cargo fmt` line-wrap) are therefore certified inert. Receipt: `receipts/glm5_dedup_sched_gpu.console` (pre-hold) + `/tmp` console re-run captured into the debt-lane report. |
| standing batteries (19 suites DEFAULT + walk suites on a COMPOSE arm with doors E/E-down ON + a door-D compose arm + a down-only arm), `--include-ignored` | **ALL 34 SUITE RUNS PASS 2026-08-31**, `receipts/run-battery.sh` exit 0, "dedup battery: ALL SUITES PASS". Per-suite logs in `receipts/*.log`. Every log carries `passed_total=` and the minimum across all 34 is **2**, so no suite was vacuously green. The runner was AMENDED during the debt run: its `run()` treated `rc=0` as PASS, which is exactly the `ok. 0 passed; 3 filtered out` shape its own header warns about; it now asserts a non-zero passed count and that assertion's failure path was exercised on a synthetic log before use. |
| split matrices (`glm5-spec-ppn-gate` 13 arms, `glm5-hyper-ppn-gate` 12, `glm5-hyper-batch-gate` 12) + dedup compose arms | **37/37 ARMS PASS in a settled window 2026-08-31**, runner banked as `receipts/run-matrices.sh` (new). PASS-line counts are ASSERTED per gate (23 / 6 / 3), not merely printed — probed against reality first. Compose classes E (doors alone), E+D+H (the shape the box prices) and a doors-pinned-`=0` base arm, never unset. FIRST PASS WAS 35/37: the two stages=2 spec-ppn compose arms failed on the flaky arm described below, and both re-ran 5/5 PASS at 23/23 once the window settled. Both the failing logs and the settled re-runs are banked (`receipts/ppn-gate/`, `receipts/ppn-gate/flake-20260831/`) — the failures are NOT deleted. |
| `memra-server` suite | **492/492 PASS 2026-08-31**, exit 0. |
| `tools/local-ci.sh --perf` | **RAN 2026-08-31 on the merge target and GREEN.** The skip is paid, not carried: the end-of-day batch ran `tools/local-ci.sh --perf` on `origin/lane/glm53-flash-bringup` @ 92ea07376 — correctness stage GREEN, perf stage **0 fail, 0 warn**, `qwen9b-plain-short 136.69 tok/s [OK]` inside the last measured-green band (~135-138.6), row banked `window_clean=true` load 5.62. The original push's `MEMRA_SKIP_PERF_CI=1` row in `.git/memra-gate-skips.log` (14:32:19Z, this lane's engine files) is what pointed the batch at this tree — the audit trail did its job and the deferral did not outlive its cause. A skipped gate was never recorded as a PASS; this row now records a real one, with the invocation named. The public-boundary scan DID run on the original push: 0 matches, 0 grandfathered. |
| `cargo clippy --all-targets` | **ZERO lints** on the final tree (`nice -n 19 -j 4`). |
| `cargo fmt --all --check` | clean on the final tree. |
| `tools/check-flags.sh` | **748 runtime literal reads, no uncovered names, no grandfather list** — both new flags resolve against `docs/FLAGS.md`. |
| CPU gate `avoided_slab_reads_equal_visits_minus_distinct_on_planted_overlaps` | PASS on the final tree (CPU-only, no rig). |

### THE FLAKE, and why it is not this lane's (measured 2026-08-31, debt run)

`glm5-spec-ppn-gate` arm **`[E forced-rejection sweep K=7]`** — the arm that forces a rejection at
`j_target = round % k` so every partial-keep rollback cycles, then asserts `out == tape` — is
**nondeterministic under host CPU load**, and the signal is perfectly bimodal:

| accepted | verdict |
|---|---|
| **14/42** | PASS, tape byte-identical |
| **13/42** | FAIL, tape diverges |

One acceptance is LOST and the continuation shifts. Nothing else in the 23-line gate moves.

Its detail string is a **static format that says "tape identical" even on the FAIL line**, so the
log reads like a contradiction (`gate FAIL [...]: ... tape identical (13/42)`). The assertion is
real (`out == tape`, `glm5_spec_ppn_gate.rs:707`); only the message is misleading. Fixing that
string is follow-up 6 below — a FAIL line that describes a PASS is how a real red gets waved past.

Attribution, measured rather than assumed — 12 reps per cell, `flock`-serialized, capped
`nice -n 19` spinners as the load source (`receipts/ppn-gate/flake-20260831/`):

| tree | doors | stages | fails/12 |
|---|---|---|---|
| `lane/glm5-dedup` @ 5848b3d0c | pinned **=0** (control) | 2 | **5/12** |
| `lane/glm5-dedup` @ 5848b3d0c | E + E-down **ON** | 2 | 2/12 |
| `lane/glm5-dedup` @ 5848b3d0c | pinned =0 | 3 | 2/12 |
| `origin/lane/glm53-flash-bringup` @ 92ea07376 (**no dedup content at all**) | n/a | 2 | **1/12** |
| any of the above, settled window (no artificial load) | either | 2 or 3 | **0/34** observed |

Three things follow, and the first is the one that matters for the merge:

1. **The doors are exonerated.** The doors-pinned-`=0` control on this tree fails at a HIGHER rate
   (5/12) than the doors-ON arm (2/12), and the same failure reproduces on the MERGE TARGET with
   none of this lane's code present. Merging this lane does not introduce the defect; the defect
   is already on the branch being merged into. This is the pin-against-truth-not-siblings shape
   from the other direction: my two compose arms were being read against sibling base arms that
   share the same flaky substrate, and the sibling comparison would have blamed the doors.
2. **It is a real engine-side nondeterminism, not a harness artifact**, and it is present at
   stages=3 — the glm5 SERVING shape. A lost acceptance is the drift class `tools/local-ci.sh`
   exists to catch (acceptance 1.000 -> 0.669 across ~40 green-gated commits, header of that
   script). It is invisible to every unloaded gate run, which is why 8 base arms and ~34 settled
   reps never saw it. Escalated to the coordinator as its own lane; NOT patched around here, and
   NOT reverted.
3. **stages=1 is an expected refusal, not a data point.** `1 24 20` panics
   `ppn door failed to open (n_layers=4, stages=1)` on every rep; it was mine to discard.

**The doors stay default OFF.** The reason has changed and is worth stating exactly, because the
original reason has now expired (exception-lists-need-expiry). It USED to be "until the deferred
gates run": door E's identity was proven across 70 GPU arms, but the composed WALK gates
(`glm5_tparallel_verify_gpu`, `glm5_verify_batch_gpu`, the ppn matrices) are what prove the door
does not perturb a walk it must not touch, and they had not run on this tree. **They have now run,
green** — battery phases 2-4 (15 walk-suite runs with doors E/E-down ON, with door D, and down-only)
and the spec-ppn compose arms. So that condition is DISCHARGED. The doors remain OFF on the
surviving, independent ground: **the box prices the flip.** The win is a block-dispatch-order
property whose size only a real-artifact box row can measure, the rig is exactness-only (no timing
number may be read from any log in this lane), and per the new-flags law a lane may ship default-ON
only with receipts attached — those receipts are a box measurement this lane never had access to.

**Box B was NOT touched.** The coordinator offered a co-tenant CPU build there; I declined on two
independent grounds. (a) The standing owner ban on inspecting, SSHing to, or acting on a user-owned
accelerator lane after handoff applies unless the OWNER delegates the specific action in the current
task — an agent-relayed offer is not that delegation, and I surfaced the conflict rather than
quietly acting. (b) Independently, `../struct-battery-20260831/WINDOW.md`'s own status log (read
locally, no SSH) records a TIMED interleaved placement A/B launched on box B with the marker raised
at ~12:56Z; co-tenanting anything onto that window risks invalidating the very cell-2 measurement
that justifies this lane.

---

## 7. NAMED FOLLOW-UPS (not built here)

1. **Fold `moe_vrows_order_from_sel` into `moe_vrows_tables_from_sel`.** Same inputs, same
   one-thread-per-pair grid, and the rank loop fits the existing thread mapping exactly. Recovers
   the door-D arm's 42 launches/round = **-0.093 ms/round**. Not taken here because it changes
   door D's kernel signature and therefore its just-shipped gate, and door D is a live 1.0154x
   win — a signature change to a winning door belongs in its own diff, not riding a schedule lane.
2. ~~**Re-run the deferred batteries and re-price** (§6).~~ **BATTERIES DONE 2026-08-31** (§6, all
   green on the final tree). The RE-PRICE half stands and is box work: the doors cannot flip before
   a real-artifact box row, per the closing paragraph of §6.
3. **The shared-slab twin's gate/up half**, on the §5 trigger only.
4. **Price door E composed with door R** (`MEMRA_BF16_TCOLS_RED_FUSED`, moe-loc §2.2, -1.0 to
   -2.0 ms/round). They are disjoint kernels on the same round, so the composed band is
   ~77-79 tok/s and the 100 bar still needs ~1.27-1.30x — worth stating so the composition is
   not oversold.
6. **`glm5-spec-ppn-gate`'s ppN partial-keep-rollback nondeterminism** (§6 flake block) — a lost
   acceptance (14/42 -> 13/42) under host load, at stages=2 AND stages=3, reproducing on
   `origin/lane/glm53-flash-bringup` @ 92ea07376 with none of this lane's code. Needs its own lane:
   it is on the serving shape, it silently costs acceptance, and no unloaded gate can see it. Two
   sub-items that are cheap and should ride along: make the arm's detail string report the actual
   comparison instead of a static "tape identical" (today a FAIL line describes a PASS), and print
   the first divergence index so a bisect has something to anchor on.
7. **`moe_vrows_pack_on()` is still an UNCACHED `std::env::var` per launcher call** (moe-loc
   follow-up 5, now read by this lane's refusal predicates as well, so the environ-scan count per
   round is unchanged but the reason to cache it is stronger). Needs its own FLAGS amendment.

---

## 8. STATUS LOG

- Lane open 2026-08-31 from `origin/lane/glm53-flash-bringup` @ b55536ab1 (fetched first). No
  predecessor commits at `~/projects/wt-glm5-dedup`; opened fresh.
- §1 mechanism written from the banked moe-loc constants + the struct-battery cell-2 measurement
  BEFORE any code change. Finding recorded there: the charter's slab-residency framing is the WEAK
  version of the lever, and transposing the grid makes the per-wave L2 footprint IDENTICAL while
  cutting the reuse distance from 2048 blocks to ~1.
- BUILT: kernels `moe_gate_up_preclamp8_q8_rows_ord`, `moe_down8_fma_q8_rows_tmaj`,
  `moe_vrows_order_from_sel` (qmatvec.cu); flags, three counters, the host order build and the
  order launcher (lib.rs); the fourth table plane and both arms' order build (hybrid_forward.rs);
  new gate suite `glm5_dedup_sched_gpu`. FLAGS.md 2 rows + KERNELS.md 2 rows in the same commit.
- `docs/KERNELS.md` qmatvec count corrected 323 -> **317** and put on a stated, reproducible basis
  (a measured `extern "C" __global__` count: 314 at the base + this lane's 3). The old number was
  unreproducible and already drifting; the other files' counts still are not measured, which is the
  vrest lane's named recount follow-up.
- Gate suite 6/6 GREEN on the rig, including two rounds of fixing a FALSE-NEGATIVE red arm (§3.1).
- Rig-hold order received; standing batteries and local-ci `--perf` DEFERRED and marked per order
  (§6). Box B declined with the conflict surfaced (§6).
- Committed on `lane/glm5-dedup`; pushed with `MEMRA_SKIP_PERF_CI=1` under the owner authorization.
  No self-merge. The box owns every flip decision and the deferred-gate re-run gates the flip.
- **2026-08-31, owner released the rig — end-of-day debt lane paid every deferred row (§6).**
  Order of work: perf batch on the MERGE TARGET first (clean attribution before this lane's content
  landed on it) -> `glm5_dedup_sched_gpu` re-run on the final tree -> 34-run standing battery ->
  37-arm split matrices -> server suite -> clippy -> fmt -> check-flags.
  Verdicts: gate 6/6, battery 34/34, matrices 37/37 (settled; 35/37 first pass), server 492/492,
  clippy **zero lints**, `cargo fmt --all --check` clean, check-flags 748 reads / no uncovered
  names / no grandfather list, `local-ci.sh --perf` 0 fail 0 warn @ 136.69 tok/s.
  Two method corrections made in-lane rather than noted for later: `run-battery.sh` was asserting
  only `rc=0` (it now asserts a non-zero passed count, failure path exercised first), and the
  clippy "zero lints" claim initially rested on a 0.44 s fully-cached replay — it was re-taken
  after touching the lane's own sources, and the gate's teeth were proven by injecting a
  `len_zero` lint, watching it get caught, and restoring the tree.
  ONE genuine finding, and it is not this lane's: the ppN partial-keep-rollback flake, measured to
  the merge target and escalated (§6 flake block, follow-up 6). Not reverted, not patched around.
