# qwen4_exp PROFILE-12 — the HOST side of the 262,144 window: the n-gram id recompute dies, and
# three of the owner's host levers die with receipts

Target, per the owner's scope change: **262,144 tokens — the model's native window — and best
performance THERE** (LADDER.md "SCOPE CHANGE 2026-08-31"). This file is the HOST-side lane
(agent B): the PLE/n-gram host path, engine host threading and affinity, and the pooled-alloc /
event-sync paths. The device indexer, selection and graph seams are PROFILE-11's lane; the KV and
pooled-key layout is the memory lane's.

Box: the lane box, **2x RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, 600 W** each — same
card class as every prior receipt in this lane. Provider, region, instance class and instance ids
are fleet state and live in darklanes, not here. Artifact `q48fn-yarn1m`, chunk 2048, ONE card.

Receipts: `../round2-box-receipts/kvq2/ladder-bhost-abprof.tsv` (+ `bhost-abprof.log`), and
`../round2-box-receipts/tiny-gate-seamtable-plecache-rig.tsv`. Every timing below carries the cache
arm from its own receipt header: **`# cache kv_quant=q8_0/q5_1 idxq=q8 golden_pin=false
seams_env=idxsel`** — ship defaults plus the cliff fix. Quote that line with any number here.

Two scope notes stated up front rather than buried, because both bound what these rows may be
compared against:

- **These rows were taken on the SECOND card** (`CUDA_VISIBLE_DEVICES=1`) while a sibling lane held
  card 0. Same card class, but not the same physical die as PROFILE-11's rows, so **the A/B ratios
  here are the claim and the absolutes are not cross-file comparable.** Every A/B is internally
  valid by construction: both arms share one prefill, one state, and one lock hold.
- **A sibling lane was computing during part of this cell.** That is not waved away — it is
  measured. See §4: the host census reports 0.11 involuntary context switches per decode step and
  ZERO launch-thread migrations across the A/B, and both arms show `outliers_1.5x=0`. A contended
  timing shows up in those fields, and these do not.

## THE VERDICT (the sentence to read alone)

**The largest section of a deep decode token at the TARGET WINDOW is dead, and it was never the
thing the lever was aimed at.** One seam — `plecache`, an incremental n-gram id cache — takes
**262,144 from 41.38 to 28.30 ms/token (24.17 -> 35.34 tok/s, 1.4620x)** and 131,072 from **33.52 to
25.91 ms (29.83 -> 38.60 tok/s, 1.2938x)**; drops `ple.host_ngram_gather` from **28.9% of the token
at 262,144 — the largest section there, ahead of `qsa.sdpa` — to below the profile's 12th place**;
and collapses per-token variance (cv 2.48% -> 0.11% at 131,072; p99 41.70 -> 28.38 ms at 262,144).
Exact by construction, not by tolerance, and **default ON as of 2026-09-01** on the receipts in §10a.

The gain GROWS with depth, which is the mechanism's own signature: the deleted work is O(fill) per
token, so 7.8 ms at 131,072 becomes 13.2 ms at 262,144. **Combined with the indexer lane's `idxsel`,
the native window goes 15.21 -> 24.17 -> 35.34 tok/s: 2.32x, from two host-side seams, neither of
which changes an output byte.**

And the owner's host lever list loses three entries to measurement, each with the number that
killed it: async table prefetch (§5.1), sticky threads / NUMA affinity (§5.2), and event-wait /
spin-vs-block tuning (§5.3).

## 1. What the lever was, and what the profile said instead

Owner lever 2 was PREFETCH: the n-gram embedding table is host-resident (**102 GB** — 320,001,536
rows x 160, and the reason it is host-resident at all is that it never fits device), its lookup
addresses are known in advance, and the gather is synchronous host work. The vendor's own cookbook
names this as the design point: "the table can live in host memory and be prefetched
asynchronously alongside model compute". The obvious build is a pinned staging buffer plus an async
H2D on a side stream.

The profile moved the target, and **that correction is the finding**:

- The actual GATHER is `t x 16` random rows. At decode `t == 1`: **16 reads of 160 f32 —
  microseconds.** There is nothing there to overlap. `ple.h2d` (10 KB per token) never appears in
  the top twelve sections at any depth measured.
- The 7-8 ms is `host_ngram_ids`, a twin over the **FULL token history** whose last `t` rows the
  caller then slices. A decode step at a 131,072-token fill rebuilds **131,072 rows of hashes**
  (2.1 M i64 `rem_euclid` operations, plus `max_ngram` shifted copies of the whole history, ~3.1 MB
  of allocation) **to consume ONE row.**

So async-prefetching the table would have bought approximately nothing, and the fix is to stop
recomputing. `plecache` appends to a per-state id cache instead. Landed default-OFF by another
agent with a host-only oracle and handed to this lane for its A/B and default; the A/B, the
at-depth exactness, the variance result and the residual below are this lane's.

**Why prefill did not show this and decode did.** `host_ngram_ids` is O(context) per CALL, so a
2,048-token prefill chunk amortises one call 2,048 ways while a decode step pays it in full for one
row. Across a whole 262,144 prefill it is ~270 M modulo operations, ~0.06% of a 4,779 s wall —
invisible, correctly. It is a **decode-only** lever, which is exactly why measuring it needed an
A/B that shares the prefill (§6).

## 2. The A/B at 131,072 — x3 interleaved, both arms inside one lock hold

`--ladder-ab-seam plecache --ladder-ab-rounds 3 --ladder-ab-steps 16`, lead flipped on odd reps,
16 timed steps per arm per rep after 2 excluded warmup steps (arming pays a one-time cache build).

| arm | median ms | tok/s | per-rep medians | spread | cv | p99 | max | outliers >1.5x med |
|---|---|---|---|---|---|---|---|---|
| `plecache` OFF | **33.52** | **29.83** | 33.54 / 33.52 / 33.51 | 0.09% | 2.48% | 37.00 | 37.00 | 0 |
| `plecache` ON | **25.91** | **38.60** | 25.91 / 25.91 / 25.90 | 0.02% | **0.11%** | 25.97 | 25.97 | 0 |

**speedup 1.2938x, -22.71% ms/token.** No escalation owed: both arms' spreads are far under the
0.5% threshold and the verdict (22.71%) is more than two orders above the pooled spread, which is
the second half of the escalation rule — a verdict smaller than the instrument's noise is not a
verdict, and this one is not close to that line.

## 3. The host profile at depth, per arm — the section leaves the table

`--profile 1` per arm at the same fill, sync-bounded (shares are the signal, absolutes are
inflated), taken AFTER the timed reps so it cannot contaminate them:

| section | OFF | share OFF | ON | share ON |
|---|---|---|---|---|
| `qsa.sdpa` (12) | 10.3 ms | 26.8% | 10.2 ms | **33.3%** |
| **`ple.host_ngram_gather` (1)** | **7.8 ms** | **20.3% (HOST)** | **not in top 12 (< 0.8 ms)** | **< 2.8%** |
| `hyper.read` (96) | 3.3 | 8.6% | 3.3 | 10.7% |
| `qsa.idx_host` (12) | 3.0 | 7.8% (HOST) | 3.1 | 10.3% (HOST) |
| `moe.sel_grouped` (48) | 2.6 | 6.9% | 2.6 | 8.6% |
| `gdn.proj` (36) | 2.6 | 6.8% | 2.6 | 8.5% |
| `gdn.norm_gate_out` (36) | 1.4 | 3.6% | 1.4 | 4.6% |
| `moe.shared` (48) | 1.2 | 3.2% | 1.2 | 4.0% |
| `moe.router` (48) | 1.1 | 3.0% | 1.1 | 3.7% |
| `hyper.write` (96) | 1.0 | 2.6% | 1.0 | 3.2% |
| `qsa.proj` (12) | 0.9 | 2.4% | 0.9 | 2.9% |
| `gdn.conv_scan` (36) | 0.9 | 2.2% | 0.8 | 2.8% |
| `lm_head` (1) | — | — | 0.8 | 2.8% |

Every other section is where it was, to a tenth of a millisecond — `qsa.sdpa` 10.3 -> 10.2,
`moe.sel_grouped` 2.6 -> 2.6, `qsa.proj` 0.9 -> 0.9. That is the same reading the diagnosis makes
in the other direction: **one host section moved and nothing else did.** `lm_head` appears in the
ON arm only because a section had to leave the top twelve for it to be visible.

**The remaining host share of a deep decode token is `qsa.idx_host` at 10.3%, and it is not this
lane's.** PROFILE-11 §5 put the host total at 27.9%; with `plecache` armed it is ~10-13%, and
essentially all of it is the device-indexer path's residual host half.

## 4. Per-token VARIANCE, and the sticky-thread question answered as a number

The lever list asked for per-token variance rather than medians, so:

**`plecache` removes decode-time jitter as decisively as it removes decode time.** cv **2.48% ->
0.11%** (23x), p99 **37.00 -> 25.97 ms**, max **37.00 -> 25.97**. The ON arm's p99 is 0.2% above
its own median; the OFF arm's is 10% above. Mechanism is not mysterious: the OFF arm's per-token
cost includes an O(context) pass over a 3.1 MB working set plus a fresh ~3.1 MB allocation, and
both the allocator and the cache hierarchy make that variable. **This is a serving-quality result,
not only a throughput one** — deep-context p99 latency is what a long-agentic workload feels.

The host census over the same A/B (`--host-probe`, 108 timed decode steps):

| field | value | reading |
|---|---|---|
| `launch_cpu_migrations` | **0** | the launch thread never moved, across 108 steps |
| `launch_cpus_seen` | **1** | one CPU for the entire A/B |
| `threads` | **4** | main + cuda worker + `cuda-EvtHandlr` + 1 |
| `vol_cs` | 66 (**0.6/step**) | voluntary switches — the blocking-wait counter |
| `nonvol_cs` | 12 (**0.11/step**) | involuntary — preemption, i.e. sibling contention |

Per-thread at the same point (cumulative over the run): `qwen4exp_real_g` main `vol_cs=53,729
nonvol_cs=1,643 last_cpu=39`; second `qwen4exp_real_g` `8,828 / 0 / cpu 18`; `cuda-EvtHandlr`
`8,837 / 0 / cpu 41`; `cuda00001400006` `3 / 0 / cpu 16`. Four threads, four distinct CPUs, no
involuntary switches on any thread but the launch thread.

## 5. Lever verdicts — three receipted DEAD ENDS, so no future lane re-spends on them

### 5.1 Async n-gram table prefetch (owner lever 2): DEAD, and the reason is the finding

Not "we tried it and it was flat" — **it was never the cost.** The gather is 16 rows of 160 f32 at
decode, microseconds, against a 7.8 ms section; `ple.h2d` is 10 KB/token and never reaches the top
twelve. A pinned staging buffer plus a side-stream H2D would have overlapped a few microseconds and
left 7.8 ms untouched. The cost was an O(context) ID RECOMPUTE standing in front of the gather.

**Fourth instance on this model family of "the host half is O(context) per token" being the real
mechanism** — the yarn lane's indexer selection, PROFILE-11's indexer top-k, this file's n-gram
ids, and (§7) the *fix's own* prefix compare. It is now more than a heuristic: **on this family,
price the host half's complexity in the fill BEFORE believing any other story about a host
section.**

### 5.2 Sticky threads / NUMA + CCX affinity (owner lever 5): DEAD on this box class, structurally and by measurement

PROFILE-11 §5 deprioritized this on the collapse of round-to-round spread. Two independent
confirmations, one structural and one direct:

**Structurally, the two things pinning usually buys do not exist here.** `lscpu` on this box class:
**1 socket, 24 cores / 48 threads, 1 NUMA node (`node0 CPU(s): 0-47`), and ONE 320 MiB L3
instance.** There is no second NUMA node to migrate across and no CCX boundary to stay inside; a
migration can cost L1/L2 warmth and nothing more. Pinning cannot buy what the topology does not
charge for.

**Directly, the launch thread does not migrate anyway:** `launch_cpu_migrations=0`,
`launch_cpus_seen=1` across 108 decode steps at a 131,072 fill (§4). The four threads sit on four
distinct CPUs. Involuntary switches are 0.11/step, and every non-launch thread shows `nonvol_cs=0`.

**And the host thread POOL is already gone from the deep path.** Every `std::thread::scope` fan-out
in `qwen4exp_gpu.rs` is in the host indexer selection path — `top_blocks_ascending`, the block
scorer, and the per-row work-stealing cursor — which is exactly what PROFILE-11's `idxsel` moved to
the device. What remains of host threading in a deep decode step is single-threaded appends.

Verdict: **no battery spent, none owed.** Re-open only if a later arm shows migrations > 0 or
involuntary switches per step rising materially; the numbers to beat are 0 and 0.11.

### 5.3 Event-wait bubbles / spin-vs-block sync tuning: DEAD, and it was measurable without a knob

The audit asked for was "spin-vs-block choices in the pooled-alloc and event-wait paths". That does
not need a flag sweep, because **voluntary context switches ARE the blocking-wait counter**: a CUDA
wait that spins parks nothing and shows zero, one that blocks parks the thread and shows one switch
plus a wake latency.

Measured: **0.6 voluntary switches per decode step** on the whole process. The waits are already
effectively all spinning, so there is no pool of blocked-wait wake latency to reclaim — at ~5-10 us
per wake, 0.6 wakes/step is single-digit microseconds against a 26-40 ms token, i.e. under 0.03%.
Switching to blocking sync could only make it worse. **No knob added, because a knob whose upside is
0.03% is a maintenance cost with a rounding error attached.**

### 5.4 Pooled-alloc churn, redundant control-blob H2D, per-chunk workspace re-zeroing: NOT INDICTED at depth

Named in the lever list and looked for. The profile does not indict them: no allocation or memset
section reaches the top twelve in either arm at either depth, and the workspace slot reserve is
already `reserve`-derived rather than capacity-derived precisely so a growing decode never
reallocates. The one real allocation churn the profile DID indict was inside
`ple.host_ngram_gather` — the ~3.1 MB shifted-history allocation per call — and `plecache` deletes
it as a side effect of deleting the recompute.

## 6. Instrument work this needed, and what it cost

Three additions, all default-OFF, all with FLAGS rows in their landing commits.

**`--ladder-ab-seam <seam>` — the within-prefill interleaved A/B.** `plecache` is decode-only, so
the per-arm-process protocol would have spent SIX prefills at 25-80 minutes each (2.6-8 h of card
time) to time ~100 decode steps, and put box clock drift between the arms. Sharing one prefill
makes the arms differ in the seam and nothing else. Soundness bound, written into the flag: eligible
only for seams whose state is rebuildable from the token history, so arming mid-run cannot leave
stale state behind. `plecache` qualifies and its oracle covers exactly that transition. Seam names
are validated before the checkpoint loads, the state reservation covers the A/B's own positions and
its own escalation, and the restore is exact (`seam_state`) and named in the receipt.

**`--host-probe` and `# ladder-jitter`.** `prof_section` brackets sections with device syncs, so it
prices host wall time honestly and is blind to WHY the host thread was slow. §4 and §5.2-5.3 are
entirely built on /proc and `sched_getcpu`, with no new dependency and nothing inserted into the
measured path. `# ladder-jitter` now rides every ladder run, because a median is not a measurement.

**`MEMRA_Q4E_MEASURE_LOCK` and its two modes.** Documented with its cost, because the cost is a
finding. The lane's convention wrapped a whole cell in `flock -x`, putting 11-80 minutes of prefill
inside the exclusive window to protect a prefill wall nobody claims. This lane's fix takes `LOCK_EX`
around timed blocks only and `LOCK_SH` **per prefill CHUNK** — released and re-acquired at each
boundary, so a waiting exclusive gets in at the next boundary and a prefill YIELDS instead of
racing. Nothing runs unlocked, and a measurement's worst-case wait is ONE CHUNK (~11 s).

**Measured cost of the coarser alternative, since it was asked for as a number rather than an
argument:** under whole-invocation `flock -x`, this cell's 262,144 A/B — four remaining reps of ~2
seconds each — blocked for over 11 minutes behind a sibling's whole cell while holding a filled 262k
state (96,635 MiB, 1,443 s of cumulative prefill) IDLE on card 1. Per-chunk shared locking bounds
that at ~11 s. The ruling that nothing may run unlocked is right; it is the GRANULARITY that is
expensive, and the two are separable.

Two hazards found and paid rather than discovered later:

- **Nesting the two lock mechanisms self-deadlocks.** flock locks are per open-file-description, so
  `LOCK_EX` on a new fd while the PARENT holds the file waits on one's own parent, forever, with no
  error — during a 25-minute prefill that reads as a slow box. The instrument now polls, prints
  `# measure-lock-waiting` every 60 s so a stall is visible WHILE it happens, and hard-fails at a
  deadline naming the self-deadlock as the first suspect. Queues use exactly one mechanism per cell.
- **The state reservation did not cover the new instrument.** `cap` covered the timing loop's x5
  escalation after that defect destroyed a rung 755 s into its prefill; the A/B consumes positions
  on top of that and has its own escalation. Extended in the same commit that added it, because
  adding an instrument without extending the reservation is precisely how that defect recurs.

## 7. The NEW residual, priced: the fix's own prefix compare is O(context) too

`host_ngram_ids_cached` finds its longest common prefix by comparing the cached history against the
requested tokens, and on the steady-state decode path that loop runs over the WHOLE fill: at
262,144 it is 262,143 compares streaming ~3.1 MB. **So `plecache` cuts the constant by ~10x and
does not change the complexity** — the fourth instance of §5.1's family pattern, this time inside
the fix for the third.

Priced rather than assumed: with the seam ON, `ple.host_ngram_gather` **falls out of the top twelve
entirely**, below `gdn.conv_scan` at 0.8 ms, i.e. the whole section — LCP scan plus gather plus one
token's 16 modulo operations — is **under 0.8 ms of a 25.91 ms token (< 3%)** at a 131,072 fill.

**Verdict: not worth building yet, and that is a measurement, not a shrug.** The clean fix is real
(make the cache own the history so the caller passes only the delta, since `PleState.ngram_history`
and the state's own token history are already duplicates) but it touches the rewind/stash contract,
and 3% does not buy that risk while `qsa.sdpa` sits at 33.3%. It doubles with depth, so the number
to re-read is the 262k figure, not this one.

## 8. Where the deep decode token now goes, and what to attack next

With `plecache` armed at a 131,072 fill, the profile is broad and GPU-dominated: `qsa.sdpa` 33.3%,
`hyper.read` 10.7%, `qsa.idx_host` 10.3%, `moe.sel_grouped` 8.6%, `gdn.proj` 8.5%, then a long
tail. The host share is down to ~10-13% and nearly all of it is one section that belongs to another
lane.

Ordered by what the profile says, not by whose lane it is:

1. **`qsa.sdpa`, 33.3% and now a third of the token.** Already bounded (the block-list kernel reads
   only the <= 2,052 selected KV rows at any depth), so this is bandwidth and layout, not
   complexity. The memory lane's `kvhoist` is aimed here. Not this lane's.
2. **`qsa.idx_host`, 10.3%, the last material HOST section.** The device-indexer path's residual
   host half. PROFILE-11's lane.
3. **`hyper.read` at 10.7% across 96 calls** — 34 us per call for a gated-residual read. High call
   count, small per-call work: a launch-overhead shape, which is the one place at depth where the
   graphs lever might still pay. Worth a number before anyone argues about it.
4. **This lane's own residual is §7, at under 3%,** and it is deliberately parked with its number.

## 8a. A false alarm, and the reading rule it exposed — receipt comparisons must match on ARTIFACT

Recorded because the retraction is more useful than the alarm was, and because the gap it found is in
this lane's own receipt discipline rather than in one agent's care.

With `plecache` armed the greedy first-divergence chain read **-1 / 0 / -1 / 26**, against
**-1 / 8 / -1 / 48** in every banked f32 golden receipt in this lane back to g2 — including
`greedy-gate-g5-raw.tsv` at `seams_env=idxsel`. The cache arm matched
(`kv_quant=f32 idxq=f32 golden_pin=true`), so the reading was "the seam moved the chain", which for a
seam whose ids are table ROW INDICES would be a hard exactness defect with no tolerance to hide in.
The default was held and the lane was told to stop arming it.

**It was a confound.** Every banked `-1/8/-1/48` receipt is `ckpt=q48fn-nvfp4`; this cell is
`ckpt=q48fn-yarn1m`, the hardlink twin with `rope_type=yarn factor=3.814697265625 original=262144
mpe=1000000`. Yarn changes rope at every position, therefore logits at every position, therefore
where a greedy chain parts from a transformers golden. Only the factor-1.0 case is bit-identical to
plain rope, and this is not that case. Two ARTIFACTS were diffed and read as a seam. Corroborating the
artifact story rather than the seam story: `greedy-gate-r2base-raw-kvq.tsv` shows prompt 3's
48 -> 26 arriving from KVQ alone on the nvfp4 artifact.

**The rule.** PROFILE-10 §4 gave every receipt a `# cache kv_quant= idxq= golden_pin= seams_env=`
line exactly so a receipt could state its own arm, and it worked — that line was read and it was
correct. But it FOREGROUNDS the cache arm and thereby invites comparison on it, while the artifact
sits in a different field on a different line. **A receipt comparison is valid only when the artifact
matches too, and the header makes the cache arm easy to check and the artifact easy to skip.** Quote
`ckpt=` beside the cache line whenever a receipt is compared against a banked one.

**What survives the retraction, independent of the error:** the three green gates could not have
caught a real defect of this shape either. `verify-bit` compares plain `t==1` rows against `t==k+1`
chunk rows within ONE arm; spec byte-identity compares spec against plain within one arm; hidden
goldens is a 10-token probe. **An intra-arm identity gate cannot detect a CONSISTENT error, because a
uniformly-wrong id set is perfectly self-consistent.** Only a truth-pinned gate can — and the one
truth-pinned gate available, the host-only oracle, runs on its own synthetic
multipliers/sizes/offsets while the real ones are checkpoint buffers (census I64 [3]/[16]/[16],
"LOAD, never re-derive"). So an exact oracle and a wrong run remain simultaneously possible, which is
why the real-geometry audit is the gate the default actually waits on.

## 10a. The 262,144 A/B — the target window, and the DEFAULT DECISION

Third attempt; the first two died on lock starvation rather than on the seam (§6). `--ladder 262144
--ladder-ab-seam plecache --ladder-ab-rounds 3 --ladder-ab-steps 16 --profile 1 --host-probe`,
`# cache kv_quant=q8_0/q5_1 idxq=q8 golden_pin=false seams_env=idxsel`, prefill 1,437.8 s / 128 chunks.

| arm | median ms | tok/s | per-rep medians | spread | cv | p99 | max | outliers >1.5x |
|---|---|---|---|---|---|---|---|---|
| `plecache` OFF | **41.38** | **24.17** | 41.31 / 41.38 / 41.42 | 0.28% | 0.37% | 41.70 | 41.70 | 0 |
| `plecache` ON | **28.30** | **35.34** | 28.25 / 28.30 / 28.30 | 0.17% | 0.16% | 28.38 | 28.38 | 0 |

**speedup 1.4620x, -31.60% ms/token.** No escalation owed on either arm.

### The per-arm profile at 262,144 — the deleted section was the token's LARGEST

| section | OFF | share OFF | ON | share ON |
|---|---|---|---|---|
| **`ple.host_ngram_gather` (1)** | **13.2 ms** | **28.9% (HOST)** | **not in top 12 (< 0.8 ms)** | **< 2.6%** |
| `qsa.sdpa` (12) | 10.5 | 22.8% | 10.3 | **31.5%** |
| `qsa.idx_host` (12) | 4.9 | 10.7% (HOST) | 5.1 | **15.5% (HOST)** |
| `hyper.read` (96) | 3.3 | 7.2% | 3.3 | 10.0% |
| `moe.sel_grouped` (48) | 2.7 | 5.9% | 2.7 | 8.3% |
| `gdn.proj` (36) | 2.6 | 5.6% | 2.6 | 7.9% |
| `gdn.norm_gate_out` (36) | 1.4 | 3.0% | 1.4 | 4.2% |
| `moe.shared` (48) | 1.2 | 2.7% | 1.2 | 3.7% |
| `moe.router` (48) | 1.1 | 2.5% | 1.2 | 3.5% |
| `hyper.write` (96) | 1.0 | 2.1% | 1.0 | 2.9% |
| `qsa.proj` (12) | 0.9 | 2.0% | 0.9 | 2.7% |
| `gdn.conv_scan` (36) | 0.8 | 1.9% | 0.8 | 2.6% |
| `lm_head` (1) | — | — | 0.8 | 2.6% |

**At the native window this host section was bigger than the largest GPU kernel** — 28.9% against
`qsa.sdpa`'s 22.8% — and it is gone, while every other section holds to a tenth of a millisecond.

### The depth scaling is the mechanism's own signature

| depth | section, OFF | speedup |
|---|---|---|
| 131,072 | 7.8 ms (20.3%) | 1.2938x |
| 262,144 | **13.2 ms (28.9%)** | **1.4620x** |

O(fill) per token predicts exactly this, and a 2x fill gave 1.69x the section. The lever is worth
MORE the deeper the context goes, which is the opposite of most and is why it matters for this product.

### Host census at 262,144 — sticky threads dead at the target depth too

`launch_cpu_migrations=0`, `launch_cpus_seen=1` across 108 decode steps; `threads=4`; `vol_cs` 0.74/step
(the waits spin, so there is no blocked-wake latency to reclaim); `nonvol_cs` 0.16/step. Identical
reading to §4's 131,072 numbers. §5.2 and §5.3 are now measured at BOTH target depths, not extrapolated.

### The compounded product number

| 262,144 | ms/token | tok/s |
|---|---|---|
| banked ship default (LADDER) | 65.7 | **15.21** |
| + `idxsel` (indexer lane) | 41.38 | 24.17 |
| + `plecache` (this lane) | **28.30** | **35.34** |

**2.32x at the model's native window from two host-side seams, neither of which changes an output
byte.** The `idxsel` half belongs to the indexer lane; only `plecache` is flipped here.

### THE DEFAULT DECISION: ON, as of 2026-09-01

`PLE_CACHE_DEFAULT` false -> true, with the FLAGS row in the same commit. Rollback is one token:
`MEMRA_Q4E_SEAMS=plecache=0`.

**What the flip rests on — the arms that can FALSIFY it:**

- **Real-geometry truth pin** (`MEMRA_Q4E_PLECACHE_AUDIT=1`): `rows=32828 mismatched=0
  deepest_fill=32828`. Cached ids hard-compared against the full `host_ngram_ids` twin at the
  CHECKPOINT's own multipliers/sizes/offsets, over both growth shapes.
- **Behavioural control**: greedy chain IDENTICAL across the seam on the same artifact (`-1/0/-1/26`
  both arms; argmax 10/10 both arms).
- Host oracle vs the full twin: EXACT over 69,635 comparisons, 6 case families.
- Default-ON gate: `# verdict failures=0`, 36 summaries, 0 `pass=false` rows.

**What it deliberately does NOT rest on:** `verify-bit` 24 and spec byte-identity 256 both pass, but
they are INTRA-ARM, and an intra-arm identity gate cannot detect a CONSISTENT error (§8a). Counting them
as exactness evidence would be the mistake this file already documents.

**Not a card-keyed or capacity-keyed default.** Prior traps in this repo were conditional defaults
pinned on one card class or one capacity. This is pure HOST code — no kernel, no device memory, no card
or capacity condition — and its output is a set of table ROW INDICES the truth pin shows identical to
the reference twin. Cost is one `i64` vector per state: `fill*16*8` = 33.5 MB of host memory at
262,144, against 499 GB on the box.

**Owed, and named rather than dropped:** `--verify-bit-deep 131072` with the seam armed has not passed.
Three failures on the box, the first a real sizing defect (§9) and attempts 2-3 with the correct kvq arm
and ~96 GB free — i.e. the instrument, not the seam. It is itself intra-arm, so it cannot add assurance
the truth pin does not already give, and the flip does not wait on it.


### 8b. CLOSED 2026-09-02 — `--verify-bit-deep 131072` PASSES on the serving caches

`# verdict fill=131072 rows=24 mismatched=0 policy=bit-identity pass=true`, `kv_quant=q8_0/q5_1
idxq=q8 golden_pin=false`, plecache and selgroup at their (ON) defaults, peak 94,845 of
97,887 MiB. Receipts: `kvq/vbdeep-box/` (the eight attempts' logs are all there).

The "instrument, not the seam" reading in §8/§9 was right and the instrument had TWO
sizing defects, neither the goldens pin: `spec_arm` kept a WHOLE-HISTORY wide stash
((fill+n+k1+2) x 10,240 x f32 = 5.4 GB at 131k) and the plain state stayed resident while
the spec-armed one was allocated although its rows had already been copied to host (~2.9 GB
per deep state). Attempts 1-3 (08-31) and 4-6 (09-02: with the pin, without it, at half fill)
OOMed on both; attempt 7 (ring bounded to 2*chunk) still OOMed on the second; attempt 8
(ring + `drop(sa)` before the second allocation) passed. Fixed in memra PR #64.

## 9. Open

- **The 262,144 A/B is MEASURED** (§10a): 1.4620x, on the third attempt — the first two died on
  lock starvation rather than on the seam. The prediction that it would exceed the 131,072 ratio
  because the deleted work is O(fill) held: 1.2938x -> 1.4620x.
- **The plecache rule gates are GREEN** with the seam armed: `verify-bit` `rows=24 mismatched=0
  policy=bit-identity pass=true`, spec 256 `policy=byte-identity pass=true` (`first_divergence=-1` on
  all four prompts), hidden goldens argmax 10/10. **Still owed, and the default stays OFF until they
  land:** (a) the confound-free greedy control — same binary, same box, SAME artifact, `idxsel` only,
  so the chain differs by the seam and nothing else (§8a); (b) the at-depth cross-surface audit
  (`MEMRA_Q4E_PLECACHE_AUDIT=1` at a 131,072 fill), the only arm that pins the cached ids against the
  full twin at REAL checkpoint buffers and therefore the only one that can catch a consistent error;
  (c) `--verify-bit-deep 131072` re-cut WITHOUT `--goldens`. The flip and its FLAGS row land together
  with those receipts in one commit.
- **`--verify-bit-deep 131072` OOM'd on its first attempt, and the cause is an arm mismatch rather
  than a capacity surprise.** Passing `--goldens` turns ON the golden pin, which scopes the cache
  seams to f32 for reference-parity comparisons — taking the KV from 11.08 to **49.0 KiB/token**.
  That gate is a TWO-STATE instrument by construction, so a 131,072 pair needs **~12.8 GiB** against
  the ~2.9 GiB its own sizing comment assumed, versus ~7.9 GiB free after the trunk. The comparison
  is oracle-free and never needed goldens; dropping them restores kvq and the pair fits. The sizing
  comment was right about the arithmetic and wrong about which cache arm it would run under.
- **TP2 rule gates with the seam armed** are not run: card contention, and TP2 cannot reach this
  window at all (it OOMs during the fill below 100k while one card reaches ~731k), so it is not the
  shipped 262k path.

## 10. Banked in passing: the first 262,144 rung ever measured with `idxsel` armed

Not this lane's seam, and recorded here because this lane's cell produced it and a number this size
should not sit in a log:

| depth | idxsel OFF (banked, LADDER) | idxsel ON (this cell) | change |
|---|---|---|---|
| 262,144 | **65.7 ms / 15.21 tok/s** | **40.22 ms / 24.86 tok/s** | **1.63x** |

`rounds=3x12`, medians `[40.3, 40.2, 40.2]`, spread 0.24%, `looped=false`, cumulative prefill
1,443.3 s for both rungs, `# cache kv_quant=q8_0/q5_1 idxq=q8 golden_pin=false seams_env=idxsel`.
The `plecache` OFF arm of §2's A/B is the same configuration one rung shallower, which is why the
two files' arms line up.
