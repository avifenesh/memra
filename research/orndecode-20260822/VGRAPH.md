# De-CPU-binding the sampled spec round: verify-graph +19.7% and DEFAULT ON, sync-batching refuted (2026-08-23)

Lane goal (owner call, after the prod-speed incident): **de-CPU-bind the sampled spec
round.** The incident established that this model's decode tracks host single-core speed
(Zen3 host ~210 tok/s vs Zen5 326–354, same GPU/binary/artifact), so the question was
where the host time goes and whether it can be removed rather than out-run with hardware.

## Where the host time goes

`MEMRA_SPEC_PHASE=1`, 35B-A3B cached-long serve shape, Zen3-class dev host:

| phase | share of round |
|---|---|
| verify-ISSUE | 50–58% (76% under an nsys capture) |
| commit-host | 24–26% |
| draft | 19–25% |
| **verify-WAIT** | **0.0%** |

verify-wait at 0.0% is the whole diagnosis: the host never waits for the device, it
spends its own time issuing the trunk. Adding per-launch host cost (nsys) inflated
verify-issue to 74–78% and left wait at 0.0 — the signature of a launch-bound phase.

## What was wired

The slice-4c verify capture already existed inside `qwen35_verify_tparallel`, with its own
comment saying it had no caller on this route ("stream rides the qwen35moe burst, graphs
ride the dspark route"). This lane adds that caller: `MEMRA_SPEC_VERIFY_GRAPH=1` arms a
model-owned pool for the MTP spec round (the dspark serve arm's pattern — persistent
across generations, cache-independent bodies, per-round refreshed pointer tables), and the
commit picks the slab twin when the round's linear stash landed in the graph's slabs.

Two integration defects found and fixed by measurement, both worth keeping in mind for any
future caller of this pool:

1. `commit_verified_prefix: verify ckpt missing for linear layer` — with graphs armed the
   GDN column stash is written inside the graph as memcpy nodes into the ctx slabs, so the
   commit must take `dspark_commit_prefix_slab`. The pool states which arm the round
   produced via `round_slab`; trust that flag, not the env, so a round that fell back to
   the eager walk still commits through the cols arm.
2. A `slice_mut` panic in the sampled arm only: the pool was sized from the adaptive cap
   (`k_cap + 1`) while the sampled window is `t_v_s = k + 1`, so its stash was sliced past
   its rows. Sized from `k + 1` now, plus a per-round capacity guard that declines the pool
   instead of slicing.

## Measured (dev host, forced ON/OFF, same boot protocol)

**Exactness: PASS.** Greedy, seed-pinned, 160 tokens: ON and OFF hash identically
(`84936c00e3eb5a42`, reasoning + content both hashed). The captured trunk is the same
kernels in the same order, and the tokens say so.

**The mechanism does what it was wired to do** — verify-issue, same shape, ON vs OFF:

| arm | verify-issue | commit-host | verify-wait |
|---|---|---|---|
| OFF | 93.8 ms (76.0%) | 18.5 ms | 0.0 ms |
| ON | **15.6 ms (10.6%)** | **108.6 ms (73.7%)** | 0.0 ms |

83% of the verify launch cost is gone.

**End-to-end: consistently positive, modest, and it took three windows to say so honestly.**
Window 1 (ABBA ×2) read FLAT — but one of its OFF boots drifted 203→290 across its own four
reps, so that window was contaminated and its "flat" is not a result. Window 2 (ABBA ×2, a
steady window) read every ON rep above every OFF rep (OFF 158–171, ON 182–224). Window 3
(balanced 8 boots, both orders twice) is the one to quote, because it spans enough of the
session to show the host drifting 297→197 tok/s on its own — and inside that drift, ON beats
OFF in **all four adjacent pairs**:

| adjacent pair (boot order) | OFF | ON | Δ |
|---|---|---|---|
| off-1, on-1 | 297.8 | 306 | +3% |
| on-2, off-2 | 207 | 215 | +4% |
| off-3, on-3 | 196.7 | 210 | +7% |
| on-4, off-4 | 204 | 211 | +3% |

The engine's own per-round accounting agrees and is immune to the drift, being internal to
each boot: window 2's phase totals put ON at 7.8 ms/round vs OFF 10.5 (22-round bursts) and
9.8 vs 12.8 (11-round bursts).

So: **+3–7% end-to-end on a host-CPU-bound host, byte-identical, with −83% of verify issue.**
Not the ~20% a single good window suggested — that window is in here as a lesson about
quoting the friendliest one.

## Why the win is small, and what that implies

Most of the saving did not reach the wall, it **moved into commit-host** (18.5 → 108.6 ms). Two
contributions, and the greedy arm separates them: on the greedy identity arm — no sampled
accept walk — the round total *did* drop (130.8 → 97.2 ms over 14 rounds) and verify-WAIT
rose to 86%, i.e. the round became device-bound, which is the de-CPU-bind landing exactly
as intended. On the sampled arm the host-side accept walk (per-round `filter_stats` +
`softmax_gather_filtered` + their readbacks) plus the slab commit's per-layer pointer
bookkeeping is now the dominant host phase, and it is the first blocking op after an async
graph launch, so it also absorbs the device time the graph no longer hides.

So the de-CPU-bind splits into a host half and a device half, and the host half is the one
this lane owns:

1. **verify-graph (this lane, proven):** removes 83% of verify issue, byte-identical, and
   moves the round off the host — after it, verify-wait dominates (86% on the greedy arm).
2. **the sampled accept walk (attempted below, refuted as a HOST problem):** batching its
   readbacks changed nothing, because by then the phase is waiting on the device rather than
   spending host time. What remains there is device work: the filter/gather pair runs over
   the full 248,320-row vocab per used column, and the slab commit's copies are real device
   copies. That is a kernel-efficiency lane, not a sync-count one.

**Disposition: DEFAULT ON for the GDN+MoE family since 2026-08-23** — see the promotion gate below, which is the evidence that was missing when this paragraph first read "candidate default, held OFF".

What is settled and does not need re-litigating: exactness (identical hashes, run-spec ×8
both arms), the mechanism (−83% verify issue), and the sign (positive in 4/4 adjacent
pairs). A graph launch of this body is also not free — the dspark receipts put the
full-verify capture at ~2.9k nodes — which is consistent with a win that is real but small.

## The second half, attempted and refuted: batched q gather (`MEMRA_SPEC_QBATCH`)

The reading above said the sampled accept walk was the next lane, and the obvious first cut
was its sync count: for every drafted position the walk issued four tiny H2D copies, an
alloc, a gather and a **blocking readback** — a K=5 round paid five host/device round trips
to learn five floats. The gather kernel already takes `npair` rows, so batching needed no new
kernel: stage the k_round rows with async D2D copies, gather in one launch, read all k_round
floats back in one sync.

**It works and it is exactly free.** Identity is the strongest in this whole lane, because
this change lives ON the sampled path and that path is seed-pinned: a fixed-seed SAMPLED
completion hashed identically with the batch off, on, and composed with the verify graph
(`08941d5bb9762b21` all three). And the speed is flat in BOTH host regimes:

| regime | base (per-position) | batched | verdict |
|---|---|---|---|
| fast host window | 301 / 294 | 299 / 297 | flat |
| slow host window | 173.6 / 164 | 163.4 / 170.8 | sign flips — noise |

Reverted per the flags doctrine (a flat arm is not a flag, and the record is this row, not
dead code). The hypothesis was wrong, and being wrong is the useful part:

**Why it was flat, and what it says about the lane.** Those readbacks were not stalls. After
the verify graph lands, the round is no longer host-bound at all — the greedy arm shows it
directly (verify-wait 86%), and `commit-host` in the sampled arm is large because it is the
first blocking op after an ASYNC graph launch, so it *absorbs device time* rather than
spending host time. Removing syncs from a phase that is waiting for the GPU buys nothing.
So the de-CPU-bind objective for this route is **met by the verify graph**: with it armed the
host stops being the constraint, and the next gains are device-side kernel work (the sampled
filter/gather over the full 248k vocab, the MoE verify bodies) — a different program from
this lane, and one to price on its own.

The composed arm is also the strongest end-to-end cell measured here: qb+graph 323/324 vs
adjacent base 297 (**+9%**), with the capture cost visible and amortizing inside each boot
(rep0 302, then 320–328).

## Promotion gate: the current-generation host (the one that decides the default)

Every number above is from a Zen 3 host, so the flag was held OFF pending a balanced
interleave on the host class that actually serves. Rented one non-serving 9950X (0.66 s
reference loop — faster than the serving box's 0.80 s), shipped the tip binary and the same
artifact (sha `72ff9600…`), ran 4+4 boots in both orders. This rig is also a far better
instrument than the Zen 3 box: each arm's four boots agree to under 1%.

| arm | boot medians | per-round |
|---|---|---|
| OFF | 266.5 · 266.0 · 266.3 · 266.4 | 6.9 ms |
| ON | 319.0 · 319.0 · 318.8 · 319.5 | 5.7 ms |

**+19.7%, no overlap between the arms at all.** verify-issue 55–62 ms → 8–10 ms per burst.
The expectation going in was that a faster host would show a SMALLER win; the opposite is
true, and the reason is visible in the numbers: the ON arm lands at ~320 tok/s on BOTH host
generations while OFF tracks host speed. The graph moves the round off the host and onto the
device, so ON is a device-bound ceiling and OFF is whatever the host can issue. A faster host
does not make the host stop being the constraint — it just raises the floor it imposes.

Exactness held here too, and across machines: the fixed-seed sampled hash is
`08941d5bb9762b21` on both hosts, both arms.

**Default flipped ON for this family**, `MEMRA_SPEC_VERIFY_GRAPH=0` as the kill switch.
Verified as a default rather than assumed: with no env var set the pool logs ENGAGED and the
box reads 319/323; with `=0` the pool is absent and it reads 266 — twice each, alternating.

Scope is deliberately the measured family only (GatedDeltaNet + routed MoE, the engine-side
twin of `model_forces_spec_replay`). Qwen3.8-27B is GDN + DENSE mlp and would otherwise have
inherited this default with no interleave of its own; it opts in with `=1` when it has one.

## Side finding closed: the ornith run-gen argmax flip is the documented near-tie class

The `run-gen` argmax assert fails on this artifact with the flag OFF as well, so it never
belonged to this lane — but "both arms fail identically" is a defence of a change, not a clean
bill of health, so it got the calibrated instrument (`tools/argmax-margin-gate.sh`, window 12):

```
explained pos=2041 margin=0.0378 < delta=0.3479
SUMMARY flips=1 bad=0
PASS: every prefill/decode argmax flip is explained by a margin the config spread covers
```

One flip, zero bad. That is the coverage artifact the gate exists to disambiguate (the assert
inspects only the last position, so whether a prompt "passes" depends on whether its final
token happens to sit on a near-tie), not a cache/threading defect.

## Measurement trap worth naming: two connectors on one tunnel split the traffic

While this lane ran, the retired host still had a live tunnel connector on the SAME tunnel
id as the replacement. A Cloudflare tunnel load-balances across its registered connectors,
so public requests were landing on either host — which means any end-to-end number taken
through the public hostname in that period is a blend of a fast and a slow host, and reads
bimodal for no engine reason. Killing the stray connector made the public path read
1.68–1.80 s per 512-token completion (284–305 tok/s), tight. Retiring a host means stopping
its connector explicitly and checking with a pattern that cannot match its own shell
(`ps -eo args | grep -cE "cloudflar[e]d --config"`); `pgrep -cf "cloudflared --config"`
counts itself and reports a phantom connector, or hides a real one behind a count of 1.

## Per-class serve cells (owner-measured on the serving host, 2026-08-23)

Reported by the owner from the live box, single-stream: **digits 354, repetitive 330,
code 316, prose 269** tok/s. Protocol (N, prompt lengths, thermal regime) is not recorded
here because these are not this lane's measurements — they are quoted as reported.

Two things follow. First, the spread is what acceptance does to a speculative path: the
draft head predicts digit and repetitive continuations far better than prose, and decode
speed tracks that directly, so a single published number for this model is a statement
about a prompt CLASS whether or not it says so. Second, the published headline (290 tok/s,
"typical 512-token completion") sits inside this range and below its midpoint, so it stays
honest as a typical figure — the class spread is the reason not to raise it to the digits
cell, and the reason a reader measuring on their own prose workload can land below it.

## Correctness battery (dev host PRO 6000, on the affected artifact — `vgraph-gates.sh`)

| gate | flag OFF | flag ON |
|---|---|---|
| `kernel-check` | ALL GREEN (87 cells, 21 skipped) | same binary, same cells |
| `run-spec` K=1..8 self-consistency | PASS ×8 | **PASS ×8, acceptance identical at every K** (14/17, 16/30, 17/45, 17/60, 17/75, 17/90, 17/105, 17/120) |
| `run-gen` argmax | **FAILS** | FAILS identically (left 369, right 25) |
| serve identity (greedy, seed-pinned, 160 tok) | sha `84936c00e3eb5a42` | sha `84936c00e3eb5a42` |

The run-gen failure is **pre-existing on main tip for this artifact and unrelated to this
lane** — it reproduces byte-for-byte with the flag OFF, and the gate's own diagnosis puts it
in the documented drift class rather than calling it a defect:

```
[gate] prefill: l[369]=13.1226 l[25]=13.0970 | decode: l[369]=12.5685 l[25]=13.2562
[gate] top-2 margin: prefill 0.0256 decode 0.6877 | config spread at these ids 0.5540
       -> NEAR-TIE class (the spread covers the margin)
```

A prefill top-2 margin of 0.0256 against a 0.5540 config spread is the near-tie class the
message describes. It is recorded here rather than left in a log because "both arms fail
identically" is only a defence of THIS change, not a clean bill of health for the model's
argmax gate — that flip deserves its own look, on its own lane, with `tools/argmax-margin-gate.sh`.

Raw: `vgraph-ab.sh` (A/B + identity protocol), `vgraph-gates.sh` (battery), `phase-probe.sh`
(phase decomposition), `launch-census.sh` (the nsys attempt that inflated verify-issue and
so confirmed the launch-bound reading).
