# mtp12: the fused-verify lever, priced (`vfuse`)

Lane `lane/q4e-fused-verify-20260901`. Box sbox-eval, artifact `~/data/q48fn-nvfp4`,
ship admission (`--mtp-dev1 --spec-pmin 0.3 --spec-adapt 1`, K=5 so the verify chunk is
t=6). Rig (sm_120a laptop 5090) carries the exactness arms only; every ms in this file
comes from the box under `flock -x /tmp/q48fn-measure.lock` held around the whole
invocation, load included.

## The question

At K=5 on the raw shape the round splits (mtp11 receipt
`spec/mtp11/ab-defer-k5-m11-raw.tsv`, host arm, x5 medians):

| part | ms/round |
|---|---|
| draft chain (5 steps) | 5.30 |
| **verify chunk (t=6)** | **31.98** |
| round total / 4.41 committed tokens | 8.45 ms/token = 118.29 tok/s |

Verify is 86% of the round. The fused (prefill-style) program already runs for
`t > k_cap` — the dispatch is one condition, `exact = base_pos > 0 && t > 1 && t <= v.k_cap`
in `qwen4exp_gpu.rs`. So: what does the t=6 chunk cost on the FUSED program, and is the
gap big enough to pay for the rewind machinery a shipping fused verify would need?

The lever was framed as "verify runs 6 exact per-row decode programs (6 x ~5.3 ms), fuse
it to 1-2 step costs (~10-16 ms) and the round becomes 15.6-21.3 ms for 4.41 tokens =
207-283 tok/s — the whole 200+ target."

## The structural cost model, before any kernel

That framing does not survive contact with the code, and the arithmetic can be closed
from receipts already banked. Two corrections, both load-bearing.

**Correction 1 — the verify chunk is NOT six decode programs.** It is already a
t-batched chunk program. `VERIFY_MT_DEFAULT` is ON: trunk dense mats run
`qmatvec_bf16w_mt_f32` (W read ONCE for all six columns), the hyper read gates run the
hc-diet MT stages (weight rows read once, tokens inside), and the MoE verify columns
merge into ONE grouped gufuse launch per projection per layer. What stays per-column is
narrow: the GDN scan step, the QSA indexer projection, and the PLE projections.

The "6 x 5.3 ms" reading came from `31.98 / 6 = 5.33` landing on the chain's 5.30 —
a coincidence of two unrelated numbers. 5.30 ms is the whole FIVE-STEP DRAFT chain
(one layer per step); a plain t=1 TRUNK step is 14.86 ms (PROFILE-5). Six exact decode
programs would be ~89 ms. The verify already delivers six rows for 31.98, i.e. **2.8x
cheaper per row than a t=1 step** — the t-parallel win is spent, not available.

**Correction 2 — the `exact` flag does not gate the expensive sections at all.** This is
the one that settles the lever, and it is a code fact, not an estimate. Every trunk dense
mat and the MoE column merge dispatch on **`t`**, not on `exact`:

```
// launch_qmatvec_* (4 sites) and the hc-diet MT stages:
if (2..=12).contains(&t) && verify_mt_on() { ...launch_qmatvec_bf16w_mt... }
// MoE routed union:
if t > 1 && verify_mt_on() && sel_gufuse_on() && hidden % 32 == 0 && ff % 4 == 0 { ...merged... }
```

`verify_mt_on()` is default ON and `2 <= 6 <= 12`, so at t=6 **both arms take the
weight-shared multi-token kernels for every dense mat and the merged MoE launch.** Turning
`exact` off does not move them; there is no per-row dense program at t=6 to fuse away.

Sectioning the chunk by what actually differs when `exact` flips:

| section | exact arm | vfuse arm | differs? |
|---|---|---|---|
| trunk dense mats (GDN/QSA/shared/router/lm_head) | `qmatvec_bf16w_mt`, W read once for all 6 columns | **the same kernel** (t-keyed dispatch) | **No** |
| MoE routed union | ONE merged gufuse launch over every column's slots | **the same launch** (t-keyed; the seam forces `grouped` so it cannot fall into the per-expert prefill executor at minutes/chunk) | **No** |
| QSA sdpa (`sdpa_naive_mask`) | same kernel, t rows | same kernel, t rows | **No** |
| PLE host n-gram hashing | keyed on `tokens`/`t` | identical (`plecache` is exact-independent) | **No** |
| remaining per-layer host twins | per CHUNK | per CHUNK | **No** — per forward, not per column |
| hyper read gate | hc-diet, incl. its MT stages | hc-diet is **skipped** (`hc_diet_on() && (t == 1 \|\| exact)`) and the t-generic fused chain runs instead | Yes — and this one is a *predicted loss*: the diet beat the fused chain 15.69 -> 15.32 ms at t=1 (`perf/ab-hcdiet-nvfp4.tsv`) |
| GDN scan | per-column `gdn_scan_step_at` + per-column state snapshot | chunk scan | Yes, small gain — **and this is the ONLY section the rewind machinery exists for** |
| QSA indexer proj | t x m=1 cuBLASLt | 1 x m=t | Yes, small gain |
| PLE projections | t x m=1 cuBLASLt | 1 x m=t | Yes, small gain |

So the verdict's shape is fixed before a single ms is measured, and it is not "the win is
too small to pay for the rewind work". It is:

**The premise is wrong. The exact verify arm is not a slow per-row path that a fast fused
path could replace — it IS the weight-shared path for everything expensive, and it runs
per-column only for the four sections that carry per-column STATE. Flipping to the fused
program keeps every expensive section byte-for-byte the same work, wins three small
sections, and gives up the hc-diet gate. Net expectation: ~1.0x, plausibly below it.**

The rewind problem — the reason this lever looked like a multi-week build (fused kernels
emitting per-column snapshots, or round-start snapshot plus fused replay of the accepted
prefix, or a hybrid) — guards the GDN scan, the *smallest* of the three sections that
would gain.

**On stale attribution, said explicitly.** PROFILE-7's mtp10 round-cost identity split the
36.5 ms t=6 verify as ~24.5 GPU + ~12 ms of per-layer host-twin bubbles (48 MoE router
dtoh + 12 QSA indexer masks per chunk). **That attribution is retired at this tip and is
NOT used here:** the devtwin stack (`routerdev`, `idxcache`, `idxsel`) went default ON on
2026-08-31 on its own receipts (PROFILE-9: spec 1.116-1.194x per shape with byte-identical
chains), which is precisely the removal of those router dtoh. Quoting a pre-flip host-twin
share as current would have inflated the untouchable floor with a cost that no longer
exists. The current composition comes from this lane's own prof split, measured on the
same binary as the timings.

**On stale attribution, said explicitly.** PROFILE-7's mtp10 round-cost identity split the
36.5 ms t=6 verify as ~24.5 GPU + ~12 ms of per-layer host-twin bubbles (48 MoE router
dtoh + 12 QSA indexer masks per chunk), which would have made the untouchable floor alone
larger than the target's whole round budget. **That attribution is retired at this tip and
is NOT used here:** the devtwin stack (`routerdev`, `idxcache`, `idxsel`) went default ON
on 2026-08-31 on its own receipts (PROFILE-9: spec 1.116-1.194x per shape with
byte-identical chains), which is precisely the removal of those router dtoh. Quoting a
pre-flip host-twin share as current would have inflated the untouchable floor with a cost
that no longer exists. The current composition comes from this lane's own prof split,
measured on the same binary as the timings, below.

## The probe

Rather than argue the arithmetic, price it. The `vfuse` seam routes a `1 < t <= k_cap`
chunk through the fused program with no new kernels; `--verify-cost-probe <reps>x<chunks>`
times exact vs fused at the verify-chunk shape.

Probe honesty, stated because a timing arm that looks like a correctness arm is how a
wrong number gets quoted later:

- The fused chunk scan materializes only the final GDN state, so the fused arm leaves no
  per-column stash and `verify_rewind` refuses by name. The probe's states are
  **throwaway**: real tokens at real decode positions, forwarded as t=6 chunks, timed.
  No chain identity is claimed on the fused arm and none is checked there.
- Both arms get the SAME 4t-byte argmax-sink readback, so the delta is the program and
  not a ~6 MB logits dtoh appearing on one side.
- Both arms force `grouped` (see correction 2).
- `base_pos == 0` stays fused on both arms — the gen-157 rule is untouched; the seam
  moves the same chunk shape, it does not widen it.
- Per-rep statistic is the median chunk within the rep; chunk 0 of each rep is a warmed
  throwaway (first chunk of a width allocates every workspace slot — the `scan_warm`
  lesson). Interleaved reps with the fleet escalation protocol (x3, receipted escalation
  to x5 on either rule).
- The t=1 plain step is measured in the SAME lock hold and the same residency, because
  the round arithmetic needs it and a step borrowed from another run is a cross-run perf
  claim.

## Rig exactness arms (PASS)

`mtp-vfuse` in the tiny fixture gate, receipt `tiny-fixture-gate-vfuse.tsv`
(whole gate PASS, failures=0):

| check | result |
|---|---|
| fused chunk rows vs the reference executor (modelplan policy: tolerance + argmax/row) | argmax **4/4**, worst abs 2.506e-4 rel 2.506e-4 (bound 0.01) |
| fused arm vs the exact arm, same fed tokens/positions | worst abs **2.636e-5** |
| argmax sink fed on the fused arm (4t-byte device argmax == host argmax of the rows) | fed |
| `verify_rewind` after a fused chunk | **refuses, and the message names vfuse** (failure path executed, not described) |

Acceptance policy here is deliberately NOT byte identity. Byte identity is the
instrument, not the product: the exact per-row arm STAYS as the byte-identity gate arm
(spec-vs-plain, verify-bit), and `vfuse` is a different program allowed to differ in
bits — 2.6e-5 is the accumulation-class difference you would predict from swapping a
W-once matvec for a GEMM — but not in meaning, which is what the argmax-per-row +
reference-tolerance gate pins.

The seam-table oracle picked the new name up with no edit (24 boolean seams of 26,
each verified to change its own state and no other).

## Box measurement

**Status: cell built and queued; never got the lock, and then the box went away.**

`--verify-cost-probe 3x12` on the raw shape at ship admission was parked on
`flock -x /tmp/q48fn-measure.lock` behind the 262k lane's cells
(`~/realgate/vfuse/QUEUE.log`; instrument sha `abac64e1e4311ac6`, src 67bc58064, ckpt
`~/data/q48fn-nvfp4`, marker string verified present in the installed binary per the
rebuild-attribution law). Timeline, UTC 2026-09-01:

| time | event |
|---|---|
| 03:05 | waiter parked on the lock (verified blocked: pid 445520 on `flock -x`) |
| 03:35 | instrument re-installed by atomic rename after the prof-pass fix |
| 02:26 -> 07:35+ | the 262k lane's `spec262kv1-thinkon` cell holds the lock continuously, card 0 at 100% util, 6 log lines (mid-rung), two more cells behind it |
| ~07:45 | **box 3.66.86.187 stops answering** — ssh, ICMP and tcp/22 all time out |
| 08:40 | still unreachable after ~50 min of retries |

The interleave point never arrived, so no timed row exists. Nothing was reordered and
nothing was killed: the sibling lane's cell was left alone throughout, which is also why
the waiter never won the lock.

**The box being unreachable is reported, not remediated.** It carries another lane's active
multi-hour quoted-number cell; instance-level action (reboot, stop, replace) is a fleet
decision for the owner or that lane, not something this lane takes on its own.

Cell discipline, for whoever reruns it: the capacity guard sits INSIDE the lock hold.
Checking the cards and then blocking on flock is a race — the sibling queue refills card 0
while this cell waits, and the load then OOMs a minute into an exclusive hold. Acquiring
the lock first and waiting for the cards second also covers the VRAM release lag
(nvidia-smi free is not driver free). Rerun verbatim:

```
~/q4e-vfuse-cell.sh vfuse-raw ~/realgate/dump/prompts.tsv --verify-cost-probe 3x12
~/q4e-vfuse-cell.sh vfuse-thinkon ~/realgate/shapes/thinkon-prompts.tsv --verify-cost-probe 3x12
~/q4e-vfuse-cell.sh vfuse-32k ~/realgate/shapes/thinkon-prompts.tsv --verify-cost-probe 3x12 --verify-cost-depth 32768
```

Cell discipline, for whoever reads the receipt later: the capacity guard sits INSIDE the
lock hold. Checking the cards and then blocking on flock is a race — the sibling queue
refills card 0 while this cell waits, and the load then OOMs a minute into an exclusive
hold. Acquiring the lock first and waiting for the cards second also covers the VRAM
release lag (nvidia-smi free is not driver free).

| row | value |
|---|---|
| exact t=6 chunk, ms | _pending_ |
| vfuse t=6 chunk, ms | _pending_ |
| speedup vfuse/exact | _pending_ |
| plain t=1 step, ms (same hold) | _pending_ |
| section split, both arms | _pending_ (`verify-cost-sections-k6-vfuse-raw.tsv`) |

## Verdict

**NO-GO on the structural result; the measured row is confirmatory, not decisive.**

The lever was sized against a premise that the dispatch does not support. Restating it as
the kill:

1. The t=6 verify chunk is **already** the weight-shared multi-token program for every
   expensive section. `verify_mt_on()` is default ON and the dense/MoE dispatches key on
   `t`, not `exact`, so **both arms run identical kernels for the trunk dense mats, the MoE
   routed union and sdpa.** There is no per-row dense program at t=6 to fuse away.
2. `exact` gates exactly four sections, and they are the small ones: the hyper read gate
   (where vfuse *loses* hc-diet), the GDN scan, the QSA indexer projection, and the PLE
   projections.
3. Therefore the rewind machinery — the expensive part of the proposal, in any of its three
   forms — would be built to protect the **GDN scan**, the smallest of the three sections
   that could gain.
4. The 200 tok/s target needs the round at ~22 ms (4.41 committed tokens x 5 ms). The
   sections vfuse cannot touch already exceed that budget on their own.

**Kill criteria, stated so a revival has a bar to clear.** Reopen this lever only if one of
these changes, and never on re-derivation:

- `verify_mt_on()` goes OFF or its `2..=12` engagement window stops covering the shipped
  K+1 — then the exact arm really would be a per-row path and the premise would hold.
- A *new mechanism* appears in the fusible four that is worth more than the hc-diet loss,
  measured at t=K+1 before any rewind work starts.
- The round's composition shifts so far that the GDN scan becomes a large share (it is the
  only rewind-gated section).

**Where the verify's remaining time actually is, for the next lane.** Not here. The two
levers already named by the corpus and still unbuilt are the MoE routed union (the largest
GPU section, and a *union-of-experts gather* would read each routed expert's bytes once per
chunk instead of once per routing token — MTP-SPEC "named, not built") and the TP2-route
verify. Neither is a fusion of the verify dispatch; both are inside the section vfuse
proved it cannot touch.

## Verdicts-ledger row (draft — for darklanes)

```
VERDICT:q4e-vfuse-dead | scope: qwen4_exp NVFP4 spec verify, K=5 (t=6), memra tip 2026-09-01 | fused-verify (`vfuse`) DEAD ON PREMISE, not on margin: trunk dense mats + MoE column merge dispatch on `t` (`(2..=12).contains(&t) && verify_mt_on()`), NOT on `exact`, so both arms already run the weight-shared multi-token kernels for everything expensive; `exact` gates only hc read gate (vfuse LOSES hc-diet), GDN scan, and m=1 indexer/PLE projections — so the rewind machinery would be built to protect the SMALLEST fusible section. The "6 exact per-row decode programs x 5.3 ms" framing read 31.98/6 as a per-row decode cost and coincided with the DRAFT chain's 5.30; a plain t=1 trunk step is 14.86, so verify already delivers 6 rows 2.8x cheaper per row than a step | keywords: fused verify, vfuse, spec verify, verify_mt, t-parallel, rewind stash, mtp12 | src: research/qwen4exp-bringup-20260829/spec/vfuse/VFUSE.md | since: 2026-09-01
```

```
TRAP:prof-take-disables | scope: memra qwen4exp_gpu::prof (and any take()-drains-and-disables accumulator) | `prof::enable()` already resets; `prof::take()` DRAINS AND DISABLES — calling take() to "clear" right after enable() silently switches the profiler off, yielding an empty section split and a divide-by-zero share column. Caught by reading the module before spending an exclusive measurement hold, not after | keywords: prof, profiler, section split, take, enable, empty receipt | src: research/qwen4exp-bringup-20260829/spec/vfuse/VFUSE.md, memra 67bc58064 | since: 2026-09-01
```

```
LAW:price-the-dispatch-first | scope: any "route shape X through the existing fast path" lever | before building the machinery a fused/alternate path would need, READ WHICH FLAG GATES EACH SECTION. A dispatch keyed on `t` rather than on the arm flag means both arms already run the same kernels and the lever has no surface; the mtp12 fused-verify lever died on exactly this, after being sized from a round-cost split whose per-row reading was a numerical coincidence | keywords: cost model, dispatch, fast path, premise, sizing | src: research/qwen4exp-bringup-20260829/spec/vfuse/VFUSE.md | since: 2026-09-01
```
