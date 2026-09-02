# mtp14: the sel matvecs' lane occupancy (`downsel`)

Lane `lane/q4e-downsel-20260901`, off `origin/main` 88c7caed0. Owner target: 200+ tok/s
single-request spec decode; current best **136.2 tok/s** raw shape, K=5
(`devtwin/ab-devtwin-k5-dt4-raw.tsv`: reps=5, median 7.341 ms/token, accept 0.861,
mean_accept_len 4.41, draft 5.31 ms, verify 26.63 ms → round 31.94 ms).

**No timing hardware.** the cloud fleet is closed account-wide and no provider was approved for this
lane, so the only GPU available was the rig (sm_120a laptop 5090), which is exactness-only
(LAW:rig-gpu-exactness-only). Every number that carries a claim in §1-2 is therefore taken
from receipts measured on a BOX and already banked in this repo; the rig rows in §4 are
DIRECTION for shape selection and are labelled as such throughout, never as the verdict.
The verdict cells are owed and scripted (§6).

## The lever, handed over by the `moeu` lane

MOEUNION.md (mtp13, merged) killed the routed-union gather and named what it had priced
underneath:

> KNEE:q4e-sel-slots-not-bytes — the sel section's cost scales with SLOT COUNT, not weight
> bytes ... `block_dim` is (32,1,1) and the DOWN projection has `in_f=640` so `pairs=20` —
> lanes 20-31 idle for the whole kernel ... Occupancy/reduce structure is the priced-next
> lever in this section; weight traffic is not.

So the section is per-slot-WORK bound, and an idle lane is wasted per-slot work.

## 1. Dispatch audit — the premise SURVIVES, and it is not arm-keyed

Per LAW:price-the-dispatch-first, read which kernel actually runs before building anything.
The `vfuse` lever died because its dispatch was keyed on `t` rather than on its arm flag, so
both arms already ran the same kernels. Here:

| flag | default | what it selects |
|---|---|---|
| `SEL_V3_DEFAULT` | **true** | `qmatvec_nvfp4_modelopt_sel_f32_v3` for every `in_f%32==0 && out_f%4==0` sel matvec |
| `SEL_GUFUSE_DEFAULT` | **true** | `qmatvec_nvfp4_modelopt_sel_gu_silu_f32` for gate+up |
| `VERIFY_MT_DEFAULT` | **true** | the merged verify chunk (`t>1`) — ONE gu launch + ONE down launch over all slots |

Both kernels serve the **verify chunk and the t=1 decode step through the same launcher**,
with `block_dim = (32,1,1)` — one warp per block — in both cases
(`launch_nvfp4_sel_matvec`, `launch_nvfp4_sel_gu_silu`). The defect is a property of the
kernel body, not of an arm, so a kernel fix lands on BOTH shapes and there is no
"the other arm already runs it" trap to fall into. That is the opposite of vfuse.

Down launch, read off the merged verify site (`qwen4exp_gpu.rs`, `t > 1 && verify_mt_on()`):
`launch_nvfp4_sel_matvec(.., s_total, ff /*in_f*/, hidden /*out_f*/, ff /*x_stride*/)`.

### Geometry, derived from the artifact rather than assumed

`ARCH.md` (real census, `Qwen/Qwen3.8-Flash-Next @ de4b8e4d43b9...`) and `SEMANTICS.md`
"MoE (L510-527)" agree: `experts.n [512, 1280, 2560]`, `experts.down_proj [512, 2560, 640]`,
48 MoE layers, intermediate 640, top-10, `hidden = 2560`. So:

| launch | in_f | out_f | `pairs = in_f/32` | 32-lane loop | grid.x (block 32) |
|---|---|---|---|---|---|
| gate+up | hidden **2560** | ff 640 | **80** | ceil(80/32) = 3 warp iterations for 2.5 iterations of work → **83.3%** lane occupancy (a 3-vs-2 tail) | 160 |
| down | ff **640** | hidden 2560 | **20** | lanes 20-31 hold NO pair for the whole kernel, each active lane runs exactly ONE iteration → **62.5%** | 640 |

Both then pay a full 5-step `shfl_down` tree over 4 accumulators.

## 2. The ceiling — box-measured absolutes, geometry-derived idle fractions

The share does not need re-measuring: the two launches are named rows in a banked nsys
kernel summary taken on the box (`spec/mtp10/nsys/win-spec_cuda_gpu_kern_sum.csv`, 2160
instances each = 48 MoE layers × 45 verify rounds, per-instance stddev 3.2 / 0.9 us so every
instance is the same shape):

| launch | median per launch | × 48 layers | share of the 31.94 ms round |
|---|---|---|---|
| `qmatvec_nvfp4_modelopt_sel_gu_silu_f32` | 111.008 us | **5.328 ms** | 16.68% |
| `qmatvec_nvfp4_modelopt_sel_f32_v3` | 64.576 us | **3.100 ms** | 9.70% |
| section | | **8.428 ms** | **26.39%** |

This replaces MOEUNION's scope-caveated 7.3 ms / 22.9% (mtp10's THINKON attribution against
a 36.5 ms verify chunk, applied to a 26.63 ms one) with a per-kernel figure whose transfer
is defensible: the sel KERNELS are untouched by the devtwin work that shortened the chunk,
so their GPU time is what it was. The two agree to about 15%, which is the honest size of
that uncertainty.

Cross-check on the split: this says down is 36.8% of the section; the `moeu` lane's rig probe
independently read 38.4% (153.1 / 398.3 us). Different card, different probe, same shape.

**Ceiling — recover exactly the idle-lane fraction of each kernel and nothing else:**

| | today | at 100% lane occupancy | saved |
|---|---|---|---|
| gate+up | 5.328 ms | 5.328 / 1.20 = 4.440 | 0.888 ms |
| down | 3.100 ms | 3.100 / 1.60 = 1.938 | 1.162 ms |
| **round** | 31.94 ms | **29.89 ms** | **2.050 ms = 6.42%** |
| **tok/s** | 136.2 | **145.5** | **+6.9%** |

**6.42% clears the ~5% kill bar, so the lane proceeded.** Two things about that number,
stated because both cut against it:

- It needs BOTH kernels. Down alone is 1.162 ms = 3.64% of the round (141.3 tok/s) and would
  have been a KILL on its own. The gate+up twin is not optional garnish here; it is what
  puts the lane over the bar.
- Deliberately EXCLUDED from the ceiling, as upside: the reduce tree shrinks (5 shfl steps
  over 4 accumulators per 4 rows → log2(g) steps), the per-warp prologue amortizes over more
  rows, and register pressure per lane drops. Also excluded: the same down kernel runs in
  PLAIN decode, where `moe.sel_grouped` is 2.6 ms and 6.9-8.6% of a deep decode token
  (PROFILE-12 §3), so the lever has a second independent surface that the round arithmetic
  above does not count.

The section is not bandwidth-bound, which is why occupancy is the right axis rather than
traffic: PROFILE-C0 measured `moe.sel_grouped` at 2.6 ms against a ~0.74 ms byte roofline
(a ~3.5x gap), PROFILE-2 put v2 at ~340-420 GB/s ("3-4x under the card"), and MOEUNION's
fixed-work arm converted a 6x distinct-byte cut into 1.101x.

## 3. The shape

`qmatvec_nvfp4_modelopt_sel_g_f32` and `qmatvec_nvfp4_modelopt_sel_gu_silu_g_f32` make the
pair loop a **sub-warp of `g` lanes**: the warp carries `32/g` groups, group `gi` owns `rows`
consecutive output rows at `o0 + gi*rows`, lane `s = lane & (g-1)` walks
`p = s, s+g, s+2g, ...`, and the reduce is log2(g) `shfl_down` steps INSIDE the group with
`s == 0` writing. Rows per warp is `(32/g) * rows`, and the launcher tiles `out_f` by it.

Two properties make this a generalization rather than a rewrite:

- **`(g=32, rows=4)` IS the shipped program** — same per-lane pair set (`p = lane; p += 32`),
  same 5-step tree, same write lane, same per-row expression tree. Byte-compared in the gate
  (§5). That is the rollback, and it is also the A/B control (§4).
- Every other `g` changes the ORDER the pairs are summed in (a lane chains several pairs into
  its accumulator; the tree is shallower), so it is an accumulation-class change of exactly
  the kind v3 was against v2 — gated on tolerance against the host decoder chain, never
  claimed bit-identical.

One implementation trap worth recording: groups inside ONE warp have different `o0`, so a
group whose rows fall past `out_f` must NOT return early — it runs the reduce with zeroed
accumulators. An early return would leave `__shfl_down_sync(0xffffffff, ...)` with an
incomplete mask. The launcher additionally refuses any geometry it cannot tile exactly and
falls back to the shipped kernel.

`block_dim` stays 32 (one warp per block). The kernels honour `blockDim.x >> 5`, but warp
packing was measured NEGATIVE on this exact launcher once already (plain decode 14.38 →
15.13 ms, flat on verify sel, mtp6) and adding it here would make the A/B carry two free
variables instead of one. It stays an owed knob, not a lane deliverable.

## 4. Shape selection — rig rows, DIRECTION ONLY, and they inverted the design

`sel_shape_probe` runs the SAME slots, the SAME distinct experts and the SAME synthetic banks
of the serving geometry through each candidate partition, so the only thing varying between
arms is the shape. No checkpoint, ~1.3 GiB, ~30 s — it interleaves between any other lane's
cells. Arms interleaved rep by rep, rep 0 a warmed throwaway, per-arm spread reported.

**Relationship to LAW:rig-gpu-exactness-only, stated exactly rather than waved at.** That law
exists because the 5090 laptop throttles, which invalidates absolutes and cross-run
comparisons. What follows is neither: it is the RATIO between arms measured in one process,
in one residency, with the arms interleaved, and the same ratio taken again in two more
independent passes. Per-arm spread on this card runs **13-23%** while the arm ratios
reproduce to under **0.5%** across passes — which is what interleaving buys, and why the
ratio is the statistic and the microsecond column is printed only so the ratio can be
checked. **No absolute here is quoted anywhere as a serving number, and nothing in §2's
ceiling arithmetic uses these rows.** They pick which shape the box cell runs first.

Instrument cross-check: this probe's `off` section at t=6/60 slots reads **397.6 us** where
the `moeu` lane's independent probe read **398.3 us**. Different lane, different probe, same
card and shape.

### The control stopped being a control, and that is the finding

`dn:32:4+gu:32:4` runs the shipped program through the NEW kernel — bit-identical output,
gated in §5. It went in as a noise floor. It is not one: across six passes it reads
**1.002-1.027× faster than `off`**, reproducibly. Same bits, different scheduling — the
source restructure (indexed arrays and unrolled row loops instead of four named
accumulators) gives nvcc a different order to work with. So `arm / off` mixes the source
rewrite with the shape and only `arm / control` isolates the shape. Both are reported.

### The ladder (`rig-selshape-t{6,1}-pass{1,2,3}.tsv`, release, reps=25, interleaved)

Ranges are min-max over three independent passes. `rpw` = rows per warp = `(32/g)*rows`.

**t = 6 (the K=5 verify chunk, 60 slots)**

| arm | gu shape / grid.x | down shape / grid.x | section vs `off` | section vs ctl | gu vs ctl | down vs ctl |
|---|---|---|---|---|---|---|
| `off` (shipped) | — / 160 | — / 640 | 1.000 | 0.980-0.998 | 0.977-1.000 | 0.982-0.996 |
| `dn:32:4+gu:32:4` **(control, bit-identical)** | g32r4/rpw4 / 160 | g32r4/rpw4 / 640 | 1.002-1.021 | 1.000 | 1.000 | 1.000 |
| **`auto`** (= `dn:4:4+gu:16:4`) | g16r4/rpw8 / 80 | g4r4/rpw32 / 80 | **1.174-1.205** | **1.172-1.181** | 1.086-1.090 | **1.334-1.360** |
| `dn:4:4+gu:off` | — / 160 | g4r4/rpw32 / 80 | 1.076-1.096 | 1.074 | 0.952-0.959 | 1.332-1.342 |
| `dn:off+gu:16:4` | g16r4/rpw8 / 80 | — / 640 | 1.040-1.060 | 1.037-1.040 | 1.070-1.080 | 0.976-0.995 |
| `dn:4:2+gu:16:2` | g16r2/rpw4 / 160 | g4r2/rpw16 / 160 | 0.991-1.018 | 0.987-0.998 | **0.861-0.871** | 1.265-1.301 |
| `dn:8:1+gu:16:2` | g16r2/rpw4 / 160 | g8r1/rpw4 / 640 | **0.799-0.822** | 0.796-0.806 | 0.856-0.868 | **0.713-0.726** |

**t = 1 (the plain decode step, 10 slots)** — same winner, same magnitude:

| arm | section vs `off` | section vs ctl | gu vs ctl | down vs ctl |
|---|---|---|---|---|
| `dn:32:4+gu:32:4` (control) | 1.008-1.027 | 1.000 | 1.000 | 1.000 |
| **`auto`** | **1.170-1.188** | **1.153-1.161** | 1.062-1.076 | **1.325-1.335** |
| `dn:4:4+gu:off` | 1.090-1.097 | 1.062-1.087 | 0.947-0.976 | 1.316-1.326 |
| `dn:off+gu:16:4` | 1.039-1.056 | 1.024-1.032 | 1.075-1.088 | 0.951-0.955 |
| `dn:8:1+gu:16:2` | 0.838-0.857 | 0.830-0.834 | 0.908-0.914 | 0.730-0.732 |

### What the ladder says, and how it inverted this lane's own design

**ROWS PER LANE IS WHAT PAYS, NOT LANE OCCUPANCY.** This lane's first AUTO rule derived
`rows` from `g` so that rows-per-warp stayed at the shipped 4 — reasoning that a fatter warp
costs warp count, which at t=1 is already thin. That rule is BACKWARDS, and the arms that
follow it are the losers in the table:

- `gu:16:2` reaches **100%** lane occupancy and measures **12-14% SLOWER** than the control.
- `dn:8:1` reaches 83.3% (up from 62.5%) and measures **27% SLOWER**.
- `dn:4:2` reaches 100% and gains only 1.27-1.30× on down where `dn:4:4` gains 1.33-1.36×.

The mechanism is in the kernel's own reason for existing. v3's body holds 4 output rows per
lane so that ONE pair's 8 activation `float4` loads are shared across 4 rows and 4
independent `uint4` code loads stay in flight. An arm that fills the lanes by SPENDING
rows-per-lane pays 4x the activation-load instructions per row-pair and drops the
memory-level parallelism from 4 to 1 — and that costs more than the idle lanes did. Filling
the lanes is only worth doing at `rows = 4`.

So AUTO is: **`rows = 4` always; `g` = the largest power of two dividing `pairs`.** At the
serving geometry that is down `g=4, rows=4` (rpw 32, grid 80) and gate+up `g=16, rows=4`
(rpw 8, grid 80) — 100% lane occupancy at 4 rows per lane in both. The feared t=1
parallelism loss did not appear on this card (800 warps per launch instead of 6,400/1,600),
but 82 SMs is not 188, and that is precisely what the box t=1 cell is owed for.

### Realization estimate — NOT a claim, and it lands under the bar

Composing §2's box absolutes with §4's rig ratios (`auto` vs `off`: gu 1.091-1.115, down
1.339-1.379 at t=6): saves ~1.335 ms = **4.18% of the round → ~142.1 tok/s (+4.4%)**.

That is BELOW the ~5% the ceiling cleared, and it is stated here rather than buried: the
ceiling is 6.42% because it assumes the idle lanes come back for free, and the ladder shows
they come back at a discount (the gate+up half realizes 1.09-1.11× of its 1.20× ceiling,
because at `g=16` the reduce is 4 steps instead of 5 and the prologue does more work). A
number composed from two cards is not a receipt either way, which is why the box A/B decides
and the default ships OFF until it runs.

## 5. Exactness — rig gate PASS, failures=0

`gate_nvfp4_sel_group` (arm 0a2 of `qwen4exp_gpu_gate`), receipt
`rig-gate-selgroup.tsv`, run under `flock -w 600 /tmp/memra-gpu.lock`:

```
nvfp4-sel-GROUP kernel oracle: 64 (geometry, shape) cells over REAL MoE geometry
(down 2560x640 pairs=20, gate_up 640x2560 pairs=80) + tiny, worst abs 4.730e-4 rel 7.498e-5
vs the host decoder chain; (g=32,rows=4) BIT-IDENTICAL to the shipped v3 and gufuse kernels
and every shape's fused arm BIT-IDENTICAL to its same-shape chain (39168 f32 byte-compared),
incl. the count-gated pack twin + the tok_map verify merge; NaN scales + non-pow2 macros +
duplicate slots; per-geometry class calibration [down_real ship_vs_host=3.672e-5
tol=1.469e-4; gateup_real ship_vs_host=6.735e-5 tol=2.694e-4; down_tiny ship_vs_host=
5.927e-7 tol=1.000e-5; gateup_tiny ship_vs_host=6.557e-7 tol=1.000e-5]
...
# verdict	failures=0
```

The arm runs at **real MoE geometry**, which is the point: the defect only exists at
`pairs = 20` and `pairs = 80`, and the shipped tiny arm's shapes have `pairs` 1 or 2 and
cannot reach either. Hostile inputs are the shipped arm's, verbatim: planted modelopt NaN
scale bytes (0x7F and 0xFF → 0.0), mixed pow2 / non-pow2 (the real mint's amax class)
macros, a duplicate expert in `sel`, both `x_stride` modes, the count-gated pack blob and
the slot→token verify merge. 14 down-family shapes × 4 geometries + 7 gu shapes × 2.

### The ARMED run, and exactly what it does and does not prove

`MEMRA_Q4E_SEAMS=selgroup ./qwen4exp_gpu_gate` also passes, `failures=0`, receipt
`rig-gate-selgroup-ARMED.tsv` — every other arm stays green with the seam armed, including
the reference-parity `dir-nvfp4-stacked` / `dir-nvfp4-perexpert` arms and
`mtp-spec-tiny`'s spec-vs-plain byte identity.

**That is a NO-REGRESSION receipt, not an in-model engagement receipt, and the difference
matters enough to state.** The tiny plan is `hidden_size 16, moe_intermediate_size 8`, so
every in-model sel matvec there has `in_f` of 8 or 16, fails the `in_f % 32 == 0` guard, and
`sel_group_resolve` returns `None` — the new kernel does not run in those arms at all. What
the armed run proves is that arming the seam is clean (it parses, it applies, it disturbs
nothing, and arm 0a2's 64 cells run with it armed). What proves the kernel itself is arm 0a2
at real geometry, which is why that arm exists and why no tiny fixture can replace it. The
first thing the box cells do is run the real artifact, and `q4e-downsel-cell.sh`'s cell B
is the first receipt where the kernel executes inside a real forward pass.

### A trap the arm found on its first run

**The shipped sel oracle's 1e-5 rel bound is a TINY-GEOMETRY bound, and the SHIPPED v3
kernel exceeds it at the real MoE widths.** The first run of this arm failed at
`down_real ... rel 2.655e-5` — on the arm's own reference comparison, with the bit-identity
control against v3 already PASSED. So it was not the reshape: v3 itself is 3.672e-5 off the
exact host chain at `in_f=640` (and 6.735e-5 at `in_f=2560`), because a length-`in_f` f32
reduction has an order-dependent error that grows with the sum, and 1e-5 was calibrated at
`in_f` 16-64.

Holding a reshaped twin to 1e-5 there would fail it for being a different but equally valid
summation order of a sum the shipped kernel cannot hold to 1e-5 either. The arm therefore
MEASURES the shipped kernel's own deviation per geometry and uses `max(4 × that, 1e-5)` as
the same-accumulation-class bound, checks every shape BOTH against the host chain and
against the shipped kernel's output, and prints the calibration in the receipt so the bound
is readable rather than asserted.

## 6. Owed cells — ready invocations, none run

**None of these ran.** No approved timing hardware existed for this lane. Each is a
whole-invocation `flock` hold with both arms inside one hold, interleaved ×3, per
LAW:interleaved-ab and LAW:ab-arm-identity.

| cell | script | what it decides |
|---|---|---|
| A. shape ladder, box card | `q4e-downsel-cell.sh` (`selshape` cells) | replaces §4's rig ratios with box ones at t=6 and t=1, and confirms `auto` is still the knee on 188 SMs (the halved/eighthed grid is the risk) |
| B. short-shape spec A/B | `q4e-downsel-cell.sh` (`spec-ab`) | the serving number: `selgroup` OFF vs ON on the real spec loop |
| C. t=1 decode A/B | `q4e-downsel-cell.sh` (`decode-ab`) | the plain-decode surface (`moe.sel_grouped` 2.6 ms, 6.9-8.6% of a deep token) |
| D. 262k rung | `q4e-downsel-cell.sh` (`rung262k`) | the product window — a fatter warp with fewer blocks is the shape most likely to behave differently under a long-context admission load |

Every receipt must name `kv_quant=` / `idxq=` / `seams_env=` / `ckpt=` and the corpus
commit; the scripts emit those lines. Cell A needs no checkpoint and ~1.3 GiB, so it can
interleave between another lane's cells in ~30 s; B/C/D need the pinned artifact.

**Default-flip rule, pre-registered so the flip is not argued after the fact.** `selgroup`
goes default ON only when ALL of:

1. Cell B shows the ON arm faster than OFF by more than **both** arms' spread, interleaved
   ×5, on one hold.
2. Cell C does not regress plain decode beyond its spread (the fatter warp's grid is 8x
   smaller; the mtp6 warp-packing revert is the precedent for this failing).
3. Cell D reaches the 262k rung at no worse tok/s than OFF.
4. The rig gate re-runs PASS on the tip that carries the flip.

If cell B lands the round gain under ~2%, the honest call is to keep the seam OFF and bank
the lane as a priced dead end rather than carry a live seam nobody flips — the section is
26.4% of the round and its ceiling is 6.4%, so there is no version of this lever that
reaches the owner's 200 target and it has to justify itself on its own size.

### 6a. The cells ran (box, 2026-09-01/02) — receipts in `box/`

| cell | result |
|---|---|
| A shape ladder (t=6, box card) | auto section 162.6 us vs shipped 194.4 us = **1.195x** (vs bit-identical control 1.102x); t=1 pass confirms the same knee |
| B spec A/B, serving caches, K=5 thinkon, 5x64 x3 holds | auto **90.07 / 90.60 / 90.08** vs off **87.38 / 87.14 / 87.47** tok/s = +3.1 / +4.0 / +3.0%; f32-pinned first run (kept in `box/spec-ab-f32pin/`) +2.6 / +3.1 / +2.9%; `first_divergence=-1` every arm |
| C t=1 decode A/B, 32k | 0.9999x, 1.0001x (5 reps, escalated) |
| D 262k rung | 1.0003x (5 reps, escalated) |

**Against the pre-registered rule:** (1) MISSED by a hair — gain 2.9-3.8% vs per-arm spreads
2.3-4.0% on every hold, though the sign never flipped across six holds; (2) PASS (flat);
(3) PASS (flat); (4) rig gate on the flip tip PASS. The "under ~2%" dead-end clause does not
bite. Owner decision 2026-09-02 on this record (PR #56): **default ON**, rollback `selgroup=0`. The
sampled twin remains owed before any SERVING claim cites the +3%.

Two cell-script defects only the live run could expose, both fixed at source: the flock
re-entry shell never saw `CARD_TOTAL_MIB` (#44), and cell B lacked `--goldens/--prompts` —
the binary's spec section is gated on `--goldens` (#49); with them, the f32 exactness pin
arms unless `kvq,idxq` are named in the seams (also #49).

## 7. Where the section's time still is, for whoever holds this next

- **Warp packing, still unpriced at the chunk shape.** `block_dim` is 32 and the revert that
  put it there was measured at t=1/verify-sel in the mtp6 era. The `_g` kernels already
  accept `blockDim.x >> 5`; with `auto`'s rpw 32 the down launch runs 80 blocks per slot
  instead of 640, which changes the block-slot arithmetic the revert was about.
- **The gate+up half realizes only ~1.09× of its 1.20× ceiling.** The 3-vs-2 tail costs less
  than the deeper reduce and heavier prologue that removing it needs. A gate+up shape that
  keeps `g=32`'s 5-step tree while filling the tail iteration is the unexplored corner.
- **`_gu_wpr` in `qmatvec.cu`** still carries the same diagnosis in prose for the dp4a family
  ("The reduce, not the load, is the cost"), still default OFF and unpriced.
- Not this section: the trunk dense matvecs are BIGGER than the MoE section in the same nsys
  summary — `qmatvec_bf16w_f32` 18.8% + `qmatvec_bf16w_mt_f32` 18.3% = **37%** of GPU kernel
  time against the sel section's 15.6%. If the 200 target is the goal, that is where the
  next lane should price a lever, not here.

## Corpus rows (for `agent-knowledge/gpu/`, darklanes side)

```
KNEE:q4e-sel-rows-per-lane-not-occupancy | scope: qwen4_exp NVFP4 MoE sel matvecs (modelopt_sel_f32_v3, modelopt_sel_gu_silu_f32) at verify-chunk and t=1 decode shapes, sm_120 | filling the idle lanes of a matvec's pair loop pays ONLY if rows-per-LANE is held: v3's body exists so ONE pair's 8 activation float4 loads are shared across 4 output rows with 4 independent uint4 code loads in flight, and an arm that reaches 100% lane occupancy by SPENDING rows-per-lane measures WORSE than the 62.5%-occupancy kernel it replaces. Measured, interleaved x3, against a BIT-IDENTICAL control arm: gate+up g=16 rows=2 hits 100% lanes and loses 12-14%; down g=8 rows=1 goes 62.5% -> 83.3% and loses 27%; down g=4 rows=4 (100% lanes AND 4 rows/lane) gains 1.33-1.36x. Rule: rows=4, then g = largest power of two dividing pairs; rows_per_warp grows and the warp count falls out of it, which is the thing to re-check on a wider card | keywords: sel matvec, lane occupancy, rows per lane, sub-warp, pair loop, ILP, activation reuse, shfl reduce, warp count | src: memra research/qwen4exp-bringup-20260829/spec/downsel/DOWNSEL.md | since: 2026-09-01
```

```
TRAP:tiny-geometry-tolerance-does-not-transfer | scope: any kernel oracle whose reference is a host reduction, when the arm is later run at the real width | a rel-tolerance bound calibrated on a tiny fixture is a bound on that fixture's REDUCTION LENGTH, not on the kernel. The qwen4_exp sel oracle holds v1/v2/v3 to 1e-5 rel at in_f 16-64; at the real MoE widths the SHIPPED v3 kernel is 3.672e-5 off the exact host chain at in_f=640 and 6.735e-5 at in_f=2560, so a real-geometry arm inherits a bound its own reference implementation fails. Symptom to recognise: the new arm's bit-identity control against the shipped kernel PASSES and its tolerance check against the host reference FAILS — that is the bound being wrong, not the kernel. Fix: MEASURE the shipped kernel's own deviation per geometry and bound the twin at a stated multiple of it (4x here), check the twin against BOTH the host reference and the shipped output, and print the calibration in the receipt | keywords: oracle, tolerance, rel bound, accumulation class, reduction length, real geometry, tiny fixture, calibration | src: memra research/qwen4exp-bringup-20260829/spec/downsel/DOWNSEL.md | since: 2026-09-01
```

```
LAW:degenerate-shape-is-the-control | scope: any kernel generalized by a shape/tile/partition parameter | make the new kernel reproduce the shipped program EXACTLY at one setting of its parameter, gate that setting BIT-IDENTICAL to the shipped kernel, and run it as an arm of every perf table. It buys three things at once: the rollback is the same binary, the oracle gets a byte-compare instead of only a tolerance, and the perf table gets an arm whose delta separates the SOURCE REWRITE from the SHAPE. The third is not hypothetical — this lane's degenerate arm measured 1.002-1.027x FASTER than the shipped kernel across six interleaved passes for identical bits (nvcc schedules the restructured source differently), so every shape ratio read against the shipped baseline instead of against the control would have been overstated by up to 2.7 points | keywords: control arm, degenerate shape, bit-identity, generalization, rollback, A/B, scheduling, attribution | src: memra research/qwen4exp-bringup-20260829/spec/downsel/DOWNSEL.md | since: 2026-09-01
```
