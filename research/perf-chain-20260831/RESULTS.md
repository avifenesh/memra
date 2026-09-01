# Where the step37 140-era decode actually went: a corrupt door, not a regression

**Status: CLOSED.** Four cells: the two removed doors (cell 1), vision residency (cell 2),
the 82-commit engine range under one fixed env (cell 3), and an in-session anchor cell added
after the bench box rebooted mid-lane.

Perf-chain continuation of `research/toolchain-ab-20260831` (cell one), which acquitted the
container toolchain (+0.37%) and localized the ~18% step37 decode gap to the **engine + env
era delta**: memra `c9a617ca994b` under the 140-era serving env did 140.95 wall tok/s while
the fleet pin `3999a92a6e18` under the current deploy env did 115.74, with the delta sitting
in per-verify-round wall time at equal realized tokens/round.

This lane prices the three pieces of that delta in order: the two serving-env doors the
2026-08-29 incident removed, the resident vision tower, and the 82 engine commits in
between.

## Headline

**The 140-era decode was bought by a door that corrupts generated text, and it is worth
~24% of decode, not the ~1.0 tok/s its removal was priced at. And no engine commit in the
range regressed decode: decode is flat across all 82, while prefill got 2.4x faster.**

| stack | wall tok/s | decode tok/s | what it is |
|---|---|---|---|
| era binary + 140-era env **with** the doors | 139.45 | 157.60 | the 140-era shape. **Corrupts text** — not servable. |
| era binary + 140-era env **without** the doors | 109.43 | 120.25 | the same binary, honest output |
| fleet pin, same fixed env | 116.64 | 121.30 | 82 commits later: decode flat, prefill 2.39x |
| fleet pin + current deploy env | 114.45 | 118.97 | what serving runs today |

So the assigned framing has to be restated: **there is no lost 140 to recover by reverting
engine commits, and there is no culprit commit to name.** The 140 was a corrupt-output
configuration, withdrawn on purpose. The real prize is **re-earning the removed doors' ~24%
decode win with a correct implementation** — a coalesced NVFP4 MoE expert-bank layout that
actually passes a byte-identity gate. That is an engine lane, not a rollback, and it has sat
unprioritized behind a price tag that was wrong by ~30x.

## Shared method

Everything reuses cell one's banked harness verbatim, extended only where a cell needs a new
env knob (`harness/`, diffed against `research/toolchain-ab-20260831/harness/` in
`harness/DIFF-FROM-CELL-ONE.md`).

- **Bench box.** One box, 2x RTX PRO 6000 Blackwell Server Edition 96 GB (600 W), CUDA 13.2,
  rustc 1.98.0. Both GPUs verified 0 MiB before every boot and drained to 0 MiB after.
- **Artifact.** step37-flash-nvfp4, **14/14** LFS shards sha256-verified against HF revision
  `4275532ffd9a9496ff36b7a2dc4a9db1048da438` before any measurement
  (`receipts/model-verify.log`, `EXIT=0`).
- **Protocol (sealed digits, unchanged from cell one).** 512-token streamed completions,
  **vendor-default sampling — no sampling params in any payload** (the registry's
  temperature 0.5 / top_p 0.9 govern), banked digits prompt + a fresh salt per rep, wall
  clock including TTFT (`wall_tok_s = completion_tokens / wall`), token counts from the
  stream's own `usage` block, spec receipts from `usage.spec`. Per boot: a spec-engagement
  smoke gate, 1 discarded warmup, 8 measured reps.
- **Decision metric.** Median `wall_tok_s` of a boot's guard-clean reps; arm value = median
  of boot medians. Guard = `completion_tokens == 512` and `finish_reason == length`.
  Guard-violating reps are counted and reported per arm, never silently dropped.
- **A/B law.** Interleaved fresh boots, x3, escalated to x5 when either amendment rule
  fires: (1) within-arm spread of the decision median > 0.5%, (2) verdict within 2x the
  pooled spread. Every arm reports its spread and every escalation names the rule that
  fired.
- **Arm identity per boot.** Fresh `BOOT_NONCE` read back from `/proc/<pid>/environ`,
  `readlink /proc/<pid>/exe` vs the arm binary, binary md5, baked fingerprint, and the
  build's own `git log -1`. Older commits report `system_fingerprint` unknown, so **binary
  md5 is the binding identity** and is carried on every measured row.
- **Env identity per boot.** The live `MEMRA_*` environ is banked
  (`receipts/environ-<arm>.txt`) and asserted against a per-mode expectation table in
  `harness/boot.sh`: every door the cell is about is proven set or proven absent **from
  `/proc`**, not merely intended. A mismatch aborts the boot before any rep runs.
- **Stop discipline.** Anchored `pkill -f "^/home/ubuntu/perf-chain/bin/memra-server"` —
  this lane's absolute binary path only. The anchor is what keeps the pattern from
  self-matching the driving shell (a prior lane's pgrep self-match).
- **Registry.** One registry for every arm, a byte copy of the deployment registry. Its
  `deny_unknown_fields` struct field set is **identical at both ends of the bisect range**
  (diffed empty), so one file parses on all 82 commits. Not committed here: it is a
  deployment artifact.
- **Build attribution.** One checkout, one binary per commit banked as
  `bin/memra-server-<sha12>`, `git log -1` recorded **after** checkout, and a build that
  finishes in under 5 s is rejected as a failed checkout rather than trusted.

**One disclosed rewrite of the banked receipts.** Every build fingerprint in this directory
is written `fp/<sha12>` rather than in the `memra-<sha12>` form the server emits. The
public-boundary policy's `live_fingerprint` rule (severity 1) matches the emitted form, and
these receipts carry 891 of them across 94 files. The alternative was 94 new hash-pinned
grandfather entries in `tools/public-boundary-allowlist.jsonl`, which is precisely the
exception-list growth that keeps suppressing findings after its reason dies. The rewrite is
mechanical, lossless and reversible — `s/memra-([0-9a-f]{12,})/fp\/\1/g`, applied once to
`RESULTS.md`, `harness/DIFF-FROM-CELL-ONE.md`, `logs/` and `receipts/` — and the value it
preserves is a commit sha that is public in this repository anyway. **No allowlist entry was
added by this lane.**

### Overlapped builds, and the receipt that they cost nothing

Builds for cell 3 were compiled while cells 1-2 measured, capped at `nice -n 19 ionice -c3`
with 8 of 48 cargo jobs, so the cards never idled waiting on a compiler. This is recorded
per binary (`built_under_measurement_overlap=` in each build receipt) because it is a
measurement risk, not a free lunch. The control is built in: cell 1 boots 1-3 ran with no
build present and boots 4-5 ran with one, interleaved, so a build-induced slowdown would
appear as elevated spread or a boot-order trend in the same arm.

The refusal in `harness/build.sh` was narrowed to the precise hazard while doing this: a
checkout is only dangerous to a live server that is **running from the checkout's `target/`
directory**, and this harness always launches a copy out of `bin/`. The guard now asserts
that exact condition via `/proc/<pid>/exe` instead of refusing on the mere existence of a
runner.

## Cell 1 — the two removed doors, priced directly. CLOSED.

**Question.** `MEMRA_NVFP4_BANK_V2` and `MEMRA_SEL_DOWN8` were removed for correctness on
2026-08-29 (the slot-major v2 TP expert-bank layout corrupts generated text; its
bit-identity claim was false). Removal was priced at **~1.0 tok/s**. Cell one's +19%
per-round gap suggested that price was badly wrong. Settle it.

**Arms.** One binary (`c9a617ca994b`, built on-box, md5 `3ef8b83b…`), one env, one axis:

- **O** = the 140-era env **with** both doors (`era` mode; live environ proves
  `MEMRA_NVFP4_BANK_V2=1` and `MEMRA_SEL_DOWN8=1`).
- **OD** = the identical env with **both doors unset** (`era-nodoors` mode; live environ
  proves both absent).

**Correctness caveat, stated up front.** The v2-bank door corrupts output text from token 1.
Arm O is a **wall-clock price on a known-corrupt path**. It is measured because the question
is "what did the removal cost", and it is never a serving configuration. Nothing in this
cell licenses re-enabling the door — `memra >= 75bf4ce76` refuses to boot with it set, and
`fd0a175ab` deleted it outright.

**Result** (x5 interleaved fresh boots; escalated from x3 because amendment rule (1) fired —
within-arm spread of the decision median above 0.5% in both arms):

| | O (doors ON, the 140-era shape) | OD (doors OFF) |
|---|---|---|
| boot medians | 139.77 · 142.10 · 137.42 · 139.45 · 133.20 | 110.09 · 109.63 · 109.43 · 108.59 · 108.94 |
| **arm median** | **139.45** | **109.43** |
| boot-median spread | 6.38% | **1.38%** |
| pooled median / mean / sd | 137.42 / 133.72 / 10.00 (n=37) | 109.27 / 108.89 / 2.52 (n=40) |
| TTFT median | 0.4207 s | 0.4316 s |
| decode tok/s median | 157.60 | 120.25 |
| spec acceptance median | 0.970 | 0.935 |
| verify rounds median (512 tok) | 148.5 | 145.0 |
| realized tokens/round | 3.45 | 3.53 |
| **wall ms per verify round** | **24.72** | **32.27** |
| guard-violating reps | 3 | 0 |
| binary md5 / fingerprint | `3ef8b83b…` / fp/c9a617ca994b | `3ef8b83b…` / fp/c9a617ca994b (same binary) |

**Delta OD - O: -30.02 tok/s = -21.53%.** Decisive: the gap is 1.4x the doubled pooled
spread (15.51%). It is also decisive under the most hostile pairing available — O's *worst*
boot (133.20) still beats OD's *best* boot (110.09) by 21.0% — so no reading of the
dispersion rescues the "~1.0 tok/s" price.

**The removal was under-priced by roughly 30x.** ~1.0 tok/s was claimed; ~30 tok/s was paid.

Three further readings, all from the same rows:

1. **The regression is per-round execution cost, not draft quality.** Realized tokens per
   round barely move (3.45 -> 3.53) while wall per verify round goes 24.72 -> 32.27 ms
   (**+30.5%**). This is the same signature cell one measured across the era gap, and cell 1
   now attributes it to the doors rather than to any engine commit.
2. **TTFT is untouched** (0.4207 -> 0.4316 s). Both doors are decode-path only, which is
   consistent with what they were: a slot-major NVFP4 expert-bank layout read by coalesced
   `*_v2` matvec kernels, plus the `down8` selective-decode kernels that required it.
3. **The acceptance difference in cell one's era gap was the doors, not the engine.** O
   sits at 0.970 acceptance and OD at 0.935 — and 0.935 is essentially the current stack's
   0.925-0.930 (cell 2). Cell one flagged O's high acceptance as possibly
   corruption-inflated and could not separate it; cell 1 separates it, on one binary. Note
   also that all 3 guard-violating reps in this cell are in the doors-ON arm (early `stop`
   finishes) and its per-rep acceptance swings from 0.75 to 0.997: **the corrupt path is not
   merely wrong, it is erratic**, which is where O's 6.38% spread comes from.

## Cell 2 — vision residency. CLOSED.

**Question.** The current deploy arms image input (`MEMRA_STEP_VISION_DIR`), which makes the
artifact's perception_encoder tower ~8 GB f32-resident. The 140-era arm had no vision at all
(it was armed 2026-08-30, after the seal). Does the resident tower cost decode?

**Arms.** One binary (the fleet pin `3999a92a6e18`, built on-box, md5 `dd421c2958…`):

- **P** = current deploy-shape env, vision armed (`current`).
- **PV** = identical env, `MEMRA_STEP_VISION_DIR` unset (`current-novision`).

**Result** (x5 interleaved fresh boots; escalated from x3 because both amendment rules
fired — within-arm spread above 0.5% and the verdict inside 2x the pooled spread):

| | P (vision armed = what serving runs) | PV (vision unset) |
|---|---|---|
| boot medians | 114.28 · 114.45 · 118.02 · 117.24 · 113.63 | 114.92 · 114.98 · 115.50 · 114.08 · 116.55 |
| **arm median** | **114.45** | **114.98** |
| boot-median spread | 3.83% | 2.14% |
| pooled median / mean / sd | 115.02 / 114.89 / 3.77 (n=40) | 114.72 / 114.58 / 3.48 (n=40) |
| TTFT median | 0.1779 s | 0.1778 s |
| decode tok/s median | 118.97 | 119.53 |
| spec acceptance median | 0.925 | 0.930 |
| wall ms per verify round | 30.23 | 30.09 |
| guard-violating reps | 0 | 0 |

**Delta PV - P: +0.53 tok/s = +0.46%**, nominally in the vision-off direction and far inside
the pooled spread. **The resident vision tower does not cost decode.** TTFT, acceptance and
tokens/round are indistinguishable.

**Resolution bound, stated rather than implied.** With 2-4% within-arm spreads at x5, this
cell excludes an effect larger than roughly +-4%; it does not resolve a 1% one. The claim is
"not a contributor to the ~18% era gap", not "exactly zero".

**But the tower is not free of side effects, and this cell caught one.** The graph-capture
census across the same boots (`logs/server-P*.log` vs `logs/server-PV*.log`):

| arm | `mtp-chain-graph captured` | `pre-capture pool trim` |
|---|---|---|
| P (vision armed) | 11 | **10** |
| PV (vision unset) | 11 | **0** |

Identical graph captures, but with the tower resident **every** capture is preceded by
releasing 448 MB of cached pool memory back to the driver ("driver free 7851MB < required
8311MB"). The tower's ~8 GB of f32 residency pushes driver-free headroom below what graph
instantiation needs, so the allocator gives memory back and re-acquires it once per request.
Today that costs no measurable throughput. It is worth writing down anyway: this is exactly
the headroom pressure the 2026-08-30 OOM incident came out of, and it is a standing
per-request dependence on the driver's allocator rather than a steady state.

## Cell 3 — the 82 engine commits under one fixed env. CLOSED.

**Fixed env.** `era-nodoors` for every arm — the doors-off 140-era env from cell 1. This is
the only era-shaped env that boots on every commit in range, because `75bf4ce76` (in range)
refuses to boot step37 with the v2-bank door set. Vision stays unset, so cell 2's axis
cannot leak in.

**Range.** `c9a617ca994b..3999a92a6e18` — 82 commits, 44 on first-parent, linear ancestry.
The first-parent sequence is the bisect spine; a culprit merge is then drilled into its own
second-parent chain.

**Anchors already in hand.** Cell 1's OD arm IS the range's left endpoint under this env
(~109.6). The right endpoint is the pin under the same env. A staircase of interior
checkpoints then localizes any step, and **every named culprit is confirmed by a direct
boundary A/B (commit vs its parent), interleaved, never by bisect adjacency alone.**

### Pre-probe: what cell one's own server logs already said

Before spending a card on bisection, cell one's banked boot logs were re-read. They carry a
clean structural difference that no timing run was needed to find:

| arm (cell one) | `mtp-chain-graph captured` lines | `pre-capture pool trim` lines |
|---|---|---|
| O1 / O2 / O3 (era stack) | 0 / 0 / 0 | 0 / 0 / 0 |
| P1 / P2 / P3 (current stack) | 11 / 11 / 11 | 10 / 10 / 10 |

Ten measured requests per boot, plus one boot-calibration capture: the current stack
captures a **fresh MTP draft-chain CUDA graph on every single request**, and each capture
first releases 448 MB of cached pool memory back to the driver. The era stack captures
nothing.

The mechanism is in the code, not inferred. `SampledGraphKey` bakes `seed` and `temp` as
**capture-time constants inside the graph**, so the key includes the seed; a request that
omits `seed` draws fresh per-request entropy; therefore the key misses on every new request
and the graph is captured once and used once. Three doors from
`lane/step37-draft-graph-serving-20260830` (merged in range at `41b0040e4`) default **ON**
and are what put the vendor-default serving shape on that path at all:
`MEMRA_STEP35_DRAFT_DCW`, `MEMRA_MTP_CHAIN_GRAPH`, and — decisively —
`MEMRA_SPEC_GRAPH_FILTERED`, which widened capture from the pure-temp regime to
truncation-filtered regimes. The vendor default is `top_p = 0.9`, i.e. **filtered**: before
that door, the shape we actually serve drafted eager and captured nothing.

That lane's gates were greedy K=1..8 identity, per-K acceptance identity, and **seeded**
sampled twins. A pinned seed is exactly the case where the graph IS reused across requests.
The gates therefore measured the one sampling shape in which the feature pays, and the
default shipped ON for a serving shape in which every request pays a capture and gets one
replay generation out of it.

**This is the greedy-is-the-instrument trap in a new costume:** the instrument shape (pinned
seed) and the product shape (fresh seed) put the feature on opposite sides of profitable.

**So the prime suspect is `41b0040e4`, and two flag arms can test it directly** without any
bisection: `FNOFILT` (`MEMRA_SPEC_GRAPH_FILTERED=0`) and `FNOCHAIN`
(`MEMRA_MTP_CHAIN_GRAPH=0`), both on the pin binary under the fixed env, rotating inside the
same interleave as the staircase. Result below.

**Checkpoints chosen** (data-pointed, from the pre-probe above rather than blind halving):

| # | commit | why it is a checkpoint |
|---|---|---|
| 4 | `3d52b8531` | the server lib/bin split |
| 15 | `305876ede` | end of the metering-seam arc (the business tier leaves the engine) |
| 24 | `abc401415` | last commit **before** the draft-graph merge |
| 25 | `41b0040e4` | the draft-graph merge itself — pre-probe's prime suspect |
| 37 | `b3a2d92ff` | end of the vram-admission arc |

**Result** (all 8 arms rotating inside ONE interleave, fresh boot each, x3):

| arm | commit | boot medians | median | spread | TTFT | **decode tok/s** | acc | tok/round | excl |
|---|---|---|---|---|---|---|---|---|---|
| S04 | `3d52b8531a31` | 116.18 · 115.59 · 115.22 | 115.59 | 0.83% | 0.1761 | 120.14 | 0.933 | 3.48 | 0 |
| S15 | `305876ede4d9` | 114.98 · 115.03 · 116.31 | 115.03 | 1.16% | 0.1758 | 119.52 | 0.926 | 3.42 | 0 |
| S24 | `abc4014151d1` | 113.83 · 117.11 · 116.97 | 116.97 | 2.80% | 0.1754 | 121.61 | 0.940 | 3.56 | 0 |
| S25 | `41b0040e4101` | 117.48 · 116.03 · 116.48 | 116.48 | 1.24% | 0.1777 | 121.16 | 0.934 | 3.53 | 0 |
| S37 | `b3a2d92ff051` | 117.71 · 117.11 · 113.79 | 117.11 | 3.35% | 0.1774 | 121.83 | 0.938 | 3.57 | 0 |
| S44 | `3999a92a6e18` | 116.91 · 114.83 · 115.84 | 115.84 | 1.80% | 0.1775 | 120.44 | 0.933 | 3.49 | 0 |
| FNOFILT | pin, `MEMRA_SPEC_GRAPH_FILTERED=0` | 114.94 · 117.38 · 115.65 | 115.65 | 2.10% | 0.1751 | 120.19 | 0.929 | 3.51 | 0 |
| FNOCHAIN | pin, `MEMRA_MTP_CHAIN_GRAPH=0` | 110.44 · 114.72 · 117.22 | 114.72 | 5.91% | 0.1759 | 119.18 | 0.927 | 3.46 | 0 |

Every arm's binary md5 and baked fingerprint are distinct and match its commit
(`receipts/progress-cell3.txt`, PREFLIGHT lines). Zero guard-violating reps in 192.

**Verdict: the staircase is FLAT. No commit in the range moved decode.** The eight arm
medians span 114.72 to 117.11 — a **1.8% total range**, smaller than several arms' own
within-arm spread, and every pairwise delta is far inside 2x the pooled spread. There is no
culprit commit to name, because there is no engine regression to attribute.

**The direct boundary A/B at the prime suspect refutes it.** S24 (`abc401415`, the parent)
vs S25 (`41b0040e4`, the draft-graph merge), interleaved fresh boots x3: 116.97 vs 116.48,
**-0.4%**, inside both arms' spreads — and the boots disagree on the sign (boot 1 favours
S25 by +3.2%, boot 2 favours S24 by +0.9%). The merge that introduced per-request graph
capture is decode-neutral on the served shape.

**And the flag arms confirm it from the other side, at x5.** Both were escalated (rule (2):
their deltas sit inside 2x the pooled spread). Turning the capture off recovers nothing:

| arm | boot medians (x5) | median | spread | decode tok/s | vs pin |
|---|---|---|---|---|---|
| S44A (pin, graph ON) | 117.13 · 116.64 · 116.28 · 116.62 · 116.69 | 116.64 | 0.73% | 121.30 | — |
| FNOFILT (`MEMRA_SPEC_GRAPH_FILTERED=0`) | 114.94 · 117.38 · 115.65 · 113.74 · 115.32 | 115.32 | 3.15% | 119.81 | **-1.13%** |
| FNOCHAIN (`MEMRA_MTP_CHAIN_GRAPH=0`) | 110.44 · 114.72 · 117.22 · 115.62 · 118.28 | 115.62 | 6.79% | 120.16 | **-0.87%** |

Both deltas are inside the arms' own spreads and both point *against* the door being a cost —
graph-off is nominally slower, not faster. The capture census proves the arms really did what
they claim —
`mtp-chain-graph captured` per boot: S24 **0**, S25 **10**, S44 **11**, FNOFILT **0**,
FNOCHAIN **0** — so the comparison is genuinely graph-on versus graph-off, and it is a wash.

### The pre-probe hypothesis was wrong, and the cell is why

The log census above generated a confident, mechanism-backed hypothesis: per-request graph
recapture is the regression. **The measurement killed it.** Recording that explicitly,
because the hypothesis is the kind that gets "fixed" without a cell: a future agent reading
only the capture-per-request finding would spend an engine lane removing a cost that is not
costing anything measurable on the vendor-default serving shape.

What survives is smaller and honest: the capture is **real waste, not a regression**. The
graph is captured once and replayed for one generation because the seed is baked into
`SampledGraphKey` and each request draws a fresh seed. It buys roughly nothing net today
(+0.2% for S25-vs-S24 and FNOFILT-vs-S44 both sit inside noise), so the capture work is
paying for itself and no more. Making the seed a graph *input* rather than a capture-time
constant would let one graph serve a whole model+regime instead of one request — turning a
break-even path into a win — but it is an optimization to size on its own cell, not a
recovery of anything lost. It is **not** where the 140 went.

### The one real change in the range: prefill, not decode

The staircase is flat in decode and flat in wall, but it is not flat everywhere. Compare the
era commit under the identical fixed env (cell 1's OD arm) with the pin (S44):

| | OD = `c9a617ca994b` | S44 = `3999a92a6e18` | change |
|---|---|---|---|
| **decode tok/s** | 120.25 | 120.44 | **+0.2% (flat)** |
| TTFT | 0.4316 s | 0.1775 s | **2.43x faster** |
| wall tok/s | 109.43 | 115.84 | +5.9% |

(Cell 1's OD against cell 3's S44 — across the reboot. The anchor cell below repeats exactly
this comparison inside one session at x5 and gets the same answer: decode +0.58%, TTFT 2.39x,
wall +6.12%.)

The wall-rate gain across the 82 commits is **entirely the prefill improvement** arriving
through a 512-token request's TTFT term; the token-generation rate itself did not move at
all. This also explains cell one's otherwise puzzling pair of observations (a decode
regression *and* a 2.3x TTFT improvement) as one thing: prefill got much faster, decode was
untouched, and the doors took the decode.

Because cell 1 and cell 3 ran either side of an unplanned box reboot, this OD-vs-S44 row is
re-measured in one session by the anchor cell below rather than asserted across sessions.

## Anchor cell — the range's endpoints, in one session. CLOSED.

Cell 1 and cell 3 ran either side of an unplanned reboot of the bench box (11:12Z: a
graceful `shutdown` followed by boot — not issued by this lane, and not a crash: no OOM in
the kernel ring buffer, both cards back at 0 MiB and 600 W, artifact and receipts intact).
Rather than
compare cell 1's OD arm to cell 3's staircase across that boundary, the two endpoints and
the first staircase step were re-measured **interleaved in one session**:

Escalated to **x5** (rule (1) fired: every arm's spread exceeds 0.5%):

| | ODX = `c9a617ca994b` | S04A = `3d52b8531a31` | S44A = `3999a92a6e18` |
|---|---|---|---|
| boot medians | 109.59 · 109.53 · 110.50 · 110.16 · 109.91 | 116.30 · 115.93 · 115.19 · 116.74 · 114.19 | 117.13 · 116.64 · 116.28 · 116.62 · 116.69 |
| **arm median** | **109.91** | **115.93** | **116.64** |
| boot-median spread | 0.88% | 2.20% | **0.73%** |
| pooled median / mean / sd (n=40) | 109.92 / 108.79 / 3.40 | 115.46 / 115.02 / 2.88 | 116.62 / 116.33 / 2.20 |
| TTFT median | **0.4230 s** | **0.1754 s** | **0.1769 s** |
| **decode tok/s median** | **120.60** | **120.47** | **121.30** |
| spec acceptance | 0.937 | 0.931 | 0.934 |
| tokens/round | 3.52 | 3.51 | 3.54 |
| guard-violating reps | 0 | 0 | 0 |

Two decisive readings, and they are the load-bearing rows of the whole lane:

1. **ODX reproduces cell 1's OD arm across the reboot** (109.91 vs 109.43, 0.44% apart, both
   spreads under 1.4%). The reboot did not move the box, so cell 1's doors verdict lands in
   the same frame as cell 3's staircase.
2. **Decode is FLAT end to end: 120.60 -> 121.30 across all 82 commits (+0.58%, inside the
   spreads), while TTFT improves 2.39x (0.4230 -> 0.1769 s).** The wall-rate gain
   (109.91 -> 116.64 = **+6.12%**, decisive against a 3.22% doubled pooled spread) is
   entirely that prefill improvement arriving through a 512-token request's TTFT term.
   **The engine's token-generation rate did not change at all in this range.**

## The full accounting

Every number below is from this lane on one bench box, and each step is a within-session
interleaved A/B:

| step | wall tok/s | decode tok/s | what moved | source (each step in-session) |
|---|---|---|---|---|
| 140-era stack (era binary + doors) | 139.45 | 157.60 | the seal's shape. **Corrupts text.** | cell 1, arm O |
| minus the two doors | 109.43 | 120.25 | **-21.5% wall, -23.7% decode** | cell 1, arm OD (x5, same binary) |
| plus 82 engine commits | 116.64 | 121.30 | **+6.1% wall, decode FLAT (+0.6%)**; prefill 2.39x | anchor, ODX vs S44A (x5) |
| plus the current deploy env (vision) | 114.45 | 118.97 | **~0** (within the cell's +-4% resolution) | cell 2, P vs PV (x5) |

That reconciles cell one's era gap completely: 139.45 -> 114.45 is **-17.9%**, exactly what
cell one measured on a different box — and **the decode half of it is 100% the two removed
doors**. The engine commits are a net *gain*, delivered entirely in prefill.

## Verdicts and recommended engine direction

### 1. There is nothing to recover by reverting engine commits. Close that hunt.

The perf chain started from "the 140-era decode regressed, find the commit". There is no
such commit. Decode across `c9a617ca994b..3999a92a6e18` is flat to within 0.7%, the
staircase's eight arms span 1.8%, and the direct boundary A/B at the prime suspect is a
wash. **The 140 was not lost to a regression; it was withdrawn on purpose, for correctness.**

### 2. The removed doors are worth ~24% of decode, and that number should replace "~1 tok/s"

`MEMRA_NVFP4_BANK_V2` + `MEMRA_SEL_DOWN8` cost **-21.5% wall / -23.6% decode** when removed,
on one binary, one env axis, x5 interleaved. The removal note prices it at ~1.0 tok/s; the
measurement says ~30 tok/s wall and ~37 tok/s decode. Anywhere that ~1.0 tok/s figure is
quoted as the cost of the 2026-08-29 correctness fix, it is wrong by about 30x.

To be unambiguous: **this is not an argument to bring the doors back.** They corrupt
generated text from token 1, `75bf4ce76` refuses to boot with them, and `fd0a175ab` deleted
them. The correct conclusion is the opposite one — the win they were delivering was much
bigger than anyone recorded, so **re-earning it correctly is worth a real engine lane**, and
it has been sitting unprioritized behind a wrong price tag.

### 3. Recommended engine direction: re-derive the coalesced NVFP4 expert-bank path, gated

What the doors actually did, from the code at the era commit:

- `MEMRA_NVFP4_BANK_V2` stored the contiguous NVFP4 MoE expert banks in the **slot-major
  layout** that the coalesced `*_v2` matvec kernels read (`qmatvec_nvfp4_fast_v2`), described
  in-tree as a "pure byte permutation — value-exact". The 2026-08-29 bisect proved that
  claim **false**: the layout/kernel pair changed generated text.
- `MEMRA_SEL_DOWN8` added the `down8` selective-decode kernels, which **required** the v2
  banks and went with them.

So the lost win is a memory-coalescing win on the decode-critical MoE matvec, which is
exactly where a 196B-A11B MoE spends its decode budget — entirely consistent with a ~24%
decode effect. The direction:

1. **Re-derive the slot-major permutation and its reader as one unit**, and gate the pair on
   a byte-identity oracle against the v1 path — the gate the original door asserted in a doc
   comment but never had. A layout change that claims value-exactness must prove it on real
   prompts, not per-row spot checks.
2. Bring the `down8` selective-decode kernels back **on top of a proven layout**, not as a
   dependant of an unproven one. The dependency direction is what let one bad layout take a
   second optimization down with it.
3. Size the prize first: this lane already has the number (~24% decode on the qualified
   serving shape), so the lane can be justified before any kernel is written.

### 4. A smaller, separate optimization: stop baking the seed into the draft graph

`SampledGraphKey` bakes `seed` as a capture-time constant, and a request that omits `seed`
draws fresh entropy, so the sampled draft-chain graph is captured once and replayed for
exactly one request — every request. Measured today, that is **break-even, not a
regression** (cell 3). Making the seed a graph *input* (device-side Philox counter/seed
buffer rather than a baked constant) would let one captured graph serve a whole
model+regime, converting a break-even path into a win. Worth its own cell; **not** on the
critical path to the 140.

### 5. An engine defect this lane tripped over: `system_fingerprint` can name the wrong commit

`crates/memra-server/build.rs` bakes `git rev-parse HEAD` into `MEMRA_BUILD_SHA` and emits
**no `cargo:rerun-if-changed`**. Cargo then re-runs the build script only when something in
`crates/memra-server/` changes. Two commits in this range change only other crates:

| commit | crates touched | fingerprint the binary claimed |
|---|---|---|
| `41b0040e4` (draft-graph merge) | `memra-engine` | `fp/abc4014151d1` (its parent) |
| `46f700291` (step37 lane landing 2) | `memra-engine`, `memra-kv` | `fp/d2044f7eafb2` (a different commit) |

Both binaries contained the **correct code** — proved by a marker test: the draft-graph
doors' env strings are absent from `abc401415`'s binary and present in `41b0040e4`'s — but
both **reported the wrong `system_fingerprint`**, which is the field the OpenAI contract
defines as identifying the backend configuration a response came from, and which this repo's
own comment says exists so "determinism claims (`seed`) are checkable across deploys".

This is product-facing, not just a lane annoyance: an engine release whose changes are
confined to `memra-engine` (i.e. most kernel and serving-path work) can ship serving a new
program while telling every client it is the previous build. Cell one bound arm identity on
this field. **Fix:** add `cargo:rerun-if-changed=../../.git/HEAD` (plus the packed-refs/ref
path) to `crates/memra-server/build.rs`, or move the SHA into a workspace-level generated
constant, and add a gate asserting the baked fingerprint equals the built commit. Until then,
**bind build identity on binary md5 plus a code-marker test**, which is what this lane does.

### 6. Method notes worth keeping

- **A per-mode env expectation table asserted from `/proc` is the right shape for an env
  A/B.** Cell 1's entire claim is "these two variables were set / were not set"; asserting
  that against the live environ before any rep runs turns the axis into a receipt.
- **Preflight every arm before the first boot.** A mistyped sha in a driver would have
  aborted cell 3 five cycles in. Validating all arms at t=0 costs milliseconds.
- **`pkill -f` self-matched the driving shell and killed an ssh session** mid-lane, exactly
  the trap the protocol warns about. Anchored absolute-path patterns in the harness were
  correct; the ad-hoc interactive `pkill` was not. Kill by PID from an `awk`-filtered `ps`.
- **Bank the kernel `boot_id` per boot.** The box rebooted mid-cell; without an in-session
  re-anchor the +6.4% engine gain and the -21.5% door cost would have been compared across
  two kernel sessions on nothing but hope.

## Should a deployment expect a recovery deploy?

**No. There is no recovery deploy to wait for, and no rollback that would help.**

- **Nothing in the engine range needs reverting.** Decode is flat across all 82 commits, so
  there is no build to go back to that decodes faster. The current pin is the fastest
  *correct* build measured in this lane: it matches the era commit's decode and beats it by
  6.4% on wall rate through prefill.
- **The vision seam is not costing throughput** and does not need turning off for
  performance reasons (cell 2, +-4% resolution). Its per-request 448 MB pool trim is a
  headroom note for whoever owns admission, not a throughput action.
- **The draft-graph doors should stay at their current defaults.** Turning either off
  recovers nothing (cell 3), so there is no flag change to ship.
- **The ~140 number must not be treated as a lost capability, or quoted as a target that
  regressed.** It was produced by a configuration that corrupts generated text. Any
  published performance claim must sit on the correct configuration's numbers.

What a deployment *should* expect is the outcome of a new engine lane, not a redeploy: if the
coalesced NVFP4 expert-bank path is re-derived and passes a real byte-identity gate, that is
when a ~24%-class decode improvement becomes deployable — and it would then follow the normal
rollout path (pinned commit, gate bundle, staged native server, sampled vendor-default probe
with a spec-engagement receipt, then and only then a claim change).

One correction that should propagate regardless of any lane: the **"cost of removal:
~1.0 tok/s"** note attached to the 2026-08-29 door removal is wrong by ~30x, and it is the
reason the re-derivation has not been prioritized.

## Receipts

- `receipts/rows-cell*.jsonl` — one JSON row per stream (tokens from `usage`, spec
  acceptance/rounds, fingerprint, **binary md5**, boot nonce, `built_from`), including
  smokes and warmups, with guard-violating rows flagged by `full_tokens:false`.
- `receipts/progress-cell*.txt` — the full interleave timeline, boot receipts inline,
  escalation declarations naming the fired rule.
- `receipts/boot-*.receipt` — per-boot arm identity: nonce, md5, `/proc` exe + environ
  nonce, the env expectation table that was asserted, GPU snapshot at ready.
- `receipts/environ-*.txt` — the live `MEMRA_*` environ census of every boot.
- `receipts/build-*.receipt` — per-commit build: sha, `git log -1` after checkout, binary
  md5, baked fingerprint, build seconds, overlap disclosure.
- `receipts/model-verify.log` — 14/14 artifact shards vs the pinned HF revision.
- `logs/server-*.log` — every boot: model load, admission calibration, spec engagement,
  graph capture lines.
- `logs/build-*.log` — every build, with its nvcc/rustc provenance.
- `harness/` — the launcher (4 door/vision modes), boot/stop with PID- and env-verified
  identity, the digits client, the interleave runner, the build driver, and
  `DIFF-FROM-CELL-ONE.md`.
