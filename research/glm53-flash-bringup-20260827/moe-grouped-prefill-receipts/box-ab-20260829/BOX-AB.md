# MEMRA_MOE_GROUPED_PREFILL: the box A/B (2026-08-29)

Owner-granted window on a rented 4x RTX PRO 6000 Blackwell Server 96 GB box (600 W verified on
all four cards). Engine `9c6bfe0a587b` (this lane's head), binary sha `958f9a3dd7...0288c60`,
artifact `glm53-nvfp4` byte-complete vs the HF publish (190,750,167,370 B). Protocol per
`../AB-PLAN.md`: interleaved x5 fresh-boot arms, one env flag apart, real prompts, greedy as the
instrument, vendor-default sampled twin, engagement receipts, `MEMRA_PREFIX_CACHE_MB=0` pinned,
`reasoning_effort` pinned `low` on every request. Full raw evidence in this directory
(`ab-run.log`, `rows-*.public.json`, `tie-*.out`, `prof-and-summary.txt`; the printed
`text_head` fragments were stripped before banking, output shas retained).

## Serving shape (and the deviation, stated loudly)

**PP3 across cards 0/1/2, full expert residency on every stage; card 3 untouched.**
`MEMRA_PP_STAGES=3 MEMRA_PP_SPLITS=15,30 MEMRA_PP_DEVICES=0,1,2 MEMRA_MOE_RESIDENT_GB=98
MEMRA_MOE_SLOTS=16 MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_PREFIX_CACHE_MB=0`, TF32 off.

The window brief named the 2-card residency config. It is NOT reachable on this branch with
this artifact, and the smoke chain proving that is banked (`serve-smoke*` on the box; summary
here):

- `MEMRA_ST_PINNED=1` (inherited from the ctxprobe recipe) builds host pinned tiers and
  `build_dev_exps` refuses tiered loads, so the first smokes silently ran the SLRU shape.
  Dropped.
- At `SPLITS=24`, stage0 = 21 expert layers (81,648 MiB slabs) + the F32-resident trunk
  (~15.6 GiB on dev0: the BF16 trunk loads as f32, the `MEMRA_PP_BF16` story) leaves 650 MiB
  free; every long-prompt session admission (~1 GiB dual-stage) OOMs. Verified insensitive to
  `MEMRA_CTX`, `MEMRA_MOE_SLOTS`, `MEMRA_MAX_SESSIONS` (byte-identical VRAM across those
  smokes).
- At `SPLITS=23`, stage1 = 22 expert layers = 98,663 MiB > 97,887 MiB capacity: load OOM, by
  arithmetic.
- The R2 residency cell's 2-card receipt rode the glm53-pp lane's fraction-based partial
  residency ("frac095", 99.2% resident); the merged branch's decision is all-or-nothing per
  device, so 2 cards cannot hold 171.2 GB of experts + a ~19 GB f32 trunk + sessions.

The residency-projection accounting also over-counts (it multiplies one MoE layer's expert
bytes by ALL stage layers including the 3 dense: `93.77 GB` projected vs `81.6 GiB` actual on a
21-MoE stage). `MEMRA_MOE_RESIDENT_GB=98` overrides the gate; the actual placement is what the
VRAM receipts show. Engine debt filed below.

Both arms share this shape; the A/B is arm-vs-arm on one config. The ctxprobe 2026-08-29
baselines (57.4/67.1/78.8 s TTFD, SLRU shape) are cross-config context only; this box's own
OFF arm is the honest baseline.

## Prompts

Real agentic content from the multiturn-pack-blessed owner transcript sources, minted to the
ctxprobe lengths with the model's own tokenizer (content tokens 4614/5535/6455 + a one-line
task tail; measured `prompt_tokens` 4626/5547/6467). Prompt text stays OFF the public repo;
`prompts.json` sha256 `de57a7a471f9b163...74b53e46`, on-box only. Warmup = one 427-token slice
per boot.

## The A/B table (interleaved x5, fresh boot per arm, medians [min..max])

| row | arm | TTFD s | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| A4630 greedy | OFF | 54.16 [54.03..54.80] | 85.4 [84.4..85.6] | 21.4 |
| A4630 greedy | **ON** | **7.51 [7.45..7.66]** | **616.1 [603.7..621.0]** | 21.3 |
| B5550 greedy | OFF | 65.53 [65.21..65.72] | 84.7 [84.4..85.1] | 22.1 |
| B5550 greedy | **ON** | **8.90 [8.90..9.12]** | **623.0 [607.9..623.5]** | 22.0 |
| C6470 greedy | OFF | 75.86 [75.81..76.14] | 85.2 [84.9..85.3] | 22.0 |
| C6470 greedy | **ON** | **10.25 [10.24..10.38]** | **630.9 [622.8..631.6]** | 21.9 |
| A4630 sampled (vendor default) | OFF | 54.58 [54.11..55.17] | 84.8 | 21.3 |
| A4630 sampled (vendor default) | **ON** | **7.24 [7.23..7.26]** | **639.0 [637.2..639.8]** | 21.1 |

**7.2x-7.4x TTFD, 85 -> 616-639 tok/s prefill, decode unchanged, spreads non-overlapping by
~45 s.** The 4096-chunk-cap crossing is inside every row (prompts 4626-6467 > 4096). The OFF
arm at 6.5k tokens rides 76 s against the platform's 90 s first-token deadline; the ON arm
rides 10.3 s. At the 90 s deadline the max servable cold prompt moves from ~7.6k to ~56k
tokens (arithmetic from the measured rates; any customer claim goes through the product-facts
pipeline, not this receipt).

## Engagement (the step37 trap, closed)

Every boot logs the announce in BOTH arms (`[moe-grouped-prefill] flag=on|off`); every ON boot
logs `execute layer=<il>` for **42/42 MoE layers**; every OFF boot logs **0** execute lines.
Boot identity: fresh PID + `readlink /proc/pid/exe` + binary sha printed per boot; both arms
are boot-deterministic (ONE 32-token greedy output sha per row per arm across all 5 boots).
The prof boot's `[moe-grouped-prefill-prof]` count is 42 x requests (chunks=1 per prompt on
this walk; the PP2 microchunk geometry is PP2-only).

## Sampled twin (serving law)

Vendor-default shape (NO sampling params) served fluently in both arms every boot; the ON
sampled row is the fastest row in the table. Spec-engagement receipt: NOT applicable on this
placement; the boot log's own policy line says `spec-admission=off` for pp-cross-device
placement (glm5 serves eager here), so the twin's receipt is the row + the policy line, stated
rather than substituted.

## The first-token argmax gate: 2 of 3 prompts MATCH, B5550 FLIPS, and the flip is a near-tie

Greedy first token, per interleaved pair (5 pairs each):

- A4630: `The` == `The`, 5/5 MATCH
- C6470: `The` == `The`, 5/5 MATCH
- B5550: OFF `Looking` vs ON `The`, 5/5 DIFFER, both boot-deterministic

Near-tie evidence (the server has no logprobs surface, so the distribution was probed with 8
vendor-default sampled max_tokens=1 draws per arm): OFF arm first tokens
`{'We': 1, 'Sum': 2, '': 4, 'The': 1}`, ON arm `{'The': 5, '': 3}`. The OFF arm itself draws
`The` at this position; the position is soft, the argmax moved within the top set. This is the
`MEMRA_BF16_MMV`-class shape (argmax movement at near-ties), not a derailment; the full
32-token greedy texts diverge between arms on every prompt, which is the expected consequence
of the non-bit-stable grouped GEMM class under greedy accumulation and is NOT part of the gate.

**Per the pre-registered decision rule this still blocks the default flip**: "argmax gate green
(or owner accepts the delta), else escalate to a logit-delta cell before any flip." The proper
logit-delta cell needs an engine-side logits dump (no logprobs API); filed as the follow-up
below.

## MEMRA_PRIME_PROF split (one ON pass, t=6467, not part of the x5)

Per MoE layer: `router=0.0ms gemm_gu=17.7-19.0ms down_scatter=10.0-10.4ms shared=1.9-4.1ms`,
i.e. ~30.5 ms/layer x 42 = **~1.28 s of the 10.25 s TTFD is now the grouped MoE**. The router
host readback (L4) reads ~0 at this shape. The remaining ~9 s is the trunk: the KDA sequential
scan (L3) and the f32 trunk GEMMs (L2), exactly the plan's post-L1 sequencing. The grouped
GEMM itself has obvious headroom (18 ms for ~3.6 TFLOP of gate/up at t=6467 is ~200 TFLOP/s
effective per layer against the step37 lane's 170-270 measured class; the down+permute+scatter
10.3 ms carries two extra passes) but that is L1 tuning, not this receipt.

## Verdict against the pre-registered flip condition

| condition | result |
|---|---|
| TTFD improves at every measured length, x5 non-overlapping | **PASS** (7.2-7.4x) |
| sampled vendor-default twin healthy | **PASS** |
| engagement receipt, announce in both arms | **PASS** (42/42 vs 0) |
| first-token argmax gate on real prompts | **1 of 3 prompts flips at a measured-soft position** |

**No default flip in this window.** The flag stays OFF; the flip needs either the owner
accepting the near-tie argmax movement (with the sampled-distribution evidence above) or the
engine-side logit-delta cell (first-position logit vectors, both arms, margin stated) coming
back green. Everything else in the condition is banked and green.

**ADDENDUM (same day, after this receipt was banked): the owner accepted the B5550 near-tie
movement verbatim ("accepted", 2026-08-29, the MEMRA_BF16_MMV acceptance class), on the 8-draw
census above. The flip to DEFAULT ON landed in the follow-up commit; `=0` is the rollback
seam. The logit-delta cell stays owed as follow-up, no longer a flip blocker.**

## Engine debt found by this window (named, not absorbed)

1. All-or-nothing per-device residency cannot place this model on 2x96GB; the glm53-pp lane's
   fractional residency did not merge. Follow-up lane if 2-card serving is wanted.
2. The residency projection over-counts stage expert bytes (counts dense layers).
3. No logprobs surface blocks API-side logit-delta cells.
4. `[loader-law]` still loads the BF16 trunk as f32 2D floats (the known MEMRA_PP_BF16/L2
   story); the f32 trunk is now the dominant prefill term post-L1.

## Box state at window close

All memra-server processes stopped (PID-verified; `pgrep` clean), VRAM 0 MiB on all four
cards, cards 2/3 were used only as PP stages 1/2 during the arms (card 3 never touched).
Artifacts kept on-box under `~/gpf-ab/` (prompts, pool, logs, rows). Repo checkout on the box
still at this lane's head.
