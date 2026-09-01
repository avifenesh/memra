# GLM-5.3-Flash — full expert residency across both cards, MEASURED (2026-08-28, lane/glm53-pp)

ROADMAP step 3, run on the real artifact. Two-card box (2x RTX PRO 6000 Blackwell Server
96 GB), exclusive, freshly booted, `NVIDIA_TF32_OVERRIDE=0`. Artifact `~/models/glm53-nvfp4`
(190.7 GB NVFP4). Binary: this lane's branch (the mHC ppN walks). Instruments: the banked
`probe.py` / `steady.py` from the decode-attribution lane, unmodified except for their port and
their own directory. reasoning_effort PINNED `low` on every row.

## Headline

**Full expert residency across both cards WORKS, on the f32 trunk, without `MEMRA_BF16_MMV`.**
p5 greedy goes **20.41 -> 29.95 tok/s (48.99 -> 33.39 ms/token), a 1.47x step**, and 3.51x
against the 8.5 tok/s configuration that was actually being served.

**Staging goes to zero, and the arms' own intercept is what the machine does.** The attribution
solved a staging-free constant X = 33.05 ms/token for the f32 trunk. With the experts resident
this cell measures **33.39 ms/token**. The gap is +0.34 ms, or 1.0%.

## The measured table

p5 = the real 785-char agentic prompt, 192 completion tokens, 4 reps after a prime, medians.
greedy = the INSTRUMENT; vendor-default sampled (temp 1.0 / top_p 0.95, seeded) = the PRODUCT.

| arm | p5 greedy | p5 sampled | p7 greedy | VRAM card0 / card1 |
|---|---|---|---|---|
| **R0** 1-card, `ST_PINNED`, 12000 slots | 20.414 tok/s · 48.99 ms | 24.355 · 41.06 | 18.922 · 52.85 | 80,211 / 3 MiB |
| **R2** 2-card PP, full residency | **29.947 · 33.39** | **27.081 · 36.93** | **29.744 · 33.62** | 93,685 / 94,039 MiB |
| **R2R** repeat boot | 29.859 · 33.49 | 26.941 · 37.12 | — | 93,589 / 94,007 MiB |

Boot-level reproducibility (the interleave unit is a BOOT — these arms are process-level env
behind a `OnceLock`, exactly the deviation METHOD.txt already documents): R2 vs R2R is 0.29% on
greedy and 0.52% on sampled. Within-arm spread across 4 reps is under 0.6%.

R0 reproduces the banked A2 arm on different hardware and a different binary: 48.99 ms here vs
49.14 ms banked (0.3%), 796.0 MiB/token staged vs 797.5 banked, and **accesses/token = 1008.0
exactly** — 42 layers x 8 routed x 3 projections, the same invariant.

## The decomposition, comparable to 84.1 / 15.9 / 17.1

Staging is converted at the banked pinned-transport slope, 49.55 MiB/ms (~53 GB/s, PCIe 5.0 x16
line rate). X = total − staging.

| cell | ms/token | staged MiB/tok | staging ms | X = ms − staging |
|---|---|---|---|---|
| R0 p5 greedy | 48.99 | 796.0 | 16.06 | **32.93** |
| R0 p7 greedy | 52.85 | 979.1 | 19.76 | **33.09** |
| R0 p5 sampled | 41.06 | 219.2 | 4.42 | **36.64** |
| R2 p5 greedy | 33.39 | (0, see below) | ~0 | 33.39 |
| R2 p7 greedy | 33.62 | (0) | ~0 | 33.62 |
| R2 p5 sampled | 36.93 | (0) | ~0 | 36.93 |

X is invariant on the single-card arm across two prompts (32.93 / 33.09) and reproduces the
banked 33.05. The resident arm lands **+0.46 ms (p5 greedy), +0.53 ms (p7), +0.29 ms (sampled)**
above the matching X. That residual is itself explained: card 1 holds 80,962 MiB of expert
blocks against the 81,648 MiB its 21 MoE layers need, i.e. **99.2% resident**, so ~0.8% of
accesses still stage. 0.8% of 796 MiB/token is 6.4 MiB = 0.13 ms, same order as the residual.

So the token now splits: **~0 ms staging + 15.9 ms roofline + ~17.2 ms launch = 33.4 ms.**
Launch structure is now 51% of the token and is the whole remaining story below the roofline.

## Against the projection — and a correction to it

ROADMAP step 3 reads `0 staging / 10.1 roofline / 16.7 launch = 26.7 ms = 37.4 tok/s`. That row
carries the **BF16 roofline**: it stacks step 2 (`MEMRA_BF16_MMV=1`) underneath it, and 26.70 ms
is the A3/A4 (BF16) intercept. The owner has not decided the BF16 trunk and this cell did not
enable it, so the prediction that applies here is the **f32** intercept:

  predicted (f32 trunk, full residency): 33.05 ms = **30.26 tok/s**
  measured                             : 33.39 ms = **29.95 tok/s**   -> −1.0%

The projection was right; it was being quoted against the wrong trunk arm. Step 3 on the f32
trunk is 30.3 tok/s, not 37.4. The 37.4 figure remains available only if the owner ratifies
`MEMRA_BF16_MMV` (a numeric-class door — its output sha differs and it needs its own acceptance).

## The roadmap's fit arithmetic was wrong, and it was wrong in our favour

ROADMAP: "an even 21/21 layer split puts ~85.6 GB of experts + ~7 GB of trunk on each card —
inside 96 GB, but only with the BF16 trunk." That sentence sizes the trunk at ~7 GB per card by
taking the **BF16** trunk (13.9 GB) and halving it, while treating the f32 trunk as though its
whole 23.6 GB landed on **each** card. Under PP the trunk shards by layer like everything else.

Measured on this box, f32 trunk, `MEMRA_PP_SPLITS=24`:

| | card 0 | card 1 |
|---|---|---|
| trunk (at load, before any expert staged) | 11,987 MiB | 13,013 MiB |
| expert blocks at plateau | 81,698 MiB | 81,026 MiB |
| total | 93,685 MiB | 94,039 MiB |
| card capacity | 97,887 MiB | 97,887 MiB |
| headroom | 4,202 MiB | 3,848 MiB |

21 MoE layers x 288 experts x 3 projections = 18,144 blocks x 4.5 MiB = 81,648 MiB per card.
**It fits on the f32 trunk with ~3.8-4.2 GiB to spare.** No BF16 needed for step 3.

Census cross-check, from the artifact rather than quoted: the repack slab is 171,228,278,784 B
= 159.47 GiB over 42 trunk MoE layers, and summing the safetensors shard headers gives 163.27
GiB of expert tensors over 43 layers (42 trunk + the unused MTP) — the same number, 3.797 GiB
per MoE layer, two independent ways.

## What had to be raised, and the OOM gate for it

`MEMRA_MOE_HARD_VRAM_FRAC` defaults to **0.80**, and it — not the slot count — was the binding
clamp: the first residency boot (`MEMRA_MOE_SLOTS=18144`, default frac) plateaued at 81,045 /
81,367 MiB, about 84% of the blocks, and p5 sat at 29.5 tok/s with fresh prompts still staging.
At **0.95** (the parser's ceiling) the clamp stops binding and the slot count governs.

FLAGS.md requires an OOM-gated sweep before raising it, and this is that receipt: 0.95 booted,
loaded, warmed over the whole prompt pool three times, ran three steady batteries and a repeat
boot, peaking at 94,039 of 97,887 MiB (96.1%) with no OOM, no eviction churn between passes, and
identical output shas. **It is a machine-specific pin, not a new default** — headroom is under
4 GiB at `MEMRA_CTX=8192` / `MEMRA_MAX_SESSIONS=4`, so a longer context or more sessions needs
its own sweep.

## Byte identity across the placement, on the real artifact

Every output sha is IDENTICAL between the 1-card and the 2-card arm:

| row | 1-card sha | 2-card sha |
|---|---|---|
| p5 greedy | `4ec98d8aeb7a30e6` | `4ec98d8aeb7a30e6` |
| p5 sampled (seeded) | `bec0d19fbf181d1b` | `bec0d19fbf181d1b` |
| p7 greedy | `fdf5109149b4ece8` | `fdf5109149b4ece8` |

This is much stronger than the fixture gate: same bytes across two devices, a sharded loader,
peer transport and per-stage caches, on a 190.7 GB NVFP4 model, greedy and sampled, two prompts.

These shas differ from the attribution's banked `fd006d0d50eb59b5`. The control isolates why:
the 1-card arm on THIS binary produces `4ec98d8aeb7a30e6` too, so the difference is the build,
not the placement. Running the control is what makes that statement rather than a guess.

## A measurement gap this cell had to work around, and it should be fixed

**The MoE cache counters are not wired through PP stage engines.** `Engine::moe_cache_stats()`
reads `self.moe_cache` on ONE engine, and the server holds only the primary; under cross-device
PP the caches live on the per-stage engines. The consequence is nasty: `steady.py` reports
`MB_per_tok = 0.0`, `miss_per_tok = 0.0`, `acc_per_tok = 0.0` on every two-card row — which
reads exactly like "staging went to zero" and is actually "staging was not measured". A cell
that trusted those zeros would have claimed the headline for free.

So the residency claim here rests on evidence that does not come from those counters:
- VRAM occupancy per card against the block arithmetic (18,144 x 4.5 MiB), measured at plateau
  and stable across two boots and three warm passes;
- `disk_mib = 0.0` on every resident rep;
- the X-solve: measured total equals the single-card arm's own staging-free intercept, which is
  a counter-independent way of saying staging is gone.

Fix (named, not built here): aggregate `moe_cache_stats` across `PpNRt`'s stage engines, or emit
one `[moe-cache] snapshot` line per stage. Until then, no two-card staging number should be
quoted from that line.

## Quality flag, not a perf row

The **sampled** p5 row degenerates into a repeat loop in BOTH arms ("User references User
referencesUser references...") with the same sha. Per LAW greedy-is-the-instrument a looped
generation is flagged and never aggregated; it is recorded here because it is identical across
placements, so it is a model/prompt property and not a placement artifact. The attribution
flagged the same behaviour on prompt idx 2 under sampling. Worth a look from the
forward-numerics lane: a loop is expected under greedy, less so at temp 1.0 / top_p 0.95.

Also worth stating so it is not mistaken for a finding: the first warm passes show a wide
per-prompt spread (p4 9.9 vs p5 30.0 tok/s) at `max_tokens=64` with no prime. That is PREFILL,
not decode. Once `steady.py` primes the prefix (TTFT 0.0035 s) decode is prompt-independent:
p5 29.95 and p7 29.74.

## Scope

- One box, one artifact revision, `MEMRA_CTX=8192`, `MEMRA_MAX_SESSIONS=4`, single-session
  decode. No concurrency sweep, no long-context sweep, no multi-turn cache twin.
- `MEMRA_BF16_MMV` deliberately NOT enabled (owner decision outstanding), so every number here
  is the f32 trunk.
- Speculative decoding is off and cannot be on: every spec entry point `refuse_hyper`s this
  residual topology.
- This is a serving-configuration measurement, not a product claim. Any roster or performance
  claim goes through the product-facts workflow in the private repo.
