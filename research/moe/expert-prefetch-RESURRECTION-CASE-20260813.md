# Expert-prefetch prediction: RESURRECTION CASE — the closing scissors was a NVMe-latency artifact

Date: 2026-08-13. Owner raised the mechanism unprompted:

> "in moe theres also spec that you can train that will tell you ahead what are the routers decisions
> for a token before you need to route then you can start fetch before you start decode saving long time."

That is exactly this lane's mechanism, and this lane was **CLOSED negative** on 2026-07-23
(`expert-prefetch-prediction-pilot.md`). This note argues the closure was **hardware-scoped, not
mechanism-scoped**, and that its own stated resurrection bar is now met. It is a case for re-opening, not
a claim of a result.

## What was actually built and measured (credit where due — the machinery is real)

- Cross-layer router application, **zero training**: apply layer j's real router to layer k's MoE input.
  Measured on 173 captured decode steps (Hy3 Layer-103.5, local 5090). Deep half k≥48 reaches
  **84–100% argmax-hit at d=1–4**. Recorded as "the first measured mechanism whose information is not
  already captured by the LRU cache" (history predictors: 28–40%, all flat e2e).
- Full implementation lives in the tree, env-off and bit-identical: `crates/memra-engine/src/cpu_experts.rs`
  (`Predictor`, `PredictLayer`, `start_prefetch_predictor`), companion symbol
  `memra_cpu_expert_prefetch_v2`, flags `MEMRA_MOE_PREFETCH` (depth 1..8, default 0 = off),
  `MEMRA_MOE_PREFETCH_TOP` (default 1), `MEMRA_MOE_PREFETCH_MIN_LAYER` (default 40).
- Four A/B increments, all negative, with honest autopsies (no cross-token dedup → 37 GB of speculative
  reads per 32-token window; cold LRU-front insertion self-cancelling; annex redesign still showing zero
  promotions despite 39 GB of completed speculative reads).

## The stated closing reason, verbatim

> "a d=1-2 lookahead is ~2.6-5 ms of lead, but a queued 2-4 MB O_DIRECT read under load takes tens of ms
> — demand beats every prefetch to its own expert"

and the scissors table:

| lead | non-resident precision |
|---|---|
| d=1–2 (2.6–5 ms — **cannot beat read latency**) | 55–75% (k≥64) |
| d=16 (42 ms — beats read latency) | 10–34% |
| d=24–32 (62–83 ms) | 10–17% |

**Read that carefully: the scissors is defined by READ LATENCY, and read latency was NVMe's.** The
precision side was fine at short lead. The mechanism died because 2–4 MB could not be moved in 5 ms *on
that storage tier*.

## Why box1 inverts it

The pilot ran on a laptop 5090 spilling to NVMe. box1 (`sbox-2card`, 2× RTX PRO 6000) has **499 GB host
RAM, 299 GB available** — for Step-3.7-Flash Q8_0 the entire spilled tail (~4–20 GB) sits in RAM, so the
prefetch source is **host RAM over PCIe Gen5**, not disk.

| source | time to move ONE expert (16.8 MB Q8_0) |
|---|--:|
| NVMe O_DIRECT under load (the pilot's tier) | "tens of ms" |
| host RAM → PCIe Gen5 x16 (~64 GB/s) | **0.263 ms** |

So within the pilot's own short-lead windows, where precision is 55–75%:

| lead | experts movable from RAM | precision at that lead |
|--:|--:|---|
| d=1 ≈ 2.6 ms | **~9.9** | 55–75% |
| d=2 ≈ 5.0 ms | **~19.0** | 55–75% |

We need the true top-8. At 55–75% precision, fetching ~8–12 candidates covers it. **d=1–2 from RAM is
sufficient.** The pilot needed d=16+ *only because* its reads were slow, and d=16 is where the router
signal collapses to 10–34%. Move the source tier and the scissors opens.

This is squarely within the lane's own resurrection bar, which named three doors:
1. *"hardware with spare bus bandwidth relative to compute"* — box1: 499 GB RAM + PCIe Gen5, and the
   spilled fraction is only ~2–7% of the model rather than 76%.
2. *"KB-scale fetch granularity (sub-expert/projection-fragment storage)"* — untried.
3. *"a fundamentally better long-lead predictor (trained probe on deeper context, not the router
   cross-application)"* — **exactly what the owner proposed.**

Door 1 alone may be enough. Door 3 is the owner's idea and would deepen the lead beyond d=2.

## What must be re-measured, not assumed

The pilot's precision table is **Hy3** (192 experts, top-8). Step-3.7-Flash is **288 experts, top-8**,
45 layers of which 42 are MoE, with a different router (sigmoid, `moe_router_scaling_factor` 3.0,
`use_moe_router_bias`). Precision and skew MUST be re-captured for Step; do not port Hy3's numbers.

Configuration defaults are also wrong for Step and would silently under-test it:
- `MEMRA_MOE_PREFETCH_MIN_LAYER` defaults to **40**, so on a 45-layer model only ~5 layers would predict.
  The pilot's k≥48 gate came from a 79-MoE-layer Hy3; the equivalent *fraction* for Step is ≈ layer 24.
- `MEMRA_MOE_PREFETCH_TOP` defaults to **1**, but the RAM-tier budget above affords 8–12.

Budget sanity, keeping the pilot's own discipline (speculation ≤ ¼ of demand io): demand expert traffic is
**5.65 GB/token**; with only ~7% of the bank spilled, demand MISS traffic is ≈**396 MB/token**, and
speculating 12 experts costs ≈**202 MB**. Same order as the misses it would eliminate — plausible, and it
must be *measured against `phase_compute`*, not just hit rate, because the pilot's invariant finding was a
fabric tax that inflated compute in every arm (2.87 → 3.4–12.6 s). That tax was measured at NVMe volumes;
re-price it at RAM volumes.

## Why this matters commercially, right now

`cx-bigtier` is measuring whether Step-3.7-Flash Q8_0 (209.42 GB) can serve on two cards (205.28 GB) with
a ~2–7% spill. The naive serial miss cost is the entire risk: 5.65 GB of expert bytes per token means a 3%
miss adds ~2.5 ms against a ~2.9 ms fully-resident floor. **A working prefetch predictor removes precisely
that term** — it is the difference between "2 cards work, zero capex" and "buy a third card". Per
[the sizing note](../bigtier-sizing-20260813.md) that is a real hardware-purchase decision.

## Per-hardware framing (owner doctrine)

Under §Per-hardware arm selection this is not "was the pilot right or wrong" — it is a **per-rig default**.
The pilot's verdict stands *for a laptop 5090 spilling to NVMe*. The question re-opened here is whether the
arm should be ON for a PRO-6000-class box with ~500 GB of host RAM. A one-rig negative sets a one-rig
default at most, and `MEMRA_MOE_PREFETCH` already exists as the measurement seam — nothing needs to be
built to ask the question.

## Effort, in the owner's units

Per CLAUDE.md §Agent-time scale (1 agent-day ≈ 1 human-week): re-capturing Step routes and re-running the
existing env-gated arm on box1 is **hours**, because the machinery is already in the tree. A trained
long-lead probe (door 3) is **week-class**. Neither is a reason to defer.
