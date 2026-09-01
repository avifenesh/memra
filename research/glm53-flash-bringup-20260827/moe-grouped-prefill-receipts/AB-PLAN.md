# MEMRA_MOE_GROUPED_PREFILL: box A/B plan (the flip condition)

The gate in this directory is exactness evidence only. The default flip needs throughput
receipts from the serving card class, and this is the protocol. The 4-card box currently runs
the 1M cell; the window is requested from the owner, never taken.

## Arms and protocol

Interleaved x5 per the interleaved-A/B law (box clock drift invalidates cross-run perf claims):

- **Arm OFF**: serving config as-is (`MEMRA_MOE_GROUPED_PREFILL` unset).
- **Arm ON**: identical config + `MEMRA_MOE_GROUPED_PREFILL=1`.
- Same binary, same artifact, same placement (full expert residency slabs; the arm is
  slab-only and fails closed elsewhere; a run with 0 engagements is VACUOUS, see receipts
  below). Alternate OFF/ON/OFF/ON... x5 each, fresh server boot per arm.

## Workload

1. **TTFD + prefill tok/s** at the ctxprobe prompt lengths (4,630 / 5,550 / 6,470 tokens; the
   receipted baseline is 57.4 / 67.1 / 78.8 s TTFD, flat 12.2-12.4 ms/token,
   `../ring-sizing-20260828/box-ctxprobe/BOXPROBE.md`), streamed, real prompts from the owner
   corpora (never synthetic).
2. **A 4096+ chunk-cap prompt** so the measured shape crosses `PRIME_CHUNK_MAX_TOKENS`.
3. **The sampled vendor-default twin** (serving law: a request with NO sampling params; a
   greedy-only receipt cannot justify a serving default).
4. **max_tokens=1 first-token argmax gate on real prompts** in both arms (the discriminator
   that rejected MEMRA_PP_BF16 and passed MEMRA_BF16_MMV): the ON arm's first token must match
   the OFF arm's per prompt, or the disagreement is escalated to a logit-delta cell before any
   flip.
5. **8-turn larger-prompt cache-on twin** per the multi-turn measurement law if the flip is to
   inform a serving default beyond the prefill claim.

## Engagement receipt (the step37 trap)

The `[moe-grouped-prefill]` announce line prints at first prefill in BOTH arms
(`flag=on|off`), so its presence is not an arm-local cost and a grep distinguishes the arms'
configs; ENGAGEMENT is the per-layer `[moe-grouped-prefill] execute layer=.. tokens=..` line
plus a nonzero `moe_grouped_prefill_dispatches` delta. A perf row from an ON arm whose log has
no execute lines is the step37 "batch prime that never ran" failure and is discarded, not
averaged.

## Numbers the A/B must produce

- TTFD per prompt length per arm, x5 interleaved, median + spread.
- Prefill tok/s (prompt_tokens / time-to-first-token, streamed).
- Engagement: execute-line count == 42 MoE layers x chunks per prompt in the ON arm.
- The sampled twin's tok/s + spec-engagement receipt (K>0), both arms.
- `MEMRA_PRIME_PROF=1` single pass (not part of the x5) for the router/gemm/scatter/shared
  split via `[moe-grouped-prefill-prof]`, to attribute whatever wall remains (the router host
  readback is expected to surface as the next term, L4 of PREFILL-GAP.md).

## Decision rule

Flip default ON only if: argmax gate green (or owner accepts the delta), sampled twin healthy,
TTFD improves at every measured prompt length with non-overlapping x5 spreads, and the FLAGS
row is updated with the measured rows in the same PR as the flip.
