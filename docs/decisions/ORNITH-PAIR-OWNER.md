# Ornith pair-owner MoE ordering — bank the study, keep rows

Date: 2026-08-25

Status: **runtime arm rejected; research result retained**

## Decision

Keep the existing slot/token rows program for Ornith's B=9..16 NVFP4 MoE
gate/up path. Do not merge the pair-owner CUDA kernels, FFI wrapper, dispatch
arm, or `MEMRA_MOE_PAIR_OWNER` flag. Do not re-admit cached CSR-NVFP4.

Bank the exact source-verbatim candidate and all raw receipts under
[`research/orndecode-pair-owner-20260825/`](../../research/orndecode-pair-owner-20260825/).

## Why

Every source-verbatim form preserves the original per-pair global-row
`expert_dot_g_v` program and passes B=2..16 bit/composition gates. Correctness
is not the blocker.

- One-warp owner forms lose 4.5% to 6.8% at B=16 in clean RTX PRO 6000
  windows.
- The final owner-ordered ordinary-warp form loses 5.3% in clean local RTX
  5090 Laptop timing.
- Its RTX PRO timing attempt ran while a separate tenant computed on the other
  card. Per the multi-card measurement law, those timed observations are
  discarded rather than used to promote or reject the final form on that card.
- Cached CSR-NVFP4 still violates batch-composition identity and was never a
  candidate for promotion in this lane.

The flags doctrine requires concluded losing arms to leave the runtime. An
inconclusive target transfer does not justify shipping dead code when the valid
measurements already point against the mechanism.

## Revisit condition

Reconsider only from the banked base-bound patch on an otherwise-idle RTX PRO
6000 box while holding `/tmp/memra-gpu.lock`. Require B=12/B=16 gate2+gate3,
same-process kernel timing, and balanced N>=5 ABBA with 250 ms telemetry. A
second-card tenant invalidates the scored window.
