# DeepSeek V4 Flash

> **Serving bring-up reopened (owner goal, 2026-09-23).** The 2026-09-12 pause is lifted for the
> bring-up and tuning lane ([issue #4](https://github.com/avifenesh/memra/issues/4)). Support is
> still experimental: the numbers below are engine measurements with their conditions, not a
> serving-grade claim.

| | Recommended use |
|---|---|
| **Status** | Experimental engine support; functional and gated, not serving-grade |
| **Path** | Exact-source sharded Safetensors checkpoint through the dedicated two-card implementation; no GGUF conversion |
| **Hardware** | 2x RTX PRO 6000 Blackwell (the bring-up and tuning rig) |
| **Use this when** | You are evaluating or contributing to the experimental path, not deploying it as a supported server |

## Current served program

memra-server naked on 2x RTX PRO 6000 Blackwell Workstation, NVFP4 checkpoint
`tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8`: PP-2 over the two cards, matrix expert program,
host sampler, chunked prefill. 256 max tokens, single runs, 2026-09-23.

| arm | decode tok/s c1 greedy | TTFT p50 ms |
|---|---|---|
| plain | 15.86 | 537 |
| DSpark drafter (`MEMRA_DSV4_DRAFTER=dspark`) | 22.07 | 695 |

- Served DSpark greedy equals served plain greedy on every tested prompt since #660. Before that
  fix the two differed on 8/8 prompts.
- The route is serial: requests queue, so c>1 aggregate equals c1.
- Per-token weight traffic is 11.44 GB, a 157 tok/s bandwidth bound for PP-2 and 313 tok/s for
  TP-2. The TP/EP program (attention TP2 plus expert-ID EP) is faster per token but not servable
  yet ([issue #454](https://github.com/avifenesh/memra/issues/454)).

Receipts: `research/dsv4f-bringup-20260923/` (`BASELINE.md`, `spec-identity-660/RESULTS.md`).
Deeper history: the [DeepSeek section](../MODELS.md#deepseek-v4-checkpoint-dirs-serve-through-their-own-door-lanedsv4-flash-revival-20260822).
Do not infer production support from a successful load or prompt.
