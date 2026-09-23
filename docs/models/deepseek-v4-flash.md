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
host sampler, chunked prefill. 256 max tokens, single runs, 2026-09-23, on a pair whose cards passed
the effective-clock acceptance check (see below).

| arm | decode tok/s c1 greedy | decode tok/s c1 sampled | TTFT p50 ms |
|---|---|---|---|
| plain | 50.10 | 36.54 (before #664) | 200 |
| DSpark drafter (`MEMRA_DSV4_DRAFTER=dspark`) | 56.01 (before #664) | 47.47 (before #664) | 267 |

Plain greedy is the median of five boots with the one-token MoE stream visitor (#664, +29.2% over
the 38.79 tok/s sktail tail it replaced, same text on every prompt). The other cells are the
rebaseline before #664 and are re-measured on top of it as the lane lands.

The TP/EP program on the same pair replays the 2026-09-08 anchor protocol at 50.04 tok/s eager
and 50.68 graph with the anchor's exact bits (anchor: 42.80 / 44.01).

- The first pair measured for this bring-up had the hardware power brake latched (722 MHz
  effective SM clock behind a reported 2865 MHz) and produced 15.86 / 22.07 tok/s. Those rates
  are withdrawn; the correctness verdicts from that pair stand.

- Served DSpark greedy equals served plain greedy on every tested prompt since #660. Before that
  fix the two differed on 8/8 prompts.
- The route is serial: requests queue, so c>1 aggregate equals c1.
- Per-token weight traffic is 11.44 GB, a 157 tok/s bandwidth bound for PP-2 and 313 tok/s for
  TP-2. The TP/EP program (attention TP2 plus expert-ID EP) is faster per token but not servable
  yet ([issue #454](https://github.com/avifenesh/memra/issues/454)).

Receipts: `research/dsv4f-bringup-20260923/` (`REBASELINE.md`, `m1-stream-664/RESULTS.md`,
`power-brake/POWER-BRAKE.md`, `spec-identity-660/RESULTS.md`; `BASELINE.md` is the withdrawn braked-pair run).
Deeper history: the [DeepSeek section](../MODELS.md#deepseek-v4-checkpoint-dirs-serve-through-their-own-door-lanedsv4-flash-revival-20260822).
Do not infer production support from a successful load or prompt.
