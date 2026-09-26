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

memra-server naked on 2x RTX PRO 6000 Blackwell, NVFP4 checkpoint
`tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8`. Since 2026-09-25 (memra #710) the two-card load
is TP/EP:
- every layer sits on both cards;
- experts are split by id;
- attention is split by head (exact attention TP2);
- the DSpark drafter's experts are split the same way.

Plain requests replay each step on full-token CUDA graphs, with the device sampler.
`MEMRA_DSV4_TOPOLOGY=pp` is the PP-2 rollback (decision:
[DSV4-TPEP-DEFAULT](../decisions/DSV4-TPEP-DEFAULT.md)). All rows: 256 max tokens, one boot per
row, median of N=3 rows (N=2 for DSpark), 2026-09-24/25, same text on every request of every arm.

| arm | Workstation pair, c1 greedy | Server Edition pair, c1 greedy / sampled | TTFT p50 (SE) |
|---|---|---|---|
| plain, TP/EP (default) | 80.73 | 71.20 / 72.65 | 172 ms |
| plain, PP-2 (`pp`) | 68.51 | 62.38 / 58.40 | 210 ms |
| DSpark (`MEMRA_DSV4_DRAFTER=dspark`), TP/EP | 82.73 | 72.79 / 63.57 | 207 ms |
| DSpark, PP-2 | 70.60 | 61.72 / 53.90 | 267 ms |

Concurrency: the plain TP/EP route serves four lanes whose steps share one captured B-row
graph step (memra #710). Aggregate on the Workstation pair:

| | c2 | c4 |
|---|---|---|
| TP/EP, four lanes | 102.0 tok/s, TTFT 0.24 s | 132.8, TTFT 0.42 s |
| PP-2, two pipelined lanes | 120.9, TTFT 0.30 s | 120.6, TTFT 4.5 s |

Known cost of TP/EP:
- **Context.** A session holds about 370k tokens with DSpark and 790k plain, against PP-2's 1M.
  The head-split KV lane follows.

Receipts: `research/dsv4f-bringup-20260923/tpep-default/RESULTS.md`, `tp-rows/RESULTS.md`,
`dspark-ep/RESULTS.md`, `ceiling/CEILING.md`.

### Before the flip: PP-2, 2026-09-23

Medians of four boots per arm on the Workstation pair: plain 63.34 greedy / 57.38 sampled (TTFT
192 ms), DSpark 76.07 / 62.47 (TTFT 240 ms).

What the rows carry, each step measured against the tree before it with the same text on every
prompt:

| step | plain greedy | DSpark greedy |
|---|---|---|
| one-token MoE stream visitor (#664) | +29.2% over the sktail tail | |
| multi-row stream visitor on the verify rounds (#669) | | +26.8% |
| deferred MoE checks (#670): one fault readback per transaction instead of 129 synchronizing reads per step | +6.8% | +2.7% |
| small-kernel diet (#339): one fused HC finish and one fused Q norm/pack per one-token step | +4.3% (measured before #670) | flat |
| latency kernels: register-resident rmsnorm, one-warp router, one-warp expert prefix, grouped `wo_a` on the dense-fast program | +13.3% (55.92 -> 63.34) | +3.9% (73.23 -> 76.07) |

The diet and grouped `wo_a` change one-token programs only; the other latency kernels run on
verify rows too, where a round pays its layers once per 3.56 committed tokens. That is why
DSpark gains less than plain from these steps.

- The first pair measured for this bring-up had the hardware power brake latched (722 MHz
  effective SM clock behind a reported 2865 MHz) and produced 15.86 / 22.07 tok/s. Those rates
  are withdrawn; the correctness verdicts from that pair stand.

- Served DSpark greedy equals served plain greedy on every tested prompt since #660. Before that
  fix the two differed on 8/8 prompts.
- Per-token weight traffic is 11.44 GB. At the measured 1.54 TB/s practical read bandwidth
  that bounds c1 at 135 tok/s for PP-2 and 270 tok/s for TP-2 (`ceiling/CEILING.md`).

Receipts: `research/dsv4f-bringup-20260923/` (`REBASELINE.md`, `m1-stream-664/RESULTS.md`,
`mrow-stream/RESULTS.md`, `moe-defer-670/RESULTS.md`, `small-diet/RESULTS.md`, `latency/RESULTS.md`, `PRIOR-ART.md`, `power-brake/POWER-BRAKE.md`, `spec-identity-660/RESULTS.md`; `BASELINE.md` is the withdrawn braked-pair run).
Deeper history: the [DeepSeek section](../MODELS.md#deepseek-v4-checkpoint-dirs-serve-through-their-own-door-lanedsv4-flash-revival-20260822).
Do not infer production support from a successful load or prompt.
