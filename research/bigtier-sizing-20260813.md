# Big-tier resident/spill sizing — corrected units, measured artifact, and the 2-card spill case

Date: 2026-08-13. Supersedes any earlier "Step needs 3 cards" claim from this program, **including my
own** — that claim came from a GB/GiB unit error, documented below so it is not repeated.

## The unit error, stated plainly

`nvidia-smi` reports an RTX PRO 6000 Blackwell Server Edition as **97,887 MiB**. That is:

- 97,887 MiB × 1,048,576 = **102,653,657,088 bytes = 102.64 GB decimal = 95.59 GiB**
- **Two cards = 205.28 GB decimal**

An earlier card-count screen in this program treated the card as "96 GB" and divided decimal-GB weight
footprints by it. That understates capacity by ~7% — which is immaterial for a 27 GB model and
**decisive at the 200 GB boundary.** Any sizing conclusion downstream of "96 GB per card" needs redoing.

## Measured artifact (not estimated)

`stepfun-ai/Step-3.7-Flash-GGUF` @ `0b69336d2fd2adfdef9c66e425f7778196c31482`, official StepFun GGUF,
staged on box1 local NVMe at `/opt/scratch/nvme/models/step-3.7-flash/Q8_0/`:

| shard | bytes |
|---|--:|
| 00001-of-00005 | 46,446,888,192 |
| 00002-of-00005 | 46,094,750,560 |
| 00003-of-00005 | 46,099,485,664 |
| 00004-of-00005 | 46,175,264,256 |
| 00005-of-00005 | 24,601,482,592 |
| **total** | **209,417,871,264 ≈ 209.42 GB** |

This is **owner-compliant**: 8-bit, published by the model author, and not REAP-pruned.

It also validates the bit-width arithmetic. Predicted Q8_0 footprint from the config was 195.8B params ×
8.5 bpw (int8 + one fp16 scale per 32 weights) = **208.0 GB**; actual is **209.4 GB**, a 0.7% miss
explained by non-expert tensors and metadata. So the Q8_0 model is ~8.56 bits/weight, not 8.0 — the
distinction between "fp8" (1.000 B/param → 195.8 GB, which WOULD fit two cards with 9.5 GB spare) and
"Q8_0 GGUF" (209.4 GB) is what decides card count.

## The 2-card spill case (owner call 2026-08-13)

Owner: *"if we spill, might be worth try spill to 2 cards and try first, theres some very strong spill
methods if being done right."*

The arithmetic strongly supports trying it:

| | value |
|---|--:|
| artifact | 209.42 GB |
| two cards | 205.28 GB |
| **minimum spill, weights only** | **4.14 GB = 2.0% of the model** |
| spill with a ~10 GB KV/activation budget | ~14 GB = 6.7% of the model |
| expert bank | 190.3 GB = 97.2% of the model |
| one expert (3 × 4096 × 1280 @1 B/param) | 15.7 MB |
| per MoE layer | 288 experts = 4.53 GB |

So a 14 GB spill is **the coldest ~20 of 288 experts per layer** (7.4% of the bank). With **top-8 of 288
routing**, expert usage is heavily skewed, so the miss rate on a correctly-chosen cold tail should land
*below* the spilled fraction — that is the whole bet, and it is measurable.

**box1's memory tiers make this the favourable case, not the Hy3 case:**
- **499 GB host RAM** (`free -g`: 499 total, 299 available). The entire spilled tail lives in RAM with
  enormous room to spare — spill is **PCIe-bound, not disk-bound**.
- 2.8 TB free on local NVMe as the cold backstop.
- 48 vCPU, single NUMA node (Xeon Platinum 8559C) — no cross-socket penalty on host-staged buffers.

### Miss-cost arithmetic — a hypothesis to test, not a result

Expert bytes read per token = 42 MoE layers × 8 experts × 15.7 MB = **5.28 GB/token**. Against ~64 GB/s
PCIe Gen5 x16:

| expert miss rate | PCIe bytes/token | added ms/token |
|--:|--:|--:|
| 10% | 528 MB | 8.26 |
| 5% | 264 MB | 4.13 |
| 3% | 159 MB | 2.48 |
| 1% | 53 MB | 0.83 |

Reference floor: 10.3B active params ⇒ ~10.3 GB HBM read/token; two cards ≈ 3.58 TB/s aggregate ⇒
**2.88 ms/token = ~348 tok/s at B=1** if fully resident.

So at a 3% miss rate the naive serial cost roughly doubles per-token latency — **which is exactly why
prefetch/overlap and batch amortization decide this, not the miss rate alone.** At batch > 1 one fetched
expert serves many tokens, and memra's async prefetch is designed to hide the transfer behind compute.
Measure the overlap; do not assume either the optimistic or pessimistic end.

### Why this is NOT the Hy3 phase-1 regime

Hy3 phase 1 measured **0.15 tok/s** and it is the wrong anchor for this campaign. That run put an 81.5 GB
checkpoint against a **19.2 GB** VRAM budget — roughly **76% spill**, streaming from NVMe, at ~48 GB/token
of H2D traffic. This is **2-7% spill into 499 GB of host RAM**. Two different problems; do not carry the
0.15 tok/s number, or any intuition built on it, into this analysis.

## What the lane is doing (`cx-bigtier`, live on box1)

Its own script header: *"One-lock Step-3.7-Flash Q8_0 correctness, spill, route-overlap, and control
campaign"* — PP-2 near-resident SLRU, `MEMRA_SPILL_STATS=1`, `MEMRA_PREFIX_PARTIAL_RESTORE=0`, grouped
left unset (default off, per the `MEMRA_MOE_GROUPED` exactness block). Download completed 07:43.

The order that matters, and it is the owner's order: **prove 2-card spill first; a 3rd card is the
fallback, not the plan.** A measured 2-card number is what makes the card question answerable either way —
if spill is cheap we need no capex at all, and if it is expensive the measured penalty is the receipt.

## Standing rule this establishes

Card-count screens state **which unit** and **which quant format**. "96 GB" and "8-bit" are both
ambiguous enough to flip a 2-vs-3-card answer:
- card: 96 GiB = 102.64 GB decimal
- 8-bit: fp8 = 1.000 B/param; Q8_0 GGUF = ~8.56 bits/weight = 1.070 B/param
