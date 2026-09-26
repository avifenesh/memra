# The first DSv4F pair was power-braked: every #662 perf number is about 2x slow

Scope: one rented 2x RTX PRO 6000 Blackwell Workstation Edition pair, measured 2026-09-23,
against a second healthy pair of the same card, driver 595.71.05 on both. Power limit 600 W on
the braked pair and 500 W on the healthy one (`raw/*/hw.csv`); decode draws under 300 W per card
on either, so the limit is not what separates them.

## Finding

Both GPUs of the pair behind #662 had the hardware power-brake input latched:

```
HW Slowdown                    : Active
    HW Power Brake Slowdown    : Active
HW Power Braking               : 657571315734 us
```

(`raw/braked/nvsmi-perf.txt`, same at the end of the session in `nvsmi-perf-after.txt`.)
The clocks-event mask was `0x88` (HW slowdown plus power brake) in every telemetry row.
Power brake is an external signal from the board or chassis, not a driver or application
setting; 657,571 s is about 7.6 days of accumulated braking.

The reported clock hides it. During a clock64/globaltimer FMA spin (`raw/src/clk.cu`, 188x4
blocks of 256 threads) nvidia-smi reported 2865 MHz, while the ratio of SM cycles to wall time
was 722 MHz:

| pair | effective SM clock (8 reps, 2 cards) | nvidia-smi SM clock | braking counter |
|---|---|---|---|
| braked | 719..723 MHz | 2865 MHz | 657,571 s |
| healthy | 2839..2861 MHz | (spin shorter than the 250 ms sampler) | 0 s |

DRAM and the peer link were the same on both pairs, which is why the memory-bound parts of the
profile looked plausible:

| probe (`raw/src/bw.cu`, `p2p.cu`) | braked | healthy |
|---|---|---|
| DRAM read kernel | 1641.7 GB/s | 1646.6 GB/s |
| D2D copy (r+w) | 1468.2 GB/s | 1465.9 GB/s |
| 26.7 MB L2-resident read | 12.8 us | 4.1 us |
| memcpyPeer 256 MiB | 53.87 GB/s | 52.58 GB/s |
| peer flag ping-pong round trip | 2.03 us | 1.45 us |
| empty launch, host rate | 1.74 us | 1.72 us |

So L2- and issue-bound work (the M=1 expert tiles, the small-kernel tail, Sinkhorn) ran at a
quarter of the clock, and HBM-bound streams ran at full speed.

## Proof it is the card and not the code

The 2026-09-08 TP/EP full-token replay anchor (`research/dsv4f-full-token-replay-20260908/`,
source `bd30a57`, 2x RTX PRO 6000 Max-Q) measured 42.80 tok/s eager and 44.01 graph, prime wall
5.88 s. The same source on the braked pair (`raw/braked/tpep-replay-bd30a57.log`; the tree
carries a diagnostic patch that prints instead of asserting the pinned source hash, because the
gate tape was rebuilt under #657) produced the same generated-token sha256 `35e9e69e...`, final
logits sha256 `37eb73d8...`, cache digests `5673480229060882075` and hidden digests
`4231551965497114380`, at:

| arm | rows | tok/s |
|---|---|---|
| eager | 10 | 20.07 |
| graph | 10 | 19.86 |

Prime wall 12.45 s. Same program, same bits, 2.1x slower. The healthy-pair replay of current main
is in `../REBASELINE.md`.

## What is invalid and what stands

- Invalid: every rate and timing measured on the braked pair. That is `../BASELINE.md` (served
  cells, the nsys profile and its lever ranking), the served-cell rates in
  `../spec-identity-660/RESULTS.md`, and the rates in the #664 issue text. They stay in place with
  a banner, per the corpus rule; `../REBASELINE.md` holds the replacements.
- Stands: every correctness verdict from that pair (text and token hashes, spec == plain,
  batched == sequential, the #660 bisect). A clock does not change arithmetic, and the replay
  above reproduced the historical hashes exactly.

## Acceptance check added

Before staging weights on a rented card: run `clk.cu` and require about 2840 MHz effective on a
Workstation card, and require `nvidia-smi -q -d PERFORMANCE` to show HW Slowdown and HW Power
Brake Slowdown Not Active with a 0 us braking counter. A healthy DRAM number does not clear a
card.
