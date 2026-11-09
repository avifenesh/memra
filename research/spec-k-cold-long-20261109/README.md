# Cold long prompts at K=2 against K=3 on the RTX 5090 (memra#1093 follow-up, 2026-11-09)

Verdict: on the RTX 5090, K=2 beats the shipped cold-long K=3 on cold prompts of about 4,000 tokens
and more, and does not on prompts of about 1,400.

- **Sampled (T=1), 4k and 6k:** decode +5% to +6%.
- **Greedy, 4k:** +9% to +12%.
- **1.4k:** sampled and greedy are flat.
- **TTFT:** unchanged at every length.

Nothing is flipped. The cold-long rung covers every prompt of 1,024 tokens or more, and these
numbers say its best depth depends on the length within that class. A default change needs a PRO
6000 A/B as well (AGENTS.md: one-rig evidence sets a one-rig default at most).

Model and rig: Qwen3.8-27B NVFP4 + Q5_K GGUF with its untrimmed MTP head, no DFlash drafter, RTX
5090 Laptop, context 8,192, two sessions, prefix cache at its derived size, every switch at the
device default. Binary: main 1273a16f07 (the cell-evidence build, sha256 96ad59c39c4afdfb).

## The A/B

`cold_long_ab.py` through `ab_driver.sh`:
- **Arms**, on one binary: `k3`, the device default (cold-long K=3), and `k2`, with
  `MEMRA_SPEC_K_COLD_LONG=2`. That is the existing measurement door, and it moves the cold-long rung
  only.
- **Boots:** one boot per gpulock lease, in the order k3 k2 k2 k3 three times (both orders, six
  boots per arm). The driver yields between boots to a waiter whose reason says "landing".
- **Prompts:** every boot serves the same cold prompts, one at a time. Each is distinct from its
  first words, so no prefix-cache entry serves another. Each asks for about 300 words, with answers
  capped at 400 tokens (every request reached the cap) and thinking off.
- **Streamed:** TTFT is the first content chunk, and the decode rate is completion tokens over the
  time after it. ITL comes from the chunk gaps. GPU telemetry is sampled at 250 ms.
- **Noise reference:** two short T=1 prompts (34 tokens) sit on the cold-short rung, K=3 on both
  arms, so their rate tracks the boot's clock and nothing else. The card throttled through both
  runs (84-87 °C, SM clock 1,342-1,672 MHz between boots, the reference at 47-63 tok/s), so each long
  row is also read normalized by its own boot's reference median.
- **Separation test:** U counts, over the 36 pairs of a k2 boot and a k3 boot, how often the k2
  boot's median is higher. 36 of 36 is complete separation, the bar `tools/cell_evidence.py`'s perf
  check uses. Set before run 2: K=2 wins only with complete separation and a gain of at least 3%.

Run 1 (`run1/`): four T=1 prompts of about 1,240 tokens, two of about 3,700, two greedy of about
1,240, and the short reference. Run 2 (`run2/`, `PROMPT_SET=long`, the follow-up on run 1's 4k
lean): four T=1 prompts of about 3,600 tokens, two of about 5,800, one greedy of 3,565, and the
short reference.

## Results

Per-request medians and spreads (`run1/analysis.md`, `run2/analysis.md`):

| run | class | arm (K) | n | TTFT p50 / p95 (s) | decode median, range (tok/s) | pooled (tok/s) | acceptance |
|---|---|---|---:|---|---|---:|---:|
| 1 | T=1 1.4k | k3 (3) | 24 | 1.149 / 1.290 | 65.31, 55.45-69.97 | 64.44 | 0.471 |
| 1 | T=1 1.4k | k2 (2) | 24 | 1.177 / 1.464 | 63.19, 53.09-70.68 | 62.92 | 0.614 |
| 1 | T=1 3.7k | k3 (3) | 12 | 3.465 / 4.166 | 59.73, 51.89-63.55 | 59.08 | 0.472 |
| 1 | T=1 3.7k | k2 (2) | 12 | 3.596 / 4.188 | 63.30, 55.93-68.66 | 62.61 | 0.647 |
| 1 | greedy 1.4k | k3 (3) | 12 | 1.252 / 1.427 | 65.28, 57.73-73.93 | 65.32 | 0.470 |
| 1 | greedy 1.4k | k2 (2) | 12 | 1.298 / 1.423 | 66.69, 58.70-69.47 | 65.00 | 0.578 |
| 2 | T=1 3.6k | k3 (3) | 24 | 3.247 / 3.817 | 58.83, 50.31-67.83 | 58.58 | 0.440 |
| 2 | T=1 3.6k | k2 (2) | 24 | 3.310 / 3.824 | 60.84, 54.25-67.97 | 61.07 | 0.585 |
| 2 | T=1 5.8k | k3 (3) | 12 | 5.809 / 6.178 | 57.67, 49.79-67.52 | 57.44 | 0.483 |
| 2 | T=1 5.8k | k2 (2) | 12 | 5.635 / 6.058 | 60.42, 56.76-64.56 | 60.48 | 0.620 |
| 2 | greedy 3.6k | k3 (3) | 6 | 3.495 / 3.877 | 55.92, 52.54-63.06 | 55.96 | 0.380 |
| 2 | greedy 3.6k | k2 (2) | 6 | 3.424 / 3.870 | 60.80, 58.03-63.80 | 60.92 | 0.531 |

Per boot (each boot's median; 6 boots per arm; ratio = k2 median over k3 median):

| run | class | normalized decode ratio | U normalized | raw decode ratio | U raw | TTFT ratio | U TTFT |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | T=1 1.4k | 0.996 | 16/36 | 0.984 | 19/36 | 1.041 | 24/36 |
| 1 | T=1 3.7k | 1.070 | 31/36 | 1.069 | 26/36 | 1.038 | 21/36 |
| 1 | greedy 1.4k | 1.008 | 19/36 | 1.009 | 19/36 | 1.024 | 21/36 |
| 2 | T=1 3.6k | 1.056 | 31/36 | 1.064 | 28/36 | 0.976 | 19/36 |
| 2 | T=1 5.8k | 1.058 | **36/36** | 1.057 | 28/36 | 0.998 | 16/36 |
| 2 | T=1 all | 1.049 | 31.5/36 | 1.045 | 27/36 | 1.006 | 18/36 |
| 2 | greedy 3.6k | 1.121 | **36/36** | 1.087 | 31/36 | 0.972 | 14/36 |

The raw columns separate less cleanly than the normalized ones because boot 1 of each run, a k3
boot, ran coolest (80 and 84 °C, 1,635 and 1,672 MHz, against 86-87 °C after). The normalization
takes that out. ITL p95 is 34-36 ms at K=2 against 39-42 ms at K=3; p50 is 0 on both, because a
spec round emits its tokens as back-to-back chunks.

The greedy texts are byte-equal across arms for every greedy prompt (one sha per prompt), as the
K-invariance receipt in `research/spec-k-resume-20261109/` found.

## Reading

- Prompts of about 1,400 tokens: K=2 and K=3 tie, for T=1 and for greedy. K=2's higher acceptance
  per draft (0.61 against 0.47) does not pay for one draft less per round.
- About 3,600 to 5,800 tokens: K=2 decodes 5% to 7% faster at T=1, on both runs and both lengths.
  It meets the pre-set bar (complete separation, at least 3%) at 5.8k. At 3.6k to 3.7k it misses
  it narrowly, U = 31/36 on both independent runs. Greedy at 3.6k gains 9% to 12% with complete
  separation. Longer context makes each verify round dearer, so the shallower round wins.
- TTFT does not move: K only shapes decode.
- So the shipped split, K=3 for every cold prompt from 1,024 tokens up, is too coarse on this card.
  A length-aware rung (K=2 from about 4,000 tokens) is the candidate. Where between 1.4k and 3.6k
  the crossover sits is not measured. The PRO 6000 must decide too before any default moves. The
  2026-08-08 matrix that set the table (`research/kpolicy-20260808/`, another rig, aggregate tok/s
  with the prefill inside) had q27 cold-long at 5.4k with K=3 ahead of K=2 by 6%, so the card class
  may well disagree.
