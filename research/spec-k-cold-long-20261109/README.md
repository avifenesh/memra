# Cold long prompts at K=2 against K=3 on the RTX 5090 (memra#1093 follow-up, 2026-11-09)

Verdict: K=2 beats the shipped cold-long K=3 on both cards measured, with TTFT unchanged. Nothing
is flipped here; this receipt is the evidence the default decision needs.

- **RTX 5090 Laptop**, from about 3,600 tokens: T=1 decode +5% to +7%, greedy +9% to +12%. At about
  1,400 tokens it is flat for both.
- **RTX PRO 6000 Blackwell Max-Q** (250 W cap), at every length measured (1,200 to 5,800 tokens):
  T=1 +3.9% to +7.4% and greedy +1.6% to +9.6%, every class with complete separation.
- **The cost model:** on the PRO 6000 it owns K, and it chose K=3 for these prompts. The measurement
  says that choice is slower.

Model and rig: Qwen3.8-27B NVFP4 + Q5_K GGUF with its untrimmed MTP head, no DFlash drafter, RTX
5090 Laptop, context 8,192, two sessions, prefix cache at its derived size, every switch at the
device default. RTX 5090: main 1273a16f07, the cell-evidence build (sha256 96ad59c39c4afdfb). PRO
6000: the same commit built on that box with the release kit's toolchain (295f47a1af68f9af,
`pro6000-maxq/build.sh`), the same GGUF (15,705,922,304 bytes).

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

## RTX PRO 6000 Blackwell Max-Q (`pro6000-maxq/`)

The same two prompt sets ran on one Max-Q card of a shared four-card box:
- **Lease and pinning:** one boot per `tools/memra-gpu-run` lease (the card's lock and the box's
  campaign lock), in the same k3 k2 k2 k3 x3 order (`ab_wrx.py`, `run-ab.sh`), pinned to eight
  cores and their siblings.
- **Power and neighbors:** the card's power limit was left at its 250 W; another tenant trained at a
  steady load on two other cards. The card held 87 °C and 1,567-1,575 MHz after the first boot of
  each run (boot 1: 75-78 °C, a k3 boot).
- **No normalization:** the cost model owns K on this card (`[spec-k] ... source=cost-model`). It
  picks the best K up to the rung from the acceptance history of the boot's own rounds. So in the
  k2 arm it also priced K=2 for the short reference prompts, and those rows are not a fixed-K
  reference here. This card's comparison is on raw rates. Its drift is small: the k3 boots of a run
  sit within 0.4 tok/s of each other, except boot 1.
- **Clean leases:** all 24 commands exited 0. Leases 2 and 3 of the long run are marked unclean
  because a telemetry process outlived the server by a moment; the script waits for it since.

| set | class | K default / k2 | n | per-boot median, default / K=2 (tok/s) | ratio | U | TTFT ratio (U) | acceptance |
|---|---|---|---:|---|---:|---:|---|---|
| base | T=1 1.2k | 3 / 2 | 24 | 100.63 / 105.14 | 1.045 | 36/36 | 1.003 (34) | 0.463 / 0.601 |
| base | T=1 3.7k | 3 / 2 | 12 | 99.59 / 106.98 | 1.074 | 36/36 | 0.999 (14) | 0.472 / 0.647 |
| base | greedy 1.2k | 3 / 2 | 12 | 110.11 / 111.86 | 1.016 | 36/36 | 0.998 (13.5) | 0.470 / 0.574 |
| long | T=1 3.6k | 3 / 2 | 24 | 96.52 / 101.83 | 1.055 | 36/36 | 1.000 (17) | 0.440 / 0.585 |
| long | T=1 5.8k | 3 / 2 | 12 | 98.75 / 102.63 | 1.039 | 36/36 | 1.000 (20) | 0.483 / 0.620 |
| long | greedy 3.6k | 3 / 2 | 6 | 96.11 / 105.35 | 1.096 | 36/36 | 0.999 (16) | 0.380 / 0.531 |

Pooled decode, default against K=2: base T=1 100.77 / 106.15, long T=1 98.20 / 102.25. TTFT p50 and
p95 match to the millisecond (for example T=1 3.6k: 1.373 / 1.439 s against 1.374 / 1.439 s). The
greedy texts are byte-equal across arms on every prompt.

The cost model's calibration line reads `[spec-cost] model="q" depth=2048 ...`: its verify costs
are measured at a context of 2,048. The prompts here, and their decode, run at 1,200 to 6,200. That
may be why it prices K=3 above K=2 where K=2 is faster; it is not tested here.

## Both cards

| prompt length | RTX 5090, T=1 | RTX 5090, greedy | PRO 6000 Max-Q, T=1 | PRO 6000 Max-Q, greedy |
|---|---|---|---|---|
| 1,200-1,400 | x0.996, flat (16/36) | x1.008, flat (19/36) | x1.045 (36/36) | x1.016 (36/36) |
| 3,600-3,700 | x1.056 / x1.070 (31/36, two runs) | x1.121 (36/36) | x1.055 / x1.074 (36/36) | x1.096 (36/36) |
| 5,800 | x1.058 (36/36) | not run | x1.039 (36/36) | not run |

(5090 ratios are drift-normalized per boot; PRO 6000 ratios are raw. TTFT does not move on either.)

## Reading

- K=2 is at least as fast as K=3 on every cold long prompt measured, on both cards, with greedy
  texts unchanged and TTFT unchanged. Where it loses nothing (the 5090 at 1.4k) it ties. Its
  acceptance per draft is higher (0.53-0.65 against 0.38-0.48), and the round it drops costs more
  than it returns, more so as the context grows.
- A cold-long rung of K=2 is therefore the candidate default for this model on these two cards: a
  plain rung change, with no length split needed, since the 5090's 1.4k tie costs nothing.
- Not covered:
  - concurrency (all cells are single-stream; the spec gate demotes to plain from c=3 on a single
    card either way);
  - the GB10 (sm_121a shares the table; its own rung measurements live in
    `research/gb10-tuning-20261102/`);
  - other models (the 2026-08-08 matrix that set the table, `research/kpolicy-20260808/`, measured
    q9 and q27 on another rig with the prefill inside the rate and had q27 cold-long at 5.4k with
    K=3 6% ahead of K=2);
  - the trimmed-head rung.
- On the PRO 6000 the cost model owns K, and a rung of 2 caps its choice at 2. Its K=3 choice here
  is the slower one, so the cap removes nothing it gets right on these prompts.
