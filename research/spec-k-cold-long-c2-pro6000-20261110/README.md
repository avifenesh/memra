# Two requests at once on the RTX PRO 6000 Max-Q: the cold-long flip against its parent (memra#1093, 2026-11-10)

Verdict: the flip (memra#1152: cold-long K=2 on this card, the cost model's draft-step fit) does
not lose with two requests at once. Pair aggregate against its parent: +14.9% at 1.2k tokens,
+4.9% at 3.6k, +0.6% at 5.8k (flat), +12.3% greedy at 3.6k, +12.4% on the short pair. Greedy
texts are byte-equal across arms. The parent's cost model serves half of the 1.2k, greedy and
short requests on plain decode at two sessions; the flip stays on K=2.

This is the cell owed to memra#1152, which froze before it ran. Model and rig: Qwen3.8-27B NVFP4
+ Q5_K GGUF with its untrimmed MTP head, context 8,192, two sessions, every switch at the device
default. One RTX PRO 6000 Blackwell Max-Q of a four-card box at its 250 W limit, one boot per
`tools/memra-gpu-run` lease, pinned to eight cores and their siblings; another tenant trained on
two other cards.

## The A/B

`c2p_ab.py` (the RTX 5090 cell's `c2_ab.py` with the arms as two binaries) through
`research/spec-cost-draft-step-20261109/ab_bins.py` (`run-c2.sh`, `run-c2-resume.sh`):

- **Arms:** `base`, main 53eb765421 (memra#1152's parent); `flip`, 774819dab6, the same tree as
  memra#1152's head 5898f5190d. Both built on the box.
- **Boots:** 12, base flip flip base three times. Boots 1 to 8 ran 00:15 to 00:26; the cell then
  stopped at a boot boundary for a quiet window on the box and boots 9 to 12 ran 00:31 to 00:37
  (`run/YIELD-quiet-window-0026-0033` is the stop file). Boot 9, a base boot, started on a cooled
  card (42 °C, busy SM clock 1,635 MHz against 1,522 to 1,582 in the others), which favours the
  base.
- **Pairs:** the six cold pairs of `research/spec-k-cold-long-c2-20261109/`, the two requests of
  a pair sent at once, answers capped at 400 tokens, thinking off. 144 requests, 0 errors.
- On this card the cost model owns K at every concurrency, so the K a request ran is read from
  its `[spec-k]` line. The short pair is not a same-K reference here.

## Results

Per request (`run/runs.jsonl`, `run/pairs.jsonl`):

| class | arm | K served | n | TTFT median / p95 (s) | decode median, range (tok/s) | pair aggregate median (tok/s) | ITL p95 median (ms) | acceptance |
|---|---|---|---:|---|---|---:|---:|---:|
| T=1 1.2k | base | K=3 x6, plain x6 | 12 | 1.032 / 1.087 | 61.93, 59.96-69.73 | 109.44 | 26.7 | 0.483 |
| T=1 1.2k | flip | K=2 x12 | 12 | 0.772 / 1.038 | 71.81, 68.15-77.95 | 125.77 | 35.9 | 0.671 |
| T=1 3.6k | base | K=3 x24 | 24 | 1.999 / 2.769 | 60.55, 50.78-72.96 | 92.72 | 34.9 | 0.480 |
| T=1 3.6k | flip | K=2 x24 | 24 | 2.044 / 2.776 | 62.20, 53.46-74.59 | 95.08 | 37.4 | 0.614 |
| T=1 5.8k | base | K=3 x12 | 12 | 4.429 / 4.442 | 60.67, 56.82-64.79 | 70.13 | 16.9 | 0.556 |
| T=1 5.8k | flip | K=2 x12 | 12 | 4.433 / 4.455 | 61.50, 54.81-67.83 | 70.56 | 17.0 | 0.605 |
| greedy 3.6k | base | K=3 x4, K=2 x2, plain x6 | 12 | 2.713 / 2.940 | 60.83, 52.00-80.25 | 89.51 | 21.5 | 0.506 |
| greedy 3.6k | flip | K=2 x12 | 12 | 2.071 / 2.768 | 68.34, 60.08-78.76 | 100.50 | 27.1 | 0.586 |
| short | base | K=3 x2, K=2 x4, plain x6 | 12 | 0.099 / 0.194 | 62.87, 60.47-70.11 | 117.19 | 27.5 | 0.549 |
| short | flip | K=2 x12 | 12 | 0.146 / 0.195 | 67.86, 66.68-69.19 | 131.71 | 39.3 | 0.561 |

(The acceptance column is over the requests that speculated.)

Per boot (each boot's median; 6 boots per arm; flip over base; `run/summary.json`):

| class | decode ratio | U | pair aggregate ratio | U | TTFT ratio |
|---|---:|---:|---:|---:|---:|
| T=1 1.2k | 1.155 | 36/36 | 1.149 | 36/36 | 0.851 |
| T=1 3.6k | 1.059 | 32/36 | 1.049 | 30/36 | 1.005 |
| T=1 5.8k | 1.007 | 20.5/36 | 1.006 | 20/36 | 1.001 |
| greedy 3.6k | 1.113 | 30/36 | 1.123 | 30/36 | 0.941 |
| short | 1.078 | 30/36 | 1.124 | 30/36 | 1.189 |

- No class loses. The 5.8k pair is flat.
- The 1.2k, greedy and short gains are mostly the parent's cost model leaving speculation: at two
  sessions its K=3 price sits within a few tok/s of plain (131.2 against 133.1, 122.1 against
  126.4 in boot 1) and it serves six of twelve requests of those classes plain. On the flip every
  request ran K=2.
- The short pair's first token is 23 ms later on the flip (median of per-boot medians 0.146
  against 0.123 s).
- Greedy texts: `deserts-4k` de407fdef1983a03 and `forests-4k` e923e7db50e84c31 in every boot,
  both arms (the RTX 5090 cell's shas too).
- At T=1 several prompts wrote more than one text within an arm (the 5.8k pair four on the flip
  arm). That is memra#1154: a seeded request's text follows how it interleaves with its
  companion, and here also which K and whether plain the cost model picked.

## Limits

One model, one card, two requests started together. Three sessions on this card are not measured.
