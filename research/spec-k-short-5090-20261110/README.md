# Draft length on the RTX 5090 Laptop: where the cost model owns it, and the cold-short rung (2026-11-10)

Verdict, three cells, memra#1093:

1. **Qwen3.5-9B, where the spec cost model owns K on this card.** K=2 is the faster depth (3.6%
   over K=3 on the geometric mean of nine prompts, ahead in 141 of 144 boot pairs), and the
   model picks it: the first request of a boot runs at K=3 while it has no acceptance history,
   every later one at K=2. The draft length that differed between a cold run and its cache hit
   in the hit gate is that exploration, not an unstable choice. No change.
2. **Qwen3.8-27B, where the table owns K.** On short prompts K=2 against the table's K=3 is 1.4%
   ahead on the geometric mean of twelve prompts, with nine ahead and three 2 to 3% behind. The
   sign follows the prompt's acceptance (r = -0.88 over fifteen short prompts): K=2 gains 3 to
   7% where K=3 accepts 0.42 or less per drafted token, is within 3% either way from 0.45 to
   0.58, and loses 2 to 3% at 0.61 and 0.69. **The cold-short rung stays at K=3**: the loss is
   reproducible on a quarter of the prompts, and the mean depends on a prompt mix chosen by
   hand. A static rung cannot take the gain without the loss.

Decision record: [`docs/decisions/SPEC-K-COLD-SHORT-5090.md`](../../docs/decisions/SPEC-K-COLD-SHORT-5090.md).

## Method

One RTX 5090 Laptop GPU, `memra-server` built from main 031f146690 for sm_120a, one binary in
every arm. `tools/pk_k9.py`, one server boot per call and per GPU lease, each boot started from
a card at 62 C or cooler; `tools/pk_k_summary.py` reads the rows. Requests one at a time,
400-token answers, context 8,192, thinking as the template defaults. Samplers: `t1` temperature
1 with a fixed seed, `chat` temperature 0.7 and top_p 0.95 with a fixed seed, `greedy`. Rates
are decode tok/s after the first chunk; ratios are on medians of per-boot values; "wins" counts
the boot pairs the second arm is ahead in. Texts are equal across arms on every prompt (a seeded
request writes one text at every draft length since memra#1154), so a per-prompt ratio compares
the same tokens at two depths.

Thermal regime: every boot is throttled. 9B cell: 75 to 78 C median, 82 to 86 C maximum, busy
SM clock 1,807 to 1,845 MHz at 153 to 156 W, except boot 1 (the model-owned arm) at 1,777 MHz
and 144 W. 27B cells: 81 to 84 C median, 86 to 90 C maximum, 1,605 to 1,672 MHz, 150 to 157 W
(`telemetry-*.csv`, 250 ms; `tools/pk_thermal.py`). The throttling is not equal across arms in
the twelve-prompt cell: boot 8 (K=3) spends 90 of its 306 busy samples in thermal slowdown and
its last five prompts run 1 to 15% under the other K=3 boots; over the cell, thermal slowdown
covers 154 of 1,484 busy samples at K=3 and 48 of 1,743 at K=2. The medians absorb boot 8, and
the result without it is given beside the table.

## 1. Qwen3.5-9B: the cost model owns K (`q9-owned-k/`)

Qwen3.5-9B NVFP4 MTP GGUF with its own-trim drafter, four sessions (the hit gate's shape). The
boot calibration fits, so the cost model picks the draft length inside the table's rung. Arms:
`own` (the model picks), `k2` and `k3` (`MEMRA_SPEC_K` pinned). 12 boots, own k2 k3 k3 k2 own
x2, four per arm; nine cold prompts per boot (34, 1.2k and 3.6k tokens per sampler); 108
requests, 0 errors.

| class | k3 tok/s | k2 tok/s | k2 over k3 | wins of 16 | own tok/s | K the model served | own over k2 |
|---|---:|---:|---:|---:|---:|---|---:|
| t1-short | 172.6 | 177.7 | 1.030 | 16 | 169.5 | 3 in 4 of 4 boots | 0.954 |
| t1-1k | 175.0 | 179.0 | 1.023 | 16 | 177.5 | 2 | 0.992 |
| t1-4k | 167.0 | 171.9 | 1.030 | 16 | 170.4 | 2 | 0.991 |
| chat-short | 162.6 | 172.8 | 1.063 | 16 | 170.5 | 2 | 0.987 |
| chat-1k | 175.4 | 183.1 | 1.044 | 16 | 181.8 | 2 | 0.993 |
| chat-4k | 168.7 | 169.3 | 1.003 | 13 | 168.1 | 2 | 0.993 |
| greedy-short | 178.7 | 189.5 | 1.060 | 16 | 188.3 | 2 | 0.994 |
| greedy-1k | 192.8 | 201.2 | 1.044 | 16 | 199.9 | 2 | 0.994 |
| greedy-4k | 186.2 | 192.3 | 1.033 | 16 | 191.5 | 2 | 0.996 |

- K=2 over K=3: geometric mean 1.036, nine of nine prompts ahead, 141 of 144 boot pairs.
  Acceptance per drafted token 0.46 to 0.53 at K=2, 0.34 to 0.41 at K=3.
- The model's choice: `t1-short` is the first request of every boot and runs at K=3
  (`[spec-cost] ... exploring: no acceptance history yet`, the table's short-prompt rung); the
  other eight run at K=2 in all four boots. That first request is 4.6% slower than at pinned
  K=2 (3.7% without boot 1, which ran at the lower clock).
- This is the difference the hit gate printed between a cold run and its hit
  (`tools/spec-on-cache-hit-gate.sh`, 64-token answers; its server log from the memra#1172
  work is `hit-gate/server.log`, built from that change's sources): the first five requests
  run at K=3; the model then prices K=2 at 170.3 tok/s, returns to K=3 at 173.0 (1.6% apart)
  and serves the sixth request, a cold run, at K=3; that run's hit is served at K=2, and K=2
  holds for the rest of the gate.
- One gap not explained: on the eight prompts it served at K=2, the model-owned arm is 0.8%
  under pinned K=2 (geometric mean 0.992, ahead in 3 of 128 boot pairs; 0.6% without boot 1).
  Its boots also run the calibration (132 to 140 busy samples against 84 to 94) and sit 1 to
  2 C hotter. Cause not determined.

## 2. Qwen3.8-27B: the table owns K

Qwen3.8-27B NVFP4 + Q5_K GGUF with its MTP head, two sessions. On this card the boot
calibration does not fit (`[spec-cost] calibration skipped: ... 16 sessions at depth 2048 do not
fit`), so the table decides: K=3 under 1,024 prompt tokens, K=2 above. Arms: `s3` (the default)
and `s2` (`MEMRA_SPEC_K_COLD_SHORT=2`).

**Nine prompts** (`q27-nine-prompts/`, 12 boots s3 s2 s2 s3 x3, six per arm, 108 requests, 0
errors). The six long prompts run at K=2 in both arms and are the control.

| class | s3 tok/s (K) | s2 tok/s (K) | s2 over s3 | wins of 36 | acceptance s3 | s2 |
|---|---:|---:|---:|---:|---:|---:|
| t1-short | 70.98 (3) | 70.62 (2) | 0.995 | 6 | 0.457 | 0.563 |
| chat-short | 63.82 (3) | 67.05 (2) | 1.051 | 36 | 0.400 | 0.540 |
| greedy-short | 66.93 (3) | 71.22 (2) | 1.064 | 36 | 0.382 | 0.523 |
| t1-1k, t1-4k | 70.62, 65.71 (2) | 70.66, 65.69 (2) | 1.000, 1.000 | 23.5, 19 | 0.590, 0.521 | same |
| chat-1k, chat-4k | 71.59, 66.34 (2) | 71.46, 66.24 (2) | 0.998, 0.999 | 12, 14.5 | 0.625, 0.560 | same |
| greedy-1k, greedy-4k | 74.77, 71.36 (2) | 74.67, 70.75 (2) | 0.999, 0.991 | 18, 15 | 0.584, 0.564 | same |

**Twelve short prompts** (`q27-twelve-short/`, `PROMPT_SET=short`: twelve tasks of different
kinds, 26 to 38 prompt tokens, four per sampler; 12 boots s3 s2 s2 s3 x3). Boot 12 is void
(`VOID.txt`: the machine left its performance power mode during it), so the table reads
`valid/runs.jsonl` (`tools/pk_k_summary.py q27-twelve-short/valid s3 s2`): five boots at K=3,
six at K=2, 132 requests, 0 errors.

| prompt | s3 tok/s | s2 tok/s | s2 over s3 | wins of 30 | acceptance at K=3 | at K=2 |
|---|---:|---:|---:|---:|---:|---:|
| t1-s1 | 67.31 | 69.35 | 1.030 | 30 | 0.417 | 0.540 |
| chat-s1 | 61.47 | 65.60 | 1.067 | 30 | 0.366 | 0.503 |
| greedy-s1 | 86.38 | 86.63 | 1.003 | 28 | 0.577 | 0.722 |
| t1-s2 | 69.66 | 70.28 | 1.009 | 30 | 0.453 | 0.569 |
| chat-s2 | 68.52 | 70.34 | 1.027 | 25 | 0.463 | 0.587 |
| greedy-s2 | 70.20 | 73.30 | 1.044 | 30 | 0.413 | 0.550 |
| t1-s3 | 86.72 | 84.21 | 0.971 | 0 | 0.690 | 0.820 |
| chat-s3 | 72.62 | 73.52 | 1.012 | 30 | 0.515 | 0.652 |
| greedy-s3 | 81.74 | 79.59 | 0.974 | 6 | 0.535 | 0.638 |
| t1-s4 | 80.09 | 78.44 | 0.979 | 6 | 0.609 | 0.731 |
| chat-s4 | 73.13 | 73.62 | 1.007 | 27 | 0.533 | 0.665 |
| greedy-s4 | 68.01 | 71.53 | 1.052 | 29 | 0.405 | 0.531 |

- All twelve: geometric mean 1.014, nine prompts ahead, 271 of 360 boot pairs.
- By sampler: `chat` 1.028 (four of four ahead), `greedy` 1.018 (three of four), `t1` 0.997
  (two of four).
- Three prompts are 2 to 3% behind, with 0, 6 and 6 of 30 boot pairs. All twelve of those
  pairs are against boot 8.
- Without boot 8 (four boots at K=3, six at K=2): geometric mean 1.013, nine ahead, 205 of 288
  boot pairs; the three prompts behind lose every pair (0.970, 0.974, 0.978).

**It follows acceptance, not the sampler.** The fifteen short prompts of the two cells, by the
acceptance per drafted token K=3 gets on them:

| K=3 acceptance | prompts | s2 over s3 |
|---|---:|---|
| 0.37 to 0.42 | 6 | 1.030 to 1.067 |
| 0.45 to 0.46 | 3 | 0.995, 1.009, 1.027 |
| 0.52 to 0.58 | 4 | 0.974, 1.003, 1.007, 1.012 |
| 0.61 to 0.69 | 2 | 0.979, 0.971 |

Pearson r = -0.88, a slope of -3.0% per 0.10 of acceptance; the fitted line crosses 1.0 at
0.54. The three prompts behind sit at 0.535, 0.609 and 0.690; two at 0.533 and 0.577 are 0.3
to 0.7% ahead.
The first cell's `t1-short` (0.995) sits at 0.457, at the edge of the band where the two
depths meet; it is neither noise (6 of 36 boot pairs) nor a property of temperature 1 (two
other `t1` prompts gain 1 to 3%, at 0.42 and 0.45).

## What was not run

- Two requests at once on the short prompts, and the Qwen3.5 MoE class on this card: both were
  queued as conditions of a flip, and there is no flip. The two-at-once cell's first boot
  started after the power mode changed and was stopped; nothing of it is kept.
- A second draft length below 2 or above 3.
- Any other card.
- Short prompts with a cached prefix (a resumed turn).

Before a flip of this rung, the evidence would need: a short-prompt set whose acceptance
distribution is the served traffic's, not a hand-picked dozen; the two-at-once and MoE cells;
cached short turns; and the cost model's own pick on the 27B once its calibration fits.

## Follow-ups

- The 27B on this card is table-owned only because the boot calibration does not fit at its
  16-session depth (`[spec-cost] calibration skipped: ... 16 sessions at depth 2048 do not
  fit: the other 14 need 7081MB (505.8MB each), 7032MB allocatable`). If it calibrated at the
  depth and session count that fit, the cost model would own K there as it does on the 9B.
  What it would then pick is not shown here. The model keeps one recent acceptance history per
  model, not per request, and the 9B cell only shows it settling on the depth that wins on
  every prompt. On a mixed workload it may settle on one depth, which is the flip rejected
  above; a per-request choice would need an acceptance estimate per request or per session.
- The 0.8% gap between the model-owned arm and pinned K=2 on the 9B.
