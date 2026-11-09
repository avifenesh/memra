# The spec cost model on cold long prompts, RTX PRO 6000 Max-Q (memra#1093, 2026-11-09)

Verdict: the cost model picked K=3 for cold long prompts on the RTX PRO 6000 Max-Q because it
charged one K=3 burst's drafting to every draft length. With a draft step fitted at boot it serves
K=2 on 38 of 50 cold-long requests after a boot's first (0 of 74 before) and decodes T=1 prompts
of 1.2k and 3.6k tokens 5.5 to 7.2% faster at the same rung. It still returns to K=3 on about one
request in four, because a served regime is priced from a few bursts; a candidate that sampled
round rates instead was flat on the picks and is deleted. With the cold-long rung at 2
(`docs/decisions/SPEC-K-COLD-LONG-K2.md`) the final binary serves K=2 on every cold-long request.

Model and rig: Qwen3.8-27B NVFP4 + Q5_K GGUF with its untrimmed MTP head, context 8,192, two
sessions (four in the four-at-once cell), every switch at the device default. One RTX PRO 6000
Blackwell Max-Q of a four-card box, power limit at its 250 W, one boot per `tools/memra-gpu-run`
lease (the card's lock and the box's), pinned to eight cores and their siblings; another tenant
trained at a steady load on two other cards. Binaries are built on the box (`pro6000-maxq/build3.sh`,
`build4.sh`). The harness is `research/spec-k-cold-long-20261109/`'s: `cold_long_ab.py` serves the
same cold prompts in every boot, one at a time, answers capped at 400 tokens, thinking off;
`ab_bins.py` drives one arm per boot; `analyze.py` prints the tables below.

## Cause 1: one drafting cost for every K (fixed)

The boot calibration (`[spec-cost]` line) measured the per-round cost beyond the verify forward
once, from a K=3 burst, and `spec_round_ms` charged it to every draft length. A K=2 round paid
three draft steps, so K=3's extra expected tokens looked free: 4.39 ms flat priced K=3 at 117.6
tok/s against K=2's 114.0 at this model's acceptance (expected tokens 2.41 and 2.20 per round).

The calibration now bursts at K=1 and K=3 and fits a fixed part plus one draft step
(`split_round_overhead`); a round at draft length k pays k steps. Over this card's boots the fit
is 0.35 to 1.08 ms fixed and 1.09 to 1.39 ms per step (21 boots). The priced K=3 round is then
2.6 ms longer than the K=2 round; the served rounds are 2.6 to 3.1 ms apart (table in "Per
prompt"). The two
extra bursts and their sessions lengthen the calibration from 19.5-20.9 s to 26.5-28.2 s per boot.

First check (`pro6000-maxq/run-bins.sh`): `base` e0447ca00e against `fix`, the same commit plus
the draft-step fit (8982282934), both at the device default (cold-long rung 3).

K picked on cold-long requests (the first request of a boot explores at the rung, before the
model has acceptance history):

| set | arm | boots | first request | later requests at K=2 | at K=3 |
|---|---|---:|---|---:|---:|
| long (`pro6000-maxq/long/`) | base | 4 | K=3 x4 | 0 | 24 |
| long | fix | 4 | K=3 x4 | 18 | 6 |
| 1.2k (`pro6000-maxq/base/`) | base | 2 | K=3 x2 | 0 | 14 |
| 1.2k | fix | 2 | K=3 x2 | 12 | 2 |

Decode rate, each boot's median, fix over base (U = boot pairs the fix wins):

| set | class | base (tok/s) | fix (tok/s) | ratio | U | acceptance base / fix |
|---|---|---:|---:|---:|---:|---|
| long | T=1 3.6k | 97.03 | 101.21 | 1.055 | 13/16 | 0.440 / 0.585 |
| long | T=1 5.8k | 99.41 | 97.78 | 0.995 | 3/16 | 0.483 / 0.529 |
| long | greedy 3.6k | 95.30 | 99.91 | 1.048 | 12/16 | 0.380 / 0.455 |
| long | T=1 short | 97.17 | 97.09 | 1.005 | 9/16 | 0.435 / 0.475 |
| 1.2k | T=1 1.2k | 99.05 | 105.17 | 1.058 | 4/4 | 0.463 / 0.558 |
| 1.2k | T=1 3.6k | 98.97 | 105.74 | 1.072 | 4/4 | 0.472 / 0.647 |
| 1.2k | greedy 1.2k | 108.09 | 110.28 | 1.003 | 2/4 | 0.470 / 0.574 |
| 1.2k | T=1 short | 97.05 | 97.39 | 1.003 | 3/4 | 0.435 / 0.519 |

(The per-request medians are in `analyze.py`'s output; the ratio column is on per-boot medians.)
TTFT is the same on both arms (1.390 against 1.392 s at 3.6k). Greedy texts are one sha per prompt
across arms. The 5.8k pair is flat because the fix served one of its two prompts at K=3 in every
boot.

## Cause 2: realized rates decide between draft lengths on a few bursts (not fixed)

Once a (sessions, K) regime has three ticks, `SpecCostEntry::priced` replaces its table price with
its realized rate: an EMA, weight 0.3, of per-tick aggregate tok/s. One tick of one speculative
session is one 32-token burst. A burst's tokens per round on cold-long requests
(`burst_spread.py pro6000-maxq/long pro6000-maxq/base pro6000-maxq/final-long`):

| K | bursts | tokens per round, min to max | mean | standard deviation over mean |
|---|---:|---|---:|---:|
| 3 | 1031 | 1.72 to 3.88 | 2.40 | 15.1% |
| 2 | 713 | 1.72 to 3.00 | 2.22 | 11.4% |

and an EMA of weight 0.3 keeps 42% of that. K=2 leads K=3 by about 5%. So the served depth's
price wanders across the stale price of the depth not being served, and the pick moves between
admissions: in the fix boots the printed prices of the chosen depth run 91.4 to 112.7 tok/s
(`grep 'rung=3 ->' pro6000-maxq/long/server-*-fix.log`).

**Candidate, rejected: round-rate samples** (`round-rate-sample.patch.gz`, built as 9259d4f451 on
main 4568606267 with the draft-step fit and the rung change). A speculative tick sampled its round
rate times the acceptance window's expected tokens per round at that K, instead of the tokens the
burst emitted. Arms (`pro6000-maxq/run-flip.sh`; the directories ran as `flip-long` and `flip-c4`):
`base` e0447ca00e; `rung3`, the candidate with `MEMRA_SPEC_K_COLD_LONG=3`, so the model prices
K=1..3 as on the base; `flip`, the candidate at its default (cold-long rung 2).

| arm | boots | first request | later cold-long requests at K=2 | at K=3 |
|---|---:|---|---:|---:|
| base | 3 | K=3 x3 | 0 | 18 |
| rung3 | 3 | K=3 x3 | 14 | 4 |
| flip | 3 | K=2 x3 | 18 | 0 |

- The prices did hold still: after the first, extrapolated K=2 price (109.9 to 113.4 tok/s) the
  chosen depth's price runs 96.1 to 100.0, and consecutive verdicts sit within 1 tok/s of each
  other (100.0, 99.8, 100.0; 96.1, 96.8; 97.1, 96.5:
  `grep 'rung=3 ->' pro6000-maxq/sample-long/server-*-rung3.log`).
- But they tied, where the K=2 requests of the same boots decoded at 99.9 to 107.1 tok/s (median
  101.95) and the K=3 requests at 94.3 to 103.2 (median 96.4). The window pools the per-position
  acceptance of every round it holds, whatever draft length and prompt it came from, and the third
  position's share comes only from the few K=3 requests in it.
- Picks: 14 of 18 at K=2, against 18 of 24 on the same set without the candidate. Flat. The code
  is deleted; the patch and these receipts stay.
- One request of the flip arm (boot 3, `volcanoes-6k`) took 14.3 s for 400 tokens with its chunk
  gaps and acceptance normal: the card's utilization sat under 20% for 7.8 s in the middle of it
  (`pro6000-maxq/sample-long/telemetry-3-flip.csv`, 22:33:53 to 22:34:00). Cause unknown. The
  per-boot medians do not move with it.

## Per prompt: K=2's lead is an average

`per_prompt.py` over the fixed-K A/B of `research/spec-k-cold-long-20261109/pro6000-maxq/` (6
boots per arm, one K per boot). The last column is the third draft position's acceptance if the
first two positions accept alike at both depths: 3 x acceptance(K=3) - 2 x acceptance(K=2).

| prompt | prompt tokens | K=3 decode (tok/s) | K=2 decode (tok/s) | K=2 over K=3 | acceptance K=3 / K=2 | round K=3 / K=2 (ms) | implied third-position acceptance |
|---|---:|---:|---:|---:|---|---|---:|
| rivers-4k | 3565 | 97.7 | 100.2 | 1.026 | 0.451 / 0.566 | 24.1 / 21.3 | 0.22 |
| bridges-4k | 3566 | 95.4 | 105.0 | 1.100 | 0.430 / 0.621 | 24.0 / 21.4 | 0.05 |
| glaciers-4k | 3566 | 94.8 | 101.2 | 1.067 | 0.425 / 0.578 | 24.0 / 21.3 | 0.12 |
| harbors-4k | 3697 | 101.9 | 102.6 | 1.007 | 0.487 / 0.593 | 24.2 / 21.3 | 0.28 |
| volcanoes-6k | 5727 | 101.9 | 101.5 | 0.996 | 0.510 / 0.602 | 24.8 / 21.7 | 0.32 |
| canyons-6k | 5928 | 95.6 | 103.7 | 1.085 | 0.455 / 0.637 | 24.8 / 21.9 | 0.09 |
| islands-4k (greedy) | 3565 | 96.1 | 105.3 | 1.096 | 0.380 / 0.531 | 22.3 / 19.6 | 0.08 |
| rivers-1k | 1239 | 106.4 | 105.0 | 0.987 | 0.514 / 0.599 | 23.9 / 20.9 | 0.34 |
| bridges-1k | 1240 | 99.4 | 109.7 | 1.104 | 0.455 / 0.652 | 23.8 / 21.0 | 0.06 |
| glaciers-1k | 1240 | 100.5 | 105.2 | 1.047 | 0.462 / 0.602 | 23.7 / 21.0 | 0.18 |
| harbors-1k | 1286 | 100.9 | 104.0 | 1.031 | 0.463 / 0.593 | 23.7 / 21.0 | 0.20 |
| volcanoes-4k | 3697 | 101.1 | 104.2 | 1.030 | 0.483 / 0.615 | 24.2 / 21.4 | 0.22 |
| canyons-4k | 3828 | 98.1 | 109.9 | 1.121 | 0.460 / 0.680 | 24.3 / 21.5 | 0.02 |
| islands-1k (greedy) | 1239 | 111.1 | 112.8 | 1.015 | 0.479 / 0.584 | 21.9 / 19.2 | 0.27 |
| meadows-1k (greedy) | 1286 | 109.2 | 111.0 | 1.016 | 0.460 / 0.564 | 21.8 / 19.2 | 0.25 |

K=2 leads by 8 to 12% where the third position is accepted least (0.02 to 0.09) and is within 1.5%
either way where it is accepted most (0.27 to 0.34). The model's own rule says the same: a third
position pays for its 2.6 ms once its acceptance passes about 0.3. An online choice between two
depths 5% apart on average, from one request each, does not settle on this spread; a 12-boot A/B
does. That is the case for carrying the default in the rung and leaving the model the choices
below it.

## Not the cause: the calibration depth

The table is measured at a depth of 2,048 tokens and these prompts run at 1,200 to 6,200. That was
the first suspect. The served K=3 round is 2.6 to 3.0 ms longer than the K=2 round at 1.2k and
3.6k and 2.8 to 3.1 ms at 5.8k; the table prices 2.6 ms. And a served regime is priced from serving, not from the
table.

## The final binary (`pro6000-maxq/final-long/`, `final-c4/`)

7eb0ee7d98: main 4568606267 with the draft-step fit and the cold-long rung at 2, both keyed there
on the device class (the `120a` build family without unified memory). The flip PR keys them on
the two measured cards' names instead, which is the same on this card; its other later changes
are comments, tests and docs. Arms as above (`pro6000-maxq/run-final.sh`).

| arm | boots | first request | later cold-long requests at K=2 | at K=3 |
|---|---:|---|---:|---:|
| base | 3 | K=3 x3 | 0 | 18 |
| rung3 | 2 | K=3 x2 | 8 | 4 |
| flip | 3 | K=2 x3 | 18 | 0 |

| class | base (tok/s) | flip (tok/s) | flip over base | U | rung3 over base | U |
|---|---:|---:|---:|---:|---:|---:|
| T=1 3.6k | 95.69 | 100.11 | 1.060 | 9/9 | 1.086 | 6/6 |
| T=1 5.8k | 97.40 | 100.84 | 1.042 | 9/9 | 1.019 | 3/6 |
| greedy 3.6k | 94.30 | 103.87 | 1.101 | 9/9 | 1.017 | 3/6 |
| T=1 short | 96.60 | 97.03 | 1.010 | 6/9 | 1.017 | 4/6 |

- **The flip arm ran K=2 on all 27 requests**, the short ones included: the model prices a draft
  length only while its acceptance window holds rounds that offered that length's last position,
  and a boot that never ran K=3 has none for the third.
- **The base is not the flip's parent.** It is the first check's e0447ca00e, and main moved
  between the two: requests that ran K=3 on both binaries decode 1.7 to 2.9% faster on the new
  one (`rivers-4k` 95.91 against 98.69 tok/s, the two short prompts 98.07 and 94.28 against 99.77
  and 95.88) and their first token at 3.6k comes 60 to 75 ms earlier (1.394 against 1.320 s). So
  the ratios to base above overstate the rung by about 2 points. The clean K=2 against K=3 ratios
  are the fixed-K A/B's (`research/spec-k-cold-long-20261109/`). What this run adds is the picks.
- Greedy text: one sha (`b6f8668e30e210c7`) on every arm.

Four requests at once (`c4_ab.py`: per boot one group of four T=1 prompts of 3.6k tokens, one of
1.2k and one of 34 tokens, the four sent together; 2 boots per arm, order base flip flip base):

| binary | group | base boots: K, aggregate (tok/s) | flip boots: K, aggregate (tok/s) |
|---|---|---|---|
| candidate (`sample-c4/`) | 3.6k | 3 then plain: 106.81, 109.71 | 2 then plain: 112.32, 112.42 |
| candidate | 1.2k | plain: 154.13, 157.50 | plain: 156.96, 157.26 |
| candidate | short | plain: 191.38, 195.09 | plain: 196.91, 195.68 |
| final (`final-c4/`) | 3.6k | 3 then plain: 47.94 (boot 1), 107.83 | 2 then plain: 112.82, 112.23 |
| final | 1.2k | plain: 159.68, 156.75 | K=1: 153.24, 151.68 |
| final | short | plain: 199.72, 196.53 | plain: 196.88, 195.28 |

- At four sessions the model explores speculation on the first group, prices it under plain
  within about 100 to 135 tokens and demotes all four sessions to batched decode
  (`[spec-gate] demoted session to batched decode: 4 active, cost model prefers plain`), on every
  binary. The 3.6k group gains about 4% on the flip binaries from its K=2 opening.
- **Final binary, 1.2k group: K=1 instead of plain, 3.6% less aggregate** (152.5 against 158.2
  tok/s; per request 42 to 48 against 47 to 51 tok/s, first tokens at 0.5 to 2.1 s against 1.7 to
  2.2 s). After the K=2 opening the spec price at four sessions sits 2.5% under plain (190.1
  against 194.9 tok/s) where the K=3 opening left it 13 to 19% under, so the gate's tie probe
  speculates once more to measure the slot (`n=4 rung=2 -> K=1 (predicted 190.1 vs plain 194.9)`),
  then serves plain. Two boots, one group: a probe's cost, not a settled result.
- **Base boot 1 of the final cell is an outlier:** its 3.6k group took 33.4 s where boot 4 and
  both base boots of the candidate cell took 14.6 to 15.0 s, at the same K and the same texts.
  Cause unknown; nothing is concluded from it.

## Scope

- One card, one model. The fit is on for this card only. GB10 keeps its calibration as it was
  (one K=3 burst): its model
  picks K=3 for cold long prompts, and its own cell has K=3 ahead of K=2 by 1 to 3%
  (`research/gb10-decode-20261109/kcl-unit1/`, with memra#1145). For this model the RTX 5090
  Laptop skips the calibration (16 sessions at depth 2,048 do not fit beside it in 24 GB), so its
  table rung serves and the fit does not reach it. A smaller model that calibrates there keeps
  the older pricing.
- Two and three sessions on this card are not measured. The four-session cell is 2 boots per arm.
- Cause 2 stands. A fix needs each draft length priced from its own rounds over many requests and
  a margin before the served depth changes; with the rung at 2 its remaining reach on this card is
  K=2 against K=1 and plain.
