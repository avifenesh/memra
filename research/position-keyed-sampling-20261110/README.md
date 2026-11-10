# Position-keyed sampled verify on the MTP route (memra#1154, 2026-11-10)

Verdict: with the verify walk keyed by position, one seeded sampled request writes one text at
K=0, 1, 2 and 3, alone or beside another request, cold or resumed, and across a demotion from
speculation to plain decode. On main every K writes another text and a pair differs from its
solo runs. The price is acceptance: about 2 points per drafted token on average over the eleven
estimates here, from 1 point better to 6 worse by cell. In the 12-boot decode cells, sampled
decode against main over six prompts is 0.999 (prototype) and 1.039 (landed form) on the RTX 5090
with the greedy control at 0.987 and 1.009, and 0.972 on GB10 in both forms with the control at
1.000. Greedy is unchanged (same texts, same acceptance).

Model: Qwen3.8-27B NVFP4 + Q5_K GGUF with its MTP head, context 8,192, two sessions, thinking
off, every switch at the device default unless a cell pins `MEMRA_SPEC_K`. Devices: an RTX 5090
Laptop GPU (sm_120a) and GB10, one DGX Spark (sm_121a). One boot per GPU lease in the timed
cells, 250 ms `nvidia-smi` telemetry per boot.

The rule and the decision: [`docs/decisions/SAMPLED-VERIFY-POSITION-KEYED.md`](../../docs/decisions/SAMPLED-VERIFY-POSITION-KEYED.md).

## Two rounds of measurement

- `prototype/`: main 53eb765421 against the first form of the change (`cand`: the coupled walk
  with one readback per position; `cand2`: plus plain decode's first token on the event-0 device
  draw). This is the measurement the decision was taken on.
- `pr/`: main ae69d86b05 against the change as it lands (`pr`: all K+1 draws issued before one
  readback, per-position penalty windows, the penalised plain first token), and the serving gate
  `tools/seeded-invariance-gate.py`. The two decode cells here ran the build of that walk from
  before the review's fixes (where plain decode's first token takes the device draw, the
  parked-prompt rule, the constrained first token, the trimmed head's id map); none of those
  touches a sampled round on an untrimmed head. The gate rows of this change, the trimmed-head
  rows marked final code and the hit gate ran the final code.

Cell programs are in `tools/` here: `pk_ab.py` (MODE=perf: nine cold prompts per boot, 34, 1.2k
and 3.6k tokens at temperature 1, at temperature 0.7 with top_p 0.95, and greedy, 400-token
answers; MODE=inv: the invariance cell), `pk_ab2.py` (adds the repeated request and MODE=acc,
16 prompts by two samplers with 200-token answers), `pk_summary.py`, `pk_pool.py`, `pk_acc.py`,
`pk_thermal.py`. Receipts are the per-request rows (`runs.jsonl`, with a sha of each text), the
server logs and the telemetry; the texts themselves are not kept.

## The invariance cell

Six prompts, each with a fixed seed at temperature 1: a 1.2k-token one alone (a), a 34-token one
alone (b), two 1.2k-token ones sent at once and each alone (c, d), and a two-turn conversation
whose second turn (e2) is sent resumed and cold. `MEMRA_SPEC_K` pinned to 0 (plain decode), 1, 2
and 3; one boot per (binary, K, part).

RTX 5090, prototype (`prototype/rtx5090/inv/runs.jsonl`):

| binary | a | b | c | d | e1 | e2 |
|---|---|---|---|---|---|---|
| main | 4 texts in 4 runs | 4 in 4 | 7 in 8 | 7 in 8 | 4 in 4 | 4 in 8 |
| candidate (`cand` at K=1, 2, 3; `cand2` at K=0, 1) | 1 in 5 | 1 in 5 | 1 in 10 | 1 in 10 | 1 in 5 | 1 in 10 |

- On main every K writes another text, and the pair differs from the solo runs at K=1, 2 and 3.
  Cold equals resumed inside a K.
- `cand` at K=0 differed from K>=1 on b, d and e2, from the first token: plain decode drew its
  first token from the host sampler's RNG, not from Philox event 0. `cand2` moves it.
- The same request sent twice in one boot (the second finds its whole prompt cached) writes the
  same text on `cand2` at K=0 and K=2.

At the landed form the cell is the gate:

`tools/seeded-invariance-gate.py`: a plain reference boot (`MEMRA_SPEC_K=0`, every prompt alone
and cold), then `MEMRA_SPEC_K` pinned 0, 1, 2, 3 (the lone prompts, the pair, the resumed turn)
and a forced demotion (`MEMRA_SPEC_DEMOTE_AT=60`, no pin). Eight prompts: a and b as above, f
(1.2k tokens, temperature 0.7, top_p 0.95), g (34 tokens, frequency penalty 0.4), the pair c and
d, the conversation e1 and e2. 240-token answers, 44 rows. Texts per prompt over its runs:

| device, binary | a | b | f | g | c | d | e1 | e2 | p1, p2 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| RTX 5090, main ae69d86b05 (`pr/rtx5090/gate-main.log`) | 5 in 6 | 5 in 6 | 5 in 6 | 5 in 6 | 4 in 5 | 4 in 5 | 4 in 5 | 4 in 5 | not run | FAIL, 28 of 44 rows |
| RTX 5090, this change (`pr/rtx5090/gate.log`) | 1 in 6 | 1 in 6 | 1 in 6 | 1 in 6 | 1 in 5 | 1 in 5 | 1 in 5 | 1 in 5 | 1 in 5 each | PASS, 54 rows |
| GB10, this change, `--penalised host` (`pr/gb10/gate.log`) | 1 in 6 | 1 in 6 | 1 in 6 | 3 in 6 | 1 in 5 | 1 in 5 | 1 in 5 | 1 in 5 | 1 in 5, 1 in 5 | PASS, 54 rows |

p1 and p2 are the parked-prompt pair, added to the gate after the first runs: a text completion
p1 parks, and p2's prompt is p1's prompt plus its answer, so the session p1 leaves holds exactly
p2's prompt. p2 writes its cold text at K=0, 1, 2 and 3 (192 tokens restored each time). Main's row ran
the gate before that pair existed (`pr/rtx5090/gate-first.log` is this change's run of that
form). The two rows of this change are at the final code.

- On main only the K=0 boot repeats the reference; K=1, 2, 3 and the demoted run each write
  another text.
- Demotion: all four demoted requests log `generated 64, sampled, plain draws from philox event
  65` (66 and 67 for the penalised one) and write the reference text on the RTX 5090. The
  session counter is the emitted count, so the handoff base is 0.
- The penalised prompt g: on the RTX 5090 plain decode draws it on the device (the dense Qwen35
  class has the device penalty arm on) and it writes one text everywhere. On GB10 the arm is
  off: plain decode draws g on the host, speculation writes one other text at K=1, 2 and 3, and
  the demoted run a third (speculative prefix, host-drawn tail). That is the class the decision
  record lists as not covered, and what `--penalised host` tells the gate.

## Decode rate and acceptance

Twelve boots per device, base cand cand base three times, nine cold prompts per boot, both arms
at the device's own K. Ratios are candidate over base on per-boot medians; U counts the boot
pairs (of 36) the candidate wins; acceptance is per drafted token.

### Prototype, RTX 5090 (`prototype/rtx5090/perf/`; main 53eb765421 against `cand2`; 108 requests, 0 errors; both arms K=3)

| class | decode ratio (cand over base, medians) | U | acceptance base | acceptance cand | tokens per round base | cand |
|---|---:|---:|---:|---:|---:|---:|
| t1-short | 1.084 | 36/36 | 0.397 | 0.457 | 2.19 | 2.37 |
| t1-1k | 0.997 | 15/36 | 0.475 | 0.455 | 2.42 | 2.37 |
| t1-4k | 0.897 | 0/36 | 0.506 | 0.414 | 2.52 | 2.24 |
| chat-short | 0.980 | 14/36 | 0.423 | 0.400 | 2.27 | 2.20 |
| chat-1k | 1.048 | 25/36 | 0.455 | 0.478 | 2.36 | 2.43 |
| chat-4k | 0.996 | 17/36 | 0.475 | 0.474 | 2.42 | 2.42 |
| greedy-short | 0.978 | 13/36 | 0.382 | 0.382 | 2.15 | 2.15 |
| greedy-1k | 0.995 | 14/36 | 0.473 | 0.473 | 2.42 | 2.42 |
| greedy-4k | 0.987 | 9/36 | 0.424 | 0.424 | 2.27 | 2.27 |

- T=1: 3 prompts, decode ratio geometric mean 0.990 (per prompt 1.084, 0.997, 0.897), acceptance 0.460 -> 0.442
- T=0.7 top-p 0.95: 3 prompts, decode ratio geometric mean 1.008 (per prompt 0.980, 1.048, 0.996), acceptance 0.451 -> 0.451
- both sampled: 6 prompts, decode ratio geometric mean 0.999 (per prompt 1.084, 0.997, 0.897, 0.980, 1.048, 0.996), acceptance 0.455 -> 0.446
- greedy control: 3 prompts, decode ratio geometric mean 0.987 (per prompt 0.978, 0.995, 0.987), acceptance 0.426 -> 0.426

Boot 1, a base boot, ran at an SM clock of 1,635 MHz after the cool start; the others ran at 1,190 to 1,420 MHz under the thermal throttle (`pk_thermal.py`).

### Prototype, GB10 (`prototype/gb10/perf/`; 108 requests, 0 errors; both arms K=3; decode range inside a class +-0.3%)

| class | decode ratio (cand over base, medians) | U | acceptance base | acceptance cand | tokens per round base | cand |
|---|---:|---:|---:|---:|---:|---:|
| t1-short | 0.911 | 0/36 | 0.470 | 0.391 | 2.41 | 2.17 |
| t1-1k | 0.997 | 3/36 | 0.472 | 0.458 | 2.42 | 2.37 |
| t1-4k | 0.972 | 0/36 | 0.472 | 0.441 | 2.42 | 2.32 |
| chat-short | 0.992 | 0/36 | 0.423 | 0.414 | 2.27 | 2.24 |
| chat-1k | 0.953 | 0/36 | 0.521 | 0.477 | 2.56 | 2.43 |
| chat-4k | 1.008 | 36/36 | 0.493 | 0.494 | 2.48 | 2.48 |
| greedy-short | 1.001 | 23/36 | 0.384 | 0.384 | 2.15 | 2.15 |
| greedy-1k | 0.997 | 9.5/36 | 0.451 | 0.451 | 2.35 | 2.35 |
| greedy-4k | 1.002 | 21/36 | 0.450 | 0.450 | 2.35 | 2.35 |

- T=1: 3 prompts, decode ratio geometric mean 0.959 (per prompt 0.911, 0.997, 0.972), acceptance 0.471 -> 0.430
- T=0.7 top-p 0.95: 3 prompts, decode ratio geometric mean 0.984 (per prompt 0.992, 0.953, 1.008), acceptance 0.479 -> 0.462
- both sampled: 6 prompts, decode ratio geometric mean 0.972 (per prompt 0.911, 0.997, 0.972, 0.992, 0.953, 1.008), acceptance 0.475 -> 0.446
- greedy control: 3 prompts, decode ratio geometric mean 1.000 (per prompt 1.001, 0.997, 1.002), acceptance 0.428 -> 0.428

### The landed form, GB10 (`pr/gb10/perf/`; main ae69d86b05 against `pr`; 108 requests, 0 errors; both arms K=3)

| class | decode ratio (cand over base, medians) | U | acceptance base | acceptance cand | tokens per round base | cand |
|---|---:|---:|---:|---:|---:|---:|
| t1-short | 0.911 | 0/36 | 0.470 | 0.391 | 2.41 | 2.17 |
| t1-1k | 0.998 | 7/36 | 0.472 | 0.458 | 2.42 | 2.37 |
| t1-4k | 0.972 | 0/36 | 0.472 | 0.441 | 2.42 | 2.32 |
| chat-short | 0.991 | 0.5/36 | 0.423 | 0.414 | 2.27 | 2.24 |
| chat-1k | 0.953 | 0/36 | 0.521 | 0.477 | 2.56 | 2.43 |
| chat-4k | 1.010 | 32/36 | 0.493 | 0.494 | 2.48 | 2.48 |
| greedy-short | 1.000 | 19.5/36 | 0.384 | 0.384 | 2.15 | 2.15 |
| greedy-1k | 1.000 | 16/36 | 0.451 | 0.451 | 2.35 | 2.35 |
| greedy-4k | 1.003 | 28.5/36 | 0.450 | 0.450 | 2.35 | 2.35 |

- T=1: 3 prompts, decode ratio geometric mean 0.959 (per prompt 0.911, 0.998, 0.972), acceptance 0.471 -> 0.430
- T=0.7 top-p 0.95: 3 prompts, decode ratio geometric mean 0.984 (per prompt 0.991, 0.953, 1.010), acceptance 0.479 -> 0.462
- both sampled: 6 prompts, decode ratio geometric mean 0.972 (per prompt 0.911, 0.998, 0.972, 0.991, 0.953, 1.010), acceptance 0.475 -> 0.446
- greedy control: 3 prompts, decode ratio geometric mean 1.001 (per prompt 1.000, 1.000, 1.003), acceptance 0.428 -> 0.428
- Per sampled round (1000 x tokens per round / decode rate): 109.1 ms on main and 107.5 on the candidate at 1.2k tokens and temperature 1, 0.2 to 1.6 ms shorter across the six sampled classes; greedy rounds are equal (99.5 and 99.5, 99.6 and 99.6, 102.4 and 102.1). The prototype's walk, with one readback per position, showed the same 0.4 to 1.6 ms, so the single readback bought nothing measurable here and the whole decode difference is acceptance.

### The landed form, RTX 5090

Twelve boots, main ae69d86b05 against `pr`, in two parts: boots 1 to 3 ran 04:48 to 05:16, the
cell then waited for another session's CPU run on the same cooler, and boots 4 to 12 resumed at
05:54 from a card at 62 C (`pr/rtx5090/perf/HOLD.txt`). 108 requests, 0 errors. On this card main
serves a cold long prompt at K=2 and a short one at K=3 (memra#1152), both arms alike: 1k K=2 x36, 4k K=2 x36, short K=3 x36.

| class | decode ratio (cand over base, medians) | U | acceptance base | acceptance cand | tokens per round base | cand |
|---|---:|---:|---:|---:|---:|---:|
| t1-short | 1.095 | 36/36 | 0.397 | 0.457 | 2.19 | 2.37 |
| t1-1k | 1.025 | 25/36 | 0.569 | 0.590 | 2.14 | 2.18 |
| t1-4k | 0.960 | 10/36 | 0.599 | 0.521 | 2.20 | 2.04 |
| chat-short | 1.014 | 20/36 | 0.423 | 0.400 | 2.27 | 2.20 |
| chat-1k | 1.101 | 29/36 | 0.546 | 0.625 | 2.09 | 2.25 |
| chat-4k | 1.045 | 28/36 | 0.537 | 0.560 | 2.07 | 2.12 |
| greedy-short | 1.027 | 25/36 | 0.382 | 0.382 | 2.15 | 2.15 |
| greedy-1k | 0.999 | 21/36 | 0.584 | 0.584 | 2.17 | 2.17 |
| greedy-4k | 1.001 | 19/36 | 0.564 | 0.564 | 2.13 | 2.13 |

- T=1: 3 prompts, decode ratio geometric mean 1.025 (per prompt 1.095, 1.025, 0.960), acceptance 0.522 -> 0.523
- T=0.7 top-p 0.95: 3 prompts, decode ratio geometric mean 1.053 (per prompt 1.014, 1.101, 1.045), acceptance 0.502 -> 0.528
- both sampled: 6 prompts, decode ratio geometric mean 1.039 (per prompt 1.095, 1.025, 0.960, 1.014, 1.101, 1.045), acceptance 0.512 -> 0.525
- greedy control: 3 prompts, decode ratio geometric mean 1.009 (per prompt 1.027, 0.999, 1.001), acceptance 0.510 -> 0.510
- Busy SM clock per boot (MHz): 1 prbase 1,560, 2 pr 1,597, 3 pr 1,635, 4 prbase 1,380, 5 prbase 1,417, 6 pr 1,414, 7 pr 1,425, 8 prbase 1,635, 9 prbase 1,410, 10 pr 1,402, 11 pr 1,387, 12 prbase 1,387.
- Greedy texts: the same three texts in both arms.

Each class is one prompt and one seed, and the two arms write different answers, so a per-prompt
ratio is mostly the text. Wider reads of the acceptance difference (prototype, RTX 5090):

- 16 prompts by two samplers, two boots per arm, 128 requests, K=3 (`prototype/rtx5090/acc/`):
  temperature 1, 0.438 on main and 0.442 on the candidate (paired difference +0.003 +- 0.016);
  temperature 0.7 with top_p 0.95, 0.467 and 0.450 (-0.018 +- 0.016). The candidate's two boots
  ran at a lower SM clock than main's (1,245 and 1,312 MHz against 1,365 and 1,372), so this
  cell's decode rates are not a comparison.
- The invariance cell's nine runs per K: K=1 0.729 and 0.708, K=2 0.587 and 0.563, K=3 0.458 and
  0.443. Tokens per round 1.73 and 1.71, 2.17 and 2.13, 2.37 and 2.33.
- The gate's eight prompts per K, 240-token answers, main's run against this change's
  (`pr/rtx5090/gate-main/`, `pr/rtx5090/gate-first/`): K=1 0.736 and 0.701 (paired difference
  -0.035 +- 0.015), K=2 0.599 and 0.541 (-0.058 +- 0.031), K=3 0.469 and 0.428 (-0.041 +- 0.028).
  This is the largest difference of any cell.

Over the eleven estimates (the three decode cells with distinct texts, the two samplers of the
16-prompt cell, three K of the invariance cell, three K of the gate) the mean is 2.1 points and
the range +1.3 to -5.8.

Theory is on the same side: the rejection walk accepts with probability 1 - TV(p, q), the
maximum for any coupling; the Gumbel coupling is bounded below by (1 - TV) / (1 + TV).

## A trimmed draft head

The Gumbel kernels key their noise by row index. A draft head with a trimmed vocabulary holds
token `d2t[i]` in row i, so its draw at a position was not the target's noise for the same token
and the two draws were close to independent: the text stayed right and acceptance fell. The
draft-side kernels now take the head's map and key row i by `d2t[i]`. Acceptance per drafted
token, RTX 5090, the gate's eight seeded prompts per K (`pr/rtx5090/trim-9b/`,
`pr/rtx5090/trim-27b/`):

| model and drafter | binary | K=1 | K=2 | K=3 | gate |
|---|---|---:|---:|---:|---|
| Qwen3.5-9B, its own-trim drafter | main | 0.655 | 0.521 | 0.383 | FAIL |
| | this change without the map | 0.510 | 0.362 | 0.260 | PASS |
| | this change | 0.647 | 0.472 | 0.356 | PASS |
| Qwen3.8-27B, its trimmed FR-Spec drafter | main | 0.687 | 0.510 | 0.417 | FAIL |
| | this change without the map | 0.549 | 0.414 | 0.313 | PASS |
| | this change | 0.667 | 0.520 | 0.406 | PASS |

Is the trimmed head's remaining gap its own cost? No sign of it. Paired by prompt, this change
against main: the 9B -0.008 +- 0.022 at K=1, -0.048 +- 0.023 at K=2 (per prompt -0.17 to +0.05)
and -0.027 +- 0.019 at K=3; the 27B with its trimmed drafter -0.019 +- 0.018, +0.010 +- 0.026 and
-0.011 +- 0.035. The untrimmed 27B head in the same gate pays more (-0.035, -0.058, -0.041,
above). A token outside the kept rows can never be accepted, but the rejection walk could not
accept it either: it is the same lost mass under both rules. No decode cell was run on the 9B.

With the map, 32 of 32 texts are the ones the run without it wrote. `local-ci`'s
`spec-on-cache-hit` gate (the 9B with that drafter) failed without the map at "sampled hit
acceptance == cold acceptance exactly", because the low acceptance made the cold run serve K=3
and the hit K=2; with it both of its arms pass. At the final code the
qwen arm showed one more thing: on this card the spec cost model owns K, and it served a cold
run at K=3 and its hit at K=2 (and the reverse for another seed), so their acceptance counts
differed while their texts were equal. The gate's parity assertion is now per draft length: a
pair served at two depths is named and not compared, and at least one pair must compare
(`pr/rtx5090/hitgate/`; `qwen-before-the-per-K-rule.log` is the run that failed). `sample_check` cell 2c pins the key on the three
kernels.

## Limits

One model family, two devices, 400-token answers. Not measured: the RTX PRO 6000, three or more
requests at once, a trimmed multi-head chain. The routes that are not position-keyed and the
penalised class that stays on the host sampler are listed in the decision record.
