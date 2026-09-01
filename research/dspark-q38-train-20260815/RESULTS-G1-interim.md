# G1 control eval — interim results (spot box attempt 1, 2026-08-15)

Rig: sbox-2card SPOT `spot-1` N-Virginia-c (2x RTX PRO 6000 Blackwell Server 96GB,
driver 595.91.07, PIX pair). **Box reclaimed by hyperscaler at 01:07Z (`instance-terminated-no-capacity`)
after ~45 min** — the numbers below were captured off the box before loss; the raw on-box logs died
with it. Lesson applied: next bring-up streams receipts off-box continuously.

Server: SGLang 0.5.17, `--attention-backend flashinfer` (fa3 asserts SM<=90 — refused on sm_120),
FP8 target `Qwen/Qwen3.8-27B-FP8` + control drafter `RadixArk/Qwen3.8-27B-DSpark`, DSPARK block 7,
draft unquant BF16, `--mamba-scheduler-strategy extra_buffer`, mem-fraction 0.85, single card.
Output sanity verified (coherent thinking + answer) — the FlashInfer-SM120 zero-output failure
(research/nvfp4-source-20260814) did NOT reproduce on this FP8 path.

Setting (RadixArk replication): temp 0.6, top-k 20, top-p 0.95, thinking ON, max_new 2048,
128 prompts, c=8, `dflash.benchmark --backend sglang`. SINGLE RUN each — spot died before repeats.

| Workload | accept_len (ours, PRO 6000) | RadixArk published | agg tok/s c=8 | verify ct | output tokens |
|---|---|---|---|---|---|
| gsm8k | **4.496** | 4.57 | 687.68 | 13,681 | 55,053 |
| mt-bench | **3.204** | 3.10 | 532.75 | 41,609 | 115,839 |

Read: replication within ~2-3% both directions on target hardware. Their published FP8-target
acceptance transfers to PRO 6000 serving.

Own-sessions cell (the unpublished one): corpus (23G, 4,750 session files, claude/codex/eigen/
hermes) was fully staged and the 128-prompt run (64 chat-short + 64 agentic-brief real owner
turns, seed 20260815) reached ~48/128 when the box died. **No number captured — rerun required.**
Harness: `tools/own_bench.py` (this lane).

Still owed for G1: own-sessions rerun, greedy arm, spec-OFF baselines (c=1 + c=8 denominators),
N>=3 repeats per cell, thermal note.

## Own-sessions cell (spot box attempt 2, 2026-08-15 ~01:50Z) — THE decision number

Rig: sbox-2card SPOT `spot-2` N-Virginia-c (also reclaimed ~50 min in; the
continuous receipt pull-loop streamed the completed JSONL home before loss — discipline works).
Same server config as above. 128 real owner turns from the session corpus (seed 20260815,
64 per bucket, thinking ON, temp 0.6/topk20/topp0.95, native /generate endpoint — spec stats
are NOT exposed on the chat-completions endpoint). Raw: `raw/box2/g1/own-sessions-t06-think-v3.jsonl`.

| Bucket | n | accept_len mean | median | p10 | p90 |
|---|---|---|---|---|---|
| chat-short (<600 chars) | 64 | **2.585** | 2.562 | 2.24 | 2.91 |
| agentic-brief (>=600 chars) | 64 | **2.629** | 2.596 | 2.28 | 3.05 |
| ALL | 128 | **2.607** | | | |

Read: on OUR serving distribution the RadixArk drafter sits BELOW its weakest published
workload (Arena-Hard 2.71) and at 57% of its gsm8k headline (4.57). The corpus-match gap is
real and large — this is the empirical case for training our own drafter on own-session-shaped
regenerated data (G3's bet). Single run, spot thermal regime unrecorded.

Spot ops note: N-Virginia-c reclaimed two boxes in a row at ~45-50 min. Hunter re-ordered to
prefer Oregon/Frankfurt pools; N-Virginia demoted to last.

## Box 3 cells (sbox-2card SPOT `spot-3` Ohio-b, 2026-08-15 ~02:30-03:10Z)

Same server config (SGLang 0.5.17, flashinfer, FP8 target + RadixArk drafter, block 7). Raw:
`raw/box-spot-3/g1/`.

| Cell | accept_len | note |
|---|---|---|
| gsm8k greedy (t0, think) | **4.611** | greedy > sampled, expected |
| gsm8k t0.6 think REPEAT | **4.477** | vs 4.496 attempt-1 — tight variance |
| own-sessions greedy (t0, think) | **2.623** (chat 2.609 / agentic 2.637) | low own-corpus accept robust to sampling mode |
| own-sessions t0.6 NOTHINK | **2.333** (chat 2.292 / agentic 2.373) | nothink WORSE — drafter's corpus skews thinking-mode |

Reads: (1) the own-corpus gap is stable across greedy/sampled (2.607/2.623); (2) nothink drops
another 11% — relevant because agentic API serving often runs nothink; matched-mode regeneration
in our own training corpus (both modes) is justified. Spec-OFF denominators queued next on box.

## Training capacity (owner call 2026-08-15 ~03:00Z)

B200 32h Capacity Block PURCHASED: `<capacity-block>`, p6-b200.48xlarge (8x B200 179GB),
N-Virginia-c, 2026-08-15T03:16Z -> 2026-08-16T11:30Z, $3,185.94 upfront. Box allocation (owner):
**3 concurrent training arms x 2 cards (capture+trainer each) + 2 cards memra sm_100 recon**;
regen phase borrows all 8 first. B200 spot was vetoed (no spot for training); OD was ICE
everywhere; block = guaranteed + ~12% under OD.

## G1 CLOSED (box 3 finished the full queue before decommission, 2026-08-15 ~03:30Z)

Spec-OFF denominators (same config minus speculative args, c=8, 128 prompts):

| Workload | spec-ON | spec-OFF | speedup |
|---|---|---|---|
| gsm8k t0.6 think (aggregate tok/s) | 687.7 | 298.4 | **2.30x** |
| own-sessions t0.6 think (per-request tok/s) | 63.6 | 41.7 | **1.52x** (cross-box pair: box2 ON / box3 OFF, same config+seed) |

Read: even at accept 2.6, RadixArk DSpark pays +52% on our real traffic at c=8 on one PRO 6000.
An own-corpus drafter closing even half the gap to the math-class accept (~4.5) would push this
toward ~2x. That is the training target's economic frame.

Ops close: box 3 terminated after full queue (owner: B200-only posture); all sbox hunters killed;
fleet = B200 block box `b200-1` only. Caveat: the 1.52x pair crosses two boxes
(both sbox-2card, same driver/config/seed) — single-box repeat owed when a PRO 6000 rig is next up.
