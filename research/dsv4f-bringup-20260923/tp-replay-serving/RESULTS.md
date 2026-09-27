# TP/EP full-token replay for serving: capacity, greedy, and the served checks (memra #710)

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Server Edition, 2026-09-24. Program: the TP2 attention, expert-id EP
full-token program on the served one-token stream MoE (#719). Lane
`lane/dsv4-tp-replay-capacity-20260924`. Gate: `dsv4_tp_replay_long_gate`, eager against
armed replay at every step (token, logits bits, TP/EP cache and hidden digests), plus replay
counters on both ranks.

## What changed

Three admission limits that kept the replay graphs out of serving:

1. **Capacity** (`4a7d0433e`). Replay armed only capacities of 512..=1024, the first probe's
   admission scope. The replay indexer's real bound is 4,096 compressed blocks, 16,384 positions
   at ratio 4, so replay now admits 512..=16384.
2. **Greedy** (`12a0517b0`). Replay armed only the vendor-default sampling configuration. Now a
   temperature-0 request's commit graph captures the eager greedy step's own device argmax where
   the sampler sat, and the step reads the 4-byte result after the drain.
3. **The served checks** (`b895160d5`). Replay required route and mirror validation off. It
   predates #670 and #679, which moved those checks into device fault words read once per step.
   Replay now arms the words every step, captures the checked route kernels, and reads the words
   with the one-shot refusal words between the forward and commit launches. A nonzero word rolls
   both planes back before anything commits, which is the eager step's own rule.

## Correctness (`raw/q-v3t.summary`, `raw/q-v3u.summary`, `raw/q-v3v.summary`)

| run | checks | sampling | capacity | steps (positions) | result | tokens sha |
|---|---|---|---|---|---|---|
| capacity | off | default | 4096 | 3000 (400..3400) | PASS | b7cf831844bda91f |
| capacity | off | default | 16384 | 3000 (400..3400) | PASS | b7cf831844bda91f |
| capacity | off | default | 1024 | 600 (400..1000) | PASS | def83d3424a9f716 |
| capacity | off | default | 4096 | 600 (400..1000) | PASS | def83d3424a9f716 |
| greedy | off | greedy | 1024 | 304 (400..704) | PASS | 2b6e1eba196f53db |
| greedy | off | greedy | 4096 | 1000 (400..1400) | PASS | 8f1560aa3e903ae8 |
| checks | **on** | default | 1024 | 304 (400..704) | PASS | 112c2fc66fe1f760 |
| checks | **on** | greedy | 1024 | 304 (400..704) | PASS | 2b6e1eba196f53db |
| checks | **on** | default | 4096 | 1000 (400..1400) | PASS | 7b2668dd65acb08d |

- **Capacity:** it does not change the program. The same positions give the same tokens at
  capacity 1024, 4096 and 16384.
- **Checks:** turning them on does not change the program. The default and greedy shas with the
  checks on equal the runs with them off, and equal the #719 run.
- **Counters:** on every run all steps are replays on both ranks. The 3,000-step runs pass 727 C4
  and 23 C4-plus-C128 emissions.

## Timing (alternating order, wall time per token, N=3)

| run | eager tok/s | replay tok/s |
|---|---|---|
| checks off, default, 304 steps (#719, `../tp-replay-stream/raw/se/`) | 56.74..57.79 | 71.55..72.00 |
| checks on, default, 304 steps | 57.01..57.81 | 70.54..71.80 |
| checks on, greedy, 304 steps | 57.12..57.84 | 70.86..71.95 |
| checks on, default, 1000 steps at capacity 4096 | 56.94..58.07 | 71.31..71.52 |
| checks off, capacity 1024 against 4096, 600 steps | | 71.05..71.47 against 71.14..71.59 |

- **Checks:** reading the fault words costs under 1%.
- **Capacity:** the larger indexer grid costs nothing at equal positions.

## What still keeps replay from serving

- Replay refuses a loaded drafter. DSpark rounds verify multi-row, which is not the full-token
  step.
- Capacities above 16,384 need a streaming top-k. Until then such a request stays on the eager
  step, which is bit-identical, so a request never changes program.
- Replay is reached only through the gate API. Main carries no served TP/EP topology selector; it
  returns with the TP/EP default flip.
