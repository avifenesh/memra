# DSv4-Flash on two cards: TP/EP by default, PP-2 as the rollback (2026-09-25; pp amended to a placement 2026-11-04)

> Amended 2026-11-04: `pp` is no longer a rollback seam with a decide-by. It stays as the PP-2
> value of the placement selector ([amendment](#amendment-2026-11-04-pp-is-a-placement)). The
> TP/EP default below is unchanged.

## Decision

`memra-server` loads DSv4-Flash on two cards as TP/EP (memra #710, owner rulings 2026-09-25):

- Every layer sits on both cards.
- Experts are split by id (0..128 on card 0, 128..256 on card 1).
- Attention is split by head (exact attention TP2).
- The drafter's experts are split the same way.
- Plain greedy and vendor-default sampled requests replay each step on the full-token CUDA
  graphs.

`MEMRA_DSV4_TOPOLOGY=pp` restores the PP-2 layer split. It is decide-by 2026-10-09 (dropped
2026-11-04, see the amendment).

## Why

The bounds, from `research/dsv4f-bringup-20260923/ceiling/CEILING.md`:
- A token reads 11.44 GB of weights.
- PP-2 streams it through one card at a time: 135 tok/s at c1 on 1.54 TB/s practical.
- TP-2 streams both halves at once: 270 tok/s before its all-reduces.
- The published vLLM anchor on this shape (109.3 tok/s) is TP=2.

PP-2 cannot pass 135 at c1, so the ceiling program targets TP-2 (owner, 2026-09-24).

Measured on both pair classes (`research/dsv4f-bringup-20260923/tpep-default/RESULTS.md`), with the
same text on every request of every arm:

| | TP/EP | PP-2 |
|---|---|---|
| plain greedy c1, Workstation pair | 80.73 | 68.51 |
| plain greedy c1, Server Edition pair | 71.20 | 62.38 |
| plain sampled c1, Server Edition pair | 72.65 | 58.40 |
| DSpark greedy c1, Server Edition pair | 72.79 | 61.72 |
| TTFT p50, c1 greedy, Server Edition pair | 172 ms | 210 ms |

## What it costs, and what fixes it

- **Plain concurrency (narrowed, memra #710 B-row).** TP/EP now serves four lanes, whose plain
  steps share a captured B-row graph step (`research/dsv4f-bringup-20260923/tp-rows/`). On the
  Workstation pair: c4 aggregate 132.8 against PP-2's 120.6 with TTFT 0.42 s against 4.5 s, and
  c2 102.0 against 120.9. The rest of this item is the state at the flip. TP/EP served one lane. PP-2 pipelines two requests across its cards
  (#667). Plain c2 aggregate is 76.9 tok/s against 120.9 on the Workstation pair (-36%), and
  TTFT under load rises tenfold. The DSpark route is serial on both placements, so it does not
  pay this. The owner took the flip with this cost on the record ("Flip now, B-row next"). A
  TP/EP B-row step, several requests' rows in one TP step, is the next lane.
- **Context.** TP/EP replicated every layer's KV cache: a session held 300k tokens with DSpark
  and 800k plain, against PP-2's 1M. Since the position-split C4 store (#710,
  `research/dsv4f-bringup-20260923/kv-split/`) it holds 500k and 1M.

## Rejected

- **Keep PP-2 and bank TP/EP as gate-only.** This leaves the c1 ceiling at PP-2's 135 tok/s
  bound.
- **Hold the flip until the head-split KV lands.** Not chosen: the owner ruled to flip first.
- **Hold the flip until the TP/EP B-row step lands.** Offered with the concurrency data; the
  owner chose to flip first.
- **Default TP/EP only when the drafter is armed.** Also offered; not chosen.
- **The replicated-attention TP/EP arm (`tp_ep` without attention TP).** It ran 65.02 tok/s
  against 69.11 with attention TP (N=3/N=5, 2026-09-23, `tpep-serve/RESULTS.md`). Its served
  name is gone.

## Amendment, 2026-11-04: `pp` is a placement

Chosen (owner, 2026-11-04): `MEMRA_DSV4_TOPOLOGY=pp` stays as an explicit placement, the PP-2
value of the placement selector. It is not a rollback seam and has no decide-by.

Why, from `research/dsv4f-bringup-20260923/tp-rows/RESULTS.md` (N=3 per cell, one boot per row):

| greedy c2 | TP/EP, four lanes | PP-2, two pipelined lanes |
|---|---|---|
| aggregate, Workstation pair (final head) | 103.17 tok/s | 120.79 tok/s |
| TTFT p50, Workstation pair (final head) | 242 ms | 309 ms |
| aggregate, Server Edition pair (graph lane) | 93.36 tok/s | 113.15 tok/s |

- PP-2 aggregates more at c2, at a higher TTFT. TP/EP leads at c1 and at c4 (135.76 against
  120.63 on the Workstation pair, TTFT 430 ms against 4462 ms).
- Every DSv4 gate names its placement (`Dsv4Gpu::load` has no default since 2026-11-03), so a
  PP-2 cell runs only where a gate names `pp` and cannot stand in for the served placement.
- The same receipt records four PP-2 stalls with two requests in flight on one pod (#722); no
  TP/EP row stalled. Keeping `pp` keeps that open on the PP-2 placement.

Rejected:

- **Delete the server `pp` arm under the expired-date rule.** Its decide-by (2026-10-09) passed
  with no receipt. Deleting it takes the PP-only serving code with it (the two pipelined
  default lanes, the two-group B-row coalescer, the PP-2 rows arm) and drops a measured c2 win.
- **Re-date it as a rollback seam with a named missing gate** (a 16-lane TP/EP against PP-2 c2
  A/B; the receipt above compared four TP/EP lanes). Not chosen: the c2 gap is measured, and
  `pp` is kept for that workload, not as a way back to an old default.

What follows: the PP-2 arm of `MEMRA_DSV4_ROWS` had shared the `pp` decide-by. It now has its
own, 2026-11-18, with its missing gate unchanged (the served A/B against the two-lane pipelined
default; `docs/FLAGS.md` row).
