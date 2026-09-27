# DSv4-Flash on two cards: TP/EP by default, PP-2 as the rollback (2026-09-25)

## Decision

`memra-server` loads DSv4-Flash on two cards as TP/EP (memra #710, owner rulings 2026-09-25):

- Every layer sits on both cards.
- Experts are split by id (0..128 on card 0, 128..256 on card 1).
- Attention is split by head (exact attention TP2).
- The drafter's experts are split the same way.
- Plain greedy and vendor-default sampled requests replay each step on the full-token CUDA
  graphs.

`MEMRA_DSV4_TOPOLOGY=pp` restores the PP-2 layer split. It is decide-by 2026-10-09.

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
