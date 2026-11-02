# Pooled MTP draft across sessions

Step 3 of the batch-spec lane. In the lockstep cohort (`generate_spec_sessions_pooled`), draft
step j is one NextN head forward over the members still drafting at step j. The MTP layer and
the draft head stream once per step instead of once per session.

## What runs

- **Split** (pure refactor). `spec_round_prepare` is now three calls, and the serial driver runs
  them back to back in the old device order:
  - `spec_round_draft_begin`: snapshot, K, draft-KV sync, penalty window, draft-mask clone.
    Returns a `SpecDraftRound`.
  - `spec_round_draft_serial`: the old arm chain (chain graphs, single-head graphs, eager), moved
    verbatim.
  - `spec_round_draft_end`: verify tokens and checkpoint.
- **Rows forward** (`mtp_head_forward_rows`). The `mtp_head_forward_dev` / captured-graph op chain
  at m=b:
  - row-wise kernels and the exact small-m matvec tier (rows at m=2..16 are the m=1 program, the
    verify==decode identity) for embed, norms, eh_proj, q/k/v, wo, FFN, out_up and head;
  - per row, on that session's own scratch: the device-counter append and `fa_decode_dc`, with
    the serial launcher and bucket.
- **Pooled draft** (`spec_round_draft_pooled`). Ragged K: a member leaves after its own `k_this`,
  at its p-min stop, or on its own error. Each member's tail runs its serial arm's kernels
  (`PooledDraftArm`) on its own logits row and its own draft context:
  - `GraphGreedy`: in-graph argmax (and the pick's probability) into `g_tok` / `g_p`.
  - `GraphSampled`: q retention into `q_slots[j]`, device counter bump, pure or filtered gumbel,
    argmax, raw row max; post-chain stats for the pure-temp capture.
  - `Eager`: greedy argmax, or the eager sampled tail (penalties, filter stats, host-counter
    gumbel).
- **Who pools** (`pooled_draft_arm`, `mtp_rows_draft_capable`). Single-head, full-attention NextN
  block, dense FFN, no step35 geometry, gate or AWQ scale, one flat scratch plane, no draft mask,
  every projection on the batched tier. Others draft serially inside the same cohort; constrained
  sessions are not in cohorts. Chunks of at most 16 members.
- **Receipt**. `[spec-pool] pooled rows draft engaged: N members (arms [...])`, once per N.

## Exactness

1. **Draft cells** (`decode-batch-bench --pooled-draft`, `pooled_draft_probe`, raw
   `draft-cells-q9.log`). Each member opens a continuation burst and runs its first round twice
   from the same state. Serial is its own chain, the captured graphs where the session has them.
   - Compared: draft tokens, trimmed ids, filter stats, every retained q row, the Philox counter,
     the draft-KV length (host and device), the KV bytes the round wrote, and a follow-up eager
     step that attends over them.
   - `ROWS-STEP` also pins the rows forward's logits rows, carriers and KV rows against per-session
     m=1 eager steps.
   - 12 cells, all **BITWISE EQUAL** (94 lines, 0 mismatches):

| cell | shape | arms |
|---|---|---|
| greedy-ragged | K 3,3,2,4 | graph greedy |
| mixed-ragged | K 3,2,4,3, T=0.8 odd | graph greedy + sampled |
| sampled-pmin | 8 x K 1..4, p-min 0.3 | graph sampled, p-min stops |
| greedy-pmin | 8 x K 2..4, p-min 0.5 | graph greedy, p-min stops |
| filtered | 4 x K3, top_p 0.9 | filtered sampled graph |
| penalized | 4 x K3, repeat 1.1 | eager sampled + penalties |
| eager-greedy / eager-sampled | K 3,2,4,3, `MEMRA_SPEC_NOGRAPH=1` | eager |
| b2 / b4 / b8 | K3, depth 2048 | mixed |
| b16-mixed | 16 members, K 1..4 | mixed, exact scope at m=16 |

2. **Serial paths** (the ROUND-SPLIT battery on this binary, against the step-2 pooled reference
   `pool`). `battery/compare-batched-draft-paths.txt` is **BYTE-IDENTICAL**:
   - run-spec greedy and sampled K=1..8, adapt + p-min, replay;
   - every served cell, including the concurrent greedy pair (which now pools its draft), and the
     56 + 4 + 4 `[spec-acc]` lines.
3. **Served streams** (`battery/pool_cell.py`, against the step-2 serial reference
   `pool-serial-a`). `battery/compare-batched-draft-pool-cell.txt`: **14/14 streams identical**
   on all nine boots of the A/B below.
   - The pooled-draft boots logged rows-draft cohorts of 2 to 8 members, mixing `GraphGreedy`
     and `GraphSampled`.

## Speed

### Draft only

`DRAFT-TIME`: the serial chains of every member vs one pooled draft, same state, median of 5.

| members | K | serial ms | pooled ms | gain |
|---:|---|---:|---:|---:|
| 2 | 3 | 2.86 | 1.79 | 1.60x |
| 4 | 3 | 5.60 | 2.48 | 2.25x |
| 8 | 3 | 11.36 | 4.27 | 2.66x |
| 16 | 1..4 | 21.49 | 8.00 | 2.69x |
| 4 | 3, top_p 0.9 | 7.10 | 4.06 | 1.75x |

The filtered arm pays three scalar readbacks per member per step, as its serial graph arm does.

### Served (the pool cell)

RTX 5090 Laptop, q9 + owntrim draft, aggregate tok/s. One window, three arms, N=3 each,
interleaved in both orders (draft, pool, serial, serial, pool, draft, draft, pool, serial).
Medians:

| c | serial (split) | pooled verify (step 2) | pooled draft (this) | vs step 2 | vs serial |
|---:|---:|---:|---:|---:|---:|
| 2 | 152.19 | 169.94 | 174.24 | +2.5% | +14.5% |
| 4 | 144.10 | 162.13 | 172.00 | +6.1% | +19.4% |
| 8 | 144.09 | 161.23 | 174.49 | +8.2% | +21.1% |

Raw values per run are in `battery/compare-batched-draft-pool-cell.txt`.

**Window drift.** The step-2 receipt measured serial 101.8 / 98.0 / 98.2 and pooled 113.5 / 110.5
/ 110.9 with the same two binaries. Their ratio holds (pooled verify is +11-13% here too), but
absolute rates in this window are about 1.5x higher. The cause was not captured. Compare arms
only within one window.

No thermal or 250 ms telemetry, and no TTFT/ITL, were recorded. This is a mechanism landing with
bitwise exactness and no door, not a default decision.

## Binaries

sha256 prefixes:

| binary | sha256 |
|---|---|
| this commit `memra-server` | `730cdfe418464177` |
| this commit `run-spec` | `a48db38161212537` |
| this commit `decode-batch-bench` | `d73cc16eacb1302d` |
| step-2 pooled `memra-server` | `13ec038086e7700a` |
| serial split `memra-server` | `559b914819a7ffa4` |

An earlier build without the engagement line (no other difference) gave the same verdicts on the
draft cells and the pool cell.

clippy `-D warnings` (memra-engine, memra-server, all targets): clean. Lib tests: memra-engine
674 pass, memra-server 1140 pass.

## Not pooled

- **Multi-head chain** (`mtp_extra`): each step replays a growing row prefix per head.
- **step35 MTP blocks**: windowed dcw attention.
- **MoE MTP FFN** and **draft-masked (constrained) rounds**.

These rounds keep their serial chain inside the cohort.
