# step37 decode headroom: spec-on qualification + MEMRA_W8_VIEW (2026-08-28)

Lane question: does anything reach >90 tok/s vendor-default sampled on Step-3.7-Flash
NVFP4 TP2, now that the spec use-after-free is fixed (7694c049f8)? Candidates: spec-on
(the serve-config policy knobs) and MEMRA_W8_VIEW (q8 mirror for the row-range-view
bf16 GEMVs that HEAD_SPLIT / SHEXP_OVERLAP keep on rank 0).

Instrument: `instruments/s37h-spec.sh` driving `s37h-spec-probe.py`, interleaved x5,
one server boot per cell, vendor-default sampled (request carries NO sampling params),
curve-0400 (613-token real prompt), 3 streaming reps per cell, thinking-model bytes
counted from `message.reasoning`, empty completions rejected in the instrument.
Binary: lane tip 553a072471 + the W8_VIEW patch (banked as commit e52690aea1),
md5 c06b3c8cb9ef, fingerprinted by strings, never cargo's Finished line.
Raw receipts: `s37h-spec-results.txt`. Box: 2x RTX PRO 6000 Blackwell WS (vast).

## Per-arm interleaved table (5 cells/arm, each cell = median of 3 loop-clean reps)

| arm    | env delta vs plain                             | median | min   | max   | spread | vs plain |
|--------|------------------------------------------------|--------|-------|-------|--------|----------|
| plain  | MEMRA_SERVE_SPEC=0 MEMRA_MTP_HEADS=1           | 78.41  | 77.86 | 78.55 | 0.69   |          |
| spec   | SERVE_SPEC=1 SPEC_K=3 MTP_HEADS=3 PMIN=.5/0=1  | 92.13  | 89.66 | 97.69 | 8.03   | +17.5%   |
| specv  | spec + MEMRA_W8_VIEW=1                         | 94.61  | 92.95 | 95.81 | 2.86   | +20.7%   |
| plainv | plain + MEMRA_W8_VIEW=1                        | 78.46  | 78.02 | 78.62 | 0.60   | +0.1%    |

Exclusions stated: rnd1 spec had 1 of 3 reps flagged LOOP and excluded by the
instrument (cell median over the remaining 2, rows=2 in the raw file). Every other
cell rows=3, notes=none. `illegal=0` in all 20 boots (the 7694c049f8 fix holds under
sustained spec load; the pre-fix memo arm crashed 17/25 ILLEGAL).

Engagement audit (usage.spec from the RESPONSE BODY, both directions): spec and specv
ENGAGED 5/5 cells (e.g. rounds=44,drafted=77,accepted=73); plain and plainv ABSENT
5/5 (clean controls).

## Correctness gates

- SPEC (bar = FULL byte identity, greedy instrument, 4 real prompts incl. agentic8
  and curve-1000): first token AND full tape byte-identical to spec-off on every
  prompt. PASS. (`instruments/s37h-specgate-*.json`)
- MEMRA_W8_VIEW numeric gate (weight-precision door, MEMRA_SERVE_SPEC=0): on run-gen
  the door ENGAGES (view_mirrors=1) and the decode argmax MATCHES on real prompts at
  logit maxdiff 9.546e-2 (curve-0400; the w8view=0 baseline's own maxdiff is the same
  9.546e-2) and 1.034e-1 (curve-1000). That is the BF16_MMV ~9e-2 class, the class
  that passed the max_tokens=1 first-token discriminator; server-route byte tape
  plainv == plain on all four gate prompts.

## THE W8_VIEW FINDING: the door is inert on the served route

`[w8-view] mirror built` appeared in ZERO of the 20 server boots, including every
MEMRA_W8_VIEW=1 arm (and the OFF direction is proved by the same counter). The served
step37 decode (MEMRA_STEP_TP_DECODE_V2 t-walk and the spec verify walk) never calls
`matvec_bf16_view_into`; the only callers of the view launcher are the host
`decode_step_h` path and `decode_step_chain`, which is where run-gen engaged it.
Therefore:

- plainv vs plain (+0.1%) is the honest null: an inert flag measures nothing.
- specv's 94.61 vs spec's 92.13 is NOT a W8_VIEW effect. The door provably never
  engaged; spec's own spread (89.66..97.69) covers specv's entire range. The specv
  cells are 5 more spec replicates: pooled over all 10 engaged spec cells the median
  is 94.05 tok/s, min 89.66 (9 of 10 cells clear 90).
- The byte-ledger premise (bf16 lo halves on the critical path, ~0.33 ms/token prize)
  described the split paths, which the served walk does not take. Routing the walk's
  head/shexp lo GEMVs through the view mirror is possible follow-up work, capped at
  ~0.33 ms/token (~2.3 tok/s at this operating point) by `census-ledger-at-78.6.txt`.

## VERDICT

spec-on clears the bar; MEMRA_W8_VIEW does not participate. Recommended serving
config: the pinned step37 serving env (agentic8 ENVV, incl. MEMRA_BF16_MMV=1) +
`MEMRA_LOAD_MTP=1 MEMRA_CTX=262144 MEMRA_SERVE_SPEC=1 MEMRA_SPEC_K=3
MEMRA_MTP_HEADS=3 MEMRA_SPEC_PMIN=0.5 MEMRA_SPEC_PMIN0=1` (MEMRA_W8_VIEW unset).
Measured: 92.13 tok/s median (spec arm; 94.05 pooled over 10 engaged cells,
min 89.66), vendor-default sampled, spec engagement in every row, byte-identity
gate PASS, illegal=0. >90 tok/s: MET on the interleaved median. TTFT was already
met (968 ms @613 tokens). MEMRA_W8_VIEW: default stays OFF (inert in serving;
run-gen argmax gate PASS banked for the day the walk is rewired).

Remaining before a prod flip: this is a single-shot 613-token cell; the owner's
serving-decision law still wants the 8-turn larger-prompt cache-on twin and the
post-deploy vendor-default-shape probe with a spec-engagement receipt.

# 2026-08-29 re-baseline: rank-serialization spans at t=4096 (post-fix) + spec policy sweep

Box: the rented dev box (2x RTX PRO 6000 Blackwell Server Edition). Binary:
`/home/ubuntu/memra/target/release/memra-server`, branch `lane/step37-main-merge-20260828`,
md5 `f45c3623d958ca085eefd3207987812a` (verified before first boot and printed in the run
header; the checkout's tip drifted 8695bdef4 -> 18cb988fa mid-session under another agent,
the md5 pin is the identity). Model `/root/models/step37-flash-nvfp4`. Shipping config:
agentic8 ENVV + `MEMRA_LOAD_MTP=1 MEMRA_MTP_HEADS=3 MEMRA_SPEC_K=3 MEMRA_SPEC_PMIN=0.5
MEMRA_SPEC_PMIN0=1 MEMRA_CTX=262144 MEMRA_SERVE_SPEC=1`. Raw receipts:
`raw-rebaseline-20260829/` (orchestrator `rb-orch.py`, run logs, prof harvests, per-boot
spec-acc receipts). All 24 boots: ILLEGAL=0, #87=0.

## Task A: the grouped-MoE prime joins are STILL SERIALIZED at t=4096

`MEMRA_PRIME_PROF=1`, curve-30k-1 (39,546 tokens) and curve-30k-2 (35,681 tokens),
stream, max_tokens 16, log marked before each request, rows harvested from that
request's slice only. Per-t medians over all `[grp-prof]` rows (42 grouped layers/chunk):

| prompt | t | rows | med join | med span_sum | med span_max | serial ratio* |
|---|---|---|---|---|---|---|
| 30k-1 | 4096 | 378 | 18.70 ms | 15.70 ms | 8.50 ms | 1.42 |
| 30k-1 | 2656 | 42 | 12.40 ms | 11.40 ms | 6.20 ms | 1.19 |
| 30k-1 | 26 | 42 | 1.00 ms | 1.80 ms | 0.90 ms | 0.00 |
| 30k-2 | 4096 | 336 | 18.80 ms | 15.60 ms | 8.50 ms | 1.45 |
| 30k-2 | 2880 | 42 | 13.40 ms | 12.20 ms | 6.60 ms | 1.21 |
| 30k-2 | 33 | 42 | 1.10 ms | 2.00 ms | 1.00 ms | 0.09 |

*(join - span_max)/(span_sum - span_max): 0 = overlapped, 1 = serialized. No negative
(failed-query) spans in any row.

VERDICT: serialized at large t, post-fix. Small t overlaps (ratio ~0 at t=26/33), large
t does not, and the join wall EXCEEDS span_sum by ~3 ms/layer at t=4096 (the fused
join pass itself: two dtod partial copies + join/scatter + sync sit inside the join
window after the serialized spans). vs the banked pre-fix numbers (span_sum 33.90 /
span_max 17.90 at t=4096): the NVFP4_V2-era binary roughly HALVED the spans themselves
(15.7 / 8.5) but the ranks still do not overlap. The lever stands, and it is the top
TTFT lever for long prompts:

- 30k-1 joins total 7.64 s of the 15.31 s TTFT (50%); span_max totals 3.52 s. Collapsing
  join to span_max is worth ~4.1 s: 15.3 s -> ~11.2 s at 39.5k tokens.

## First honest 32k-class TTFT (vs vLLM's receipted 11.2 s @32k)

Cold (cached_tokens=0), spec engaged (usage.spec in the response body), ILLEGAL=0:

| prompt | prompt_tokens | TTFT | per-token |
|---|---|---|---|
| curve-30k-1 | 39,546 | 15.31 s | 0.387 ms/tok |
| curve-30k-2 | 35,681 | 13.46 s | 0.377 ms/tok |

Both prompts are LARGER than 32k; the marginal slope between the two points is
0.479 ms/tok, which puts a 32,768-token prompt at ~12.1 s (extrapolation, not a
measurement). `[prime-prof]` chunk decomposition, 30k-1 (10 chunks, 9x t=4096 + 2656):
moe 9.32 s / attn 3.04 s / norm+qkv 1.42 s / o_proj 1.05 s = 14.82 s accounted of
15.31 s. MoE is 63% of prefill and the serialized grp join is 82% of the MoE time.

## Task B: spec policy sweep (curve-0400, 613 tokens, vendor-default SAMPLED)

Instrument: `raw-rebaseline-20260829/rb-orch.py`, lineage s37h-spec-probe (tok/s =
(n_deltas-1)/(t_last-first), 3 reps/cell, cell = median of loop-clean reps, one boot per
cell, warmup on a different prompt, engagement probe = non-streaming usage.spec).
max_tokens 400, interleaved cycles with boot order alternating by parity, prune >7%
below the cycle-1 best. Cell receipts: `run.log` CELL lines + per-boot `[spec-acc]`
tails (`spec-acc-receipts.txt`). Zero LOOP flags, zero empty completions, all cells
spec-engaged.

| arm (K/heads/pmin) | cells | median | min | max | spread | probe acc | note |
|---|---|---|---|---|---|---|---|
| K3 H3 P0.5 (pinned) | 9 | 146.62 | 131.66 | 150.24 | 18.58 | 0.69-0.99 | winner, every stage |
| K4 H3 P0.5 | 3 | 134.54 | 128.53 | 139.60 | 11.07 | 0.90-0.95 | |
| K2 H3 P0.5 | 3 | 124.85 | 121.68 | 137.42 | 15.74 | 0.80-0.93 | |
| K5 H3 P0.5 | 1 | 131.60 | | | | 0.85 | pruned after cycle 1 |
| K3 H1 P0.5 | 3 | 126.36 | 121.10 | 134.46 | 13.36 | 0.82-0.97 | |
| K3 H2 P0.5 | 0 | | | | | | NOT SERVABLE, see below |
| K3 H3 P0.3 | 1 | 137.36 | | | | 0.83 | pruned after cycle 1 |
| K3 H3 P0.7 | 1 | 123.41 | | | | 0.91 | pruned after cycle 1 |

VERDICT: **the pinned policy (K=3, HEADS=3, PMIN=0.5, PMIN0=1) is already optimal.**
It won stage 1 (vs K2/K4/K5), stage 2 (vs H1), and stage 3 (vs PMIN 0.3/0.7), with 9
interleaved cells spanning all three stages (x5 bar exceeded). The aborted first run's
4 cells corroborate the ordering (`run1.log`: K3 148.35 > K4 138.31 > K2 129.56 >
K5 123.29). No policy change recommended.

- MEMRA_MTP_HEADS=2 is NOT a servable config on this pack: the loader dies at boot,
  rc=-11 after `FATAL: worker init failed: load step37: multi-head MTP requires
  embedded dense canonical blocks and matching loaded heads` (`h2-fatal-receipt.txt`).
  With MEMRA_LOAD_MTP=1 the embedded chain caps 3->2 but the loaded heads then
  mismatch. HEADS is effectively {1, 3} here.
- These medians (125-150 tok/s) sit far above the 08-28 92-94 tok/s rows: same prompt,
  same instrument arithmetic, different binary (main-merge lane tip vs 553a072471+W8
  patch). That delta is the merged main's own gain, not a policy effect; it is worth
  its own confirmation cell before any product claim.
- Known instrument defects, banked as-is: the orchestrator's success path never wrote
  cells to `taskB-cells.jsonl` (only BOOT_FAILs landed there) and the final-table step
  crashed on that empty dict (run.log ends rc=1). The CELL lines in `run.log` plus the
  per-boot server logs are the primary receipts; the table above is built from them.
  Run 2 reused boot tags, overwriting a few run-1 server logs; run 1 is corroboration
  only. Teardown of ~100 GB VRAM can outlive a 120 s pgrep window (one 08-28 cell lost
  to that before the orchestrator learned to wait; wait for pgrep-clear before boot).
