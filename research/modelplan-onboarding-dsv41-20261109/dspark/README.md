# DeepSeek-V4.1-Flash onboarding: DSpark drafting on the GPU route (2026-11-10)

Verdict so far: the checkpoint's DSpark draft head runs on the rank group as stored and sits on the
vendor's own backend floor on every tensor (28 of 28 bands). A decode call can now carry several
tokens, each row run as the one-token decode program of its position, and it gives the bytes of
feeding the tokens one call at a time: 1,729 of 1,729 rows on the full trunk (blocks of 2 to 6 rows,
443 of them rejected and dropped, across the 128-token window, at 41k and 64k context). Drafted
greedy generation equals plain greedy generation on tokens and on the logits behind every token, 15
of 15 cases. memra's own drafts are accepted at 4.8 to 5.8 tokens per verification step on code and
structured text and 1.7 to 2.5 on prose. What a verification step costs is measured per row count;
a kernel switch at 5 rows made a 6-row call 3.7x a one-row call and is fixed (2.1x). The served
route carries the draft loop behind `MEMRA_DSV41_DRAFT=1`, off by default: its served gates and its
A/B have not run (the cards went to the multi-card layout work first).

Hardware: one RTX PRO 6000 Blackwell Max-Q at a time (sm_120a, 96 GB), 280 W cap for the memra runs
and 250 W for the vendor oracle's card, CUDA 13.4, rustc 1.97.1, 16 host cores. Every time below is
from a single untimed lease on a box that ran other work: the times order the effects, they are not
a benchmark.

Commits on `dsv41/dspark-20261110` (base: the serve lane's head `8906b349c6` fast-forwarded to the
4-rank lane's merge `8c8c5544b9`, which already carries the perf lane's `29aeeee39f`; a direct
merge of the perf head onto the serve head conflicts in `dsv41_gpu.rs` and `tp.rs`):

| commit | what |
|---|---|
| `68a3833b5f` | one-rank MoE takes the cached-experts path when the layer holds no expert on the card (the one-card residency did not run on the merge) |
| `2f7f02096d` | several-token verification calls with commit, the draft head (`dsv41_gpu::draft`), `dsv41-spec` |
| `6e97238260` | `rows-time`, a short all-resident trunk, `Runner::discard` |
| `3d3dbc769a` | small-M GEMM bodies take up to 8 rows |
| `7ecafc9cd4` | depth pricing: `RowCost`, `price_depth`, `DraftGate` |
| `357a7e4177` | the served draft loop behind `MEMRA_DSV41_DRAFT=1` |

## What the head reads, from the vendor code

`inference/model.py` implements only `Transformer.forward_spec`; nothing in the vendor code calls
it. Read off that code:

- `main_hidden` `[b, s, 3 * dim]`: for each trunk position of a forward, the mean over the `hc`
  stream copies at the INPUT of each target layer (config `dspark_target_layer_ids` = 37, 38, 39:
  after the layer's Engram, before its attention), the three means side by side.
- `input_ids` `[b]`: the token the trunk chose from the forward's last position.
- `start_pos`: the forward's first trunk position.
- `forward_spec` at position 0 (the prefill) only seeds every stage's window ring with the prompt's
  rows (`kv_norm(wkv(main_norm(main_proj(main_hidden))))`, RoPE at the position, fp8 per-32
  quantize-dequantize; the last 128 positions, position p at slot p % 128) and returns nothing.
- Past it the code takes one trunk position per call: the stage KV row of `start_pos` goes into its
  slot, and the block is `[input token, noise x 4]` (block 5, noise id 128799) at RoPE positions
  `start_pos + 1 ..`. Every block row attends ring slots `0 .. min(128, start_pos + 1)` in slot
  order and then all five block rows: there is no causal mask inside the block.
- Row i's logits are the trunk's head on the last stage's collapsed stream (through the stage's own
  norm) plus the Markov head's bias of the token before it (rank 256 embed and head), so the picks
  run in order; row i's token sits at position `start_pos + 2 + i`. The confidence head scores each
  row from its collapsed stream and its Markov embedding.

The stages are trunk blocks with window-only attention and their own MoE (128 MXFP4 experts, top 3,
plus the shared FP8 expert).

## The loop in memra

`crates/memra-engine/src/dsv41_gpu/draft.rs`, `decode.rs`:

- **Draft.** The three `mtp.*` stages load as stored through the trunk's layer loader (MXFP4
  experts on the card, FP8 dense, partitioned by `Part` like a trunk layer), with `main_proj`,
  the norms, the Markov head's vocabulary rows split like the trunk head's, and the confidence
  row. The ring write is its own step, run for every position the trunk commits
  (`Runner::draft_commit`), as in memra's DSv4 drafter; a draft never mutates state. The draft is
  the head's argmax, a one-hot proposal, whatever the request's sampler.
- **Verify.** `Runner::forward` takes a call of several tokens past the prefill. Row i runs the
  one-token program of position `sp + i`: its window row is the decode ring order (oldest slot
  first) read from `[the ring as it stood | the call's own KV rows]`
  (`memra_v41_attn_idx_rows`), a ratio-2 compressor pools the pairs that complete in order with
  the open slot's row first when the call starts on an odd position, the indexer scores each row
  over its own visible keys, and the Engram hash sees the call's tokens. The call writes no
  carried state.
- **Commit.** `Runner::commit(keep)` copies the kept rows into the rings and the compressor's open
  pair, sets the compressed row counts to the kept length and cuts the token history;
  `Runner::discard` drops the call whole. A rejected row therefore never reaches the state, and
  the draft rings advance for the kept positions only.
- **Accept.** The caller chooses each row's token as it chooses a plain step's and keeps the drafts
  that equal those choices from the front (`draft::accept_walk`). Greedy: the argmax, lowest id
  among equal maxima. Sampled (the serve route): the position-keyed seeded draw and the penalties
  window of the DSv4 route, row by row, so a draft is accepted exactly when it equals the draw at
  its position.
- A draft right after the prefill reads the rings the prefill seeded: the same rows the vendor's
  `forward_spec` at position 0 writes.

### The correctness gate, stated before any speed

The vendor program is shape-sensitive (the same prefix in a longer forward moves its logits) and
below the 128-token window its prefill and decode are different programs. memra's kernels act per
row, so the gate does not need the decode lane's near-tie criterion: **a verification row must
equal, byte for byte, the row of feeding the same tokens one decode call at a time**, logits and
draft-head inputs, whatever was rejected before it. Drafted greedy output is then the plain greedy
output by construction, and the generation gate checks it end to end on tokens and on the logits
behind every token. The one-token decode program itself is gated against the vendor in `../decode`.

## Vendor tapes of the draft layers

`oracle/spec_tail.py` is the tail of a copy of the lane's oracle (`oracle/make_spec_oracle.py`: the
run box's `run_oracle.py` up to its case loop, plus a `--replay` argument; the result's sha256 is in
`receipts/tapes/manifest.json`); the shared oracle is untouched. It runs the trunk
greedily as the oracle does, calls `forward_spec` after the prefill and after every decode forward,
and tapes per step: `main_hidden`, `main_x`, each stage's stream, pre-mix, attention input and
output, MoE output and routes, the collapsed last-stage stream, the Markov bias and embeddings, the
block's logits (top-32 and log-sum-exp on every step, full on the first four), the draft and the
confidence scores; plus the prompt's `main_hidden` and the rings after the seed and at the end. The
drafts never feed the trunk, so each tape also says how many drafts the plain greedy generation
agrees with.

12 of the lane's chat and needle cases, 339 steps, 1,695 draft tokens, vendor GPU program
(`receipts/logs/oracle-tapes.log`, `receipts/tapes-raw/*/meta.json`). Drafts the vendor's own
greedy generation agrees with, mean per step of 5: `copy` 4.92, `code` 4.86, `code_pal` 4.26, `gsm`
4.00, `sys_json` 3.89, `gsm_think` 2.80, `story` 0.66, `hebrew_para` 0.62.

The floor is the same vendor code on the CPU backend over the tapes' own inputs (`--replay`: no
trunk, `forward_spec` alone from the taped `main_hidden` and tokens;
`receipts/logs/replay-cpu.log`).

## Gates

### Layer parity on the tapes' inputs: 28 of 28

`dsv41-spec draft-tape` seeds the rings from a tape's prompt rows, then per step writes the step's
ring row and drafts from its token, and dumps memra's tensors under the tape's names.
`oracle/draft_cmp.py` compares a dump with the vendor GPU tape (the same script for memra and for
the CPU floor); `oracle/draft_gate.py` applies the lane's rule: a tensor passes at twice the floor's
median relative L2 plus 1e-4, routing within one point of the floor's expert-set agreement, the
distribution at twice the floor's mean KL and at most one point (or two binomial standard errors)
under its draft-token agreement, the confidence at twice the floor's mean absolute difference.
Rows after a pick that differs are left out of the distribution rows (a row's Markov bias depends
on the pick before it). `receipts/gate/draft-gate.md`, pooled over the 12 tapes:

| metric | memra | vendor CPU floor | band |
|---|---|---|---|
| ring rows written by the case (3,969), median rel L2 | 0 (3,901 byte-equal) | 0 (3,884 byte-equal) | 1.0e-4 |
| `main_x`, median rel L2 | 3.1e-9 | 0 | 1.0e-4 |
| stage 1 stream | 0.0229 | 0.0228 | 0.0456 |
| stage 2 stream | 0.0479 | 0.0481 | 0.0963 |
| stage 3 stream | 0.0665 | 0.0666 | 0.1333 |
| collapsed last-stage stream | 0.0687 | 0.0686 | 0.1372 |
| expert-set agreement, stages 1 / 2 / 3 | 0.998 / 0.983 / 0.980 | 0.998 / 0.984 / 0.979 | floor - 0.01 |
| Markov bias, median rel L2 | 1.1e-7 | 3.6e-8 | 1.0e-4 |
| draft tokens equal to the vendor GPU's (1,695) | 95.6% | 95.3% | >= 94.3% |
| the same, rows with the same earlier picks (1,634) | 97.1% | 97.3% | >= 96.3% |
| mean KL, top-32 estimator (1,634 rows) | 6.2e-3 | 6.5e-3 | 1.30e-2 |
| mean KL, full logits (191 rows) | 8.0e-3 | 1.26e-2 | 2.51e-2 |
| confidence sigmoid, mean abs difference | 5.3e-3 | 5.5e-3 | 1.12e-2 |

The other 15 rows (attention and MoE outputs per stage, pre-mixes) are in the receipt. The stream
distance grows with the stage for the floor exactly as for memra: the vendor's own two backends
part by 2% after one draft stage. The dump is the same with either small-M limit (below).

### The verification call is the one-token program: 1,729 of 1,729 rows

`dsv41-spec verify-rows`: arm A feeds the forced tokens one decode call at a time; arm B feeds them
as blocks of 2 to 6 rows, every second block with wrong tokens past a kept prefix and committed at
that prefix; every kept row's logits and taps are compared by hash with arm A's row of the same
position. Full trunk, one-card expert-cache residency (`receipts/out/full-sm8/stdout.jsonl`):

| case | prompt | rows | equal | rejected rows |
|---|---|---|---|---|
| 8 chat cases (`gsm` .. `gsm_think`), below the window | 22 to 68 | 503 | 503 | 139 |
| `cross128`, `cross128odd` (346 forced tokens across positions 128 and 256, even and odd start) | 27, 22 | 692 | 692 | 172 |
| `needle` | 3,761 | 346 | 346 | 86 |
| `decode32k_he_summary` | 41,098 | 96 | 96 | 24 |
| `decode64k_en_continue` | 64,443 | 92 | 92 | 22 |

Also 1,058 of 1,058 on a 9-layer trunk with every weight on the card, with each small-M limit
(`receipts/out/res9-sm4`, `res9-sm8`). `Ctx::window_row_verify` is the host form of the index
kernel, with a unit test against the one-token rows.

### Drafted greedy equals plain greedy: 15 of 15

`dsv41-spec gen`: plain greedy generation, then drafted greedy generation of the same prompt at
full depth; the token lists and the hash of the logits row behind every token must be equal. The 13
chat and needle cases and the two long decode cases, 96 to 160 tokens each: all identical
(`receipts/out/full-sm8`). Nine more runs at depths 1 to 5 are in `receipts/out/smoke-1` and
`scale-1`.

## Acceptance: accepted tokens per verification step

memra's own trunk and drafts, greedy, every draft verified (depth 5), same run:

| case | rounds | drafts accepted per round | tokens per round | rounds by accepted drafts 0..5 |
|---|---|---|---|---|
| `copy` | 5 | 4.80 | 5.80 | 0 0 0 0 1 4 |
| `code` | 8 | 4.12 | 5.12 | 0 1 1 0 0 6 |
| `code_pal` | 11 | 3.82 | 4.82 | 1 2 0 0 0 8 |
| `gsm_think` | 12 | 2.83 | 3.83 | 2 1 1 4 1 3 |
| `sys_json` | 4 | 2.50 | 3.50 | 1 0 1 1 0 1 |
| `gsm` | 7 | 1.71 | 2.71 | 1 3 2 0 0 1 |
| `decode64k_en_continue` (64k context) | 30 | 1.50 | 2.50 | 8 9 6 5 1 1 |
| `story` | 62 | 1.02 | 2.02 | 24 21 12 2 3 0 |
| `decode32k_he_summary` (41k context) | 48 | 0.98 | 1.98 | 22 11 11 2 2 0 |
| `hebrew_para` | 96 | 0.66 | 1.66 | 53 25 16 2 0 0 |

The benchmark set's workloads (`decode_c1`, `chat_c1`, `conc8`) run through the server and have not
been measured: they are document continuations and essays, so the prose rows are the expectation.

The confidence head is calibrated. On the vendor tapes, among drafts whose earlier drafts held
(959), the share that held by sigmoid of the score: 0.99 at 0.9 and up (562), 0.93 at 0.8 to 0.9,
0.64 at 0.7 to 0.8, 0.59 and 0.63 at 0.6 and 0.5, 0.50 at 0.4 to 0.5, 0.42 at 0.3, 0.23 at 0.2,
0.17 at 0.1 to 0.2. Mean sigmoid of rows 1 to 5: `code` 0.99 .. 0.95, `gsm_think` 0.89 .. 0.63,
`story` 0.49 .. 0.16.

For comparison, what the two other engines verify (read from their sources and recipes, not run):
both draft `[bonus, noise x 4]` and verify 6 rows, at most 6 tokens a step. SGLang's documented
recipe is `--speculative-algorithm DSPARK --speculative-dspark-block-size 5` with a static
verification width by default; its ragged modes pick per-request widths from the sorted survival
and a cost table. vLLM's recipe is `num_speculative_tokens 5`, `draft_sample_method probabilistic`,
`rejection_sample_method block`, `enable_adaptive_verification true` (the previous step's
confidences against start-up cost tables). Neither uses a scalar confidence threshold. Their
published acceptance on this model: a synthetic acceptance length of 3.51 in vLLM's throughput
recipe, 1.67 to 2.40 measured on sm_120 in a vLLM issue, 3.0 tokens a step at block 2 in an SGLang
issue.

## What a verification step costs

`dsv41-spec rows-time`: one-token calls, then the same block of 2 to 6 rows eleven times at one
position, dropped whole after each run (`Runner::discard`); the repeats are the rows' compute with
nothing fetched. A 9-layer trunk, the head and the draft head with every weight on the card
(`receipts/out/res9-sm4`, `res9-sm8`), ms per call, context 27 / 3,761 / 18,343 tokens:

| rows | small-M limit 4 (before `3d3dbc769a`) | small-M limit 8 |
|---|---|---|
| 1 | 5.63 / 6.25 / 6.78 | 5.68 / 6.24 / 6.83 |
| 2 | 7.10 / 7.65 / 8.14 | 6.99 / 7.69 / 8.17 |
| 3 | 8.25 / 8.68 / 9.24 | 8.20 / 8.93 / 9.29 |
| 4 | 9.25 / 10.17 / 10.34 | 9.13 / 10.04 / 10.64 |
| 5 | 19.60 / 20.07 / 20.90 | 11.20 / 11.87 / 12.69 |
| 6 | 20.82 / 20.74 / 21.42 | 12.06 / 12.60 / 13.49 |

- No kernel launches per row: every operation of a block takes the rows as a grid dimension, about
  45 to 60 launches a layer a call for one row or six. The verification call adds one copy a layer.
- The step at 5 rows was a kernel family switch: `V41_SM_MAX` was 4, so a call of 5 rows or more
  left the decode ("small-M") GEMM bodies for the pipelined 64-row tiles. The block-scaled and bf16
  small-M bodies already place the rows on fragment rows 0..7; `3d3dbc769a` raises the limit to 8
  and runs the fp32 FMA small chain (4 rows by construction) as two launches for 5 to 8 rows. Bytes
  unchanged: every gate above ran on the patched binary, and the 9-layer gates on both.
- What a row costs after that, about 0.13 ms a row a layer on one card, matches the routed
  experts' bytes: a row routes to 6 experts of 17.7 MB (106 MB a layer, about 830 GB/s), while
  the dense weights are read once a launch whatever the rows. Six rows read up to 36 experts a
  layer. The grouped dispatch already runs them as one launch per projection. The term divides by
  the ranks when the experts are split across cards, which is why the row-cost table is an input
  of the pricing and not a constant. (A block of one token repeated is only 10% cheaper, but its
  rows sit at different positions and need not route alike, so that arm does not isolate the
  experts; the byte count does.)
- The head: for 5 to 8 rows the fp32 head GEMM is two passes over the 1.3 GB head, and every row's
  logits are downloaded (517 KB a row). `Runner::argmax_rows` (`memra_v41_argmax_rows`) reduces
  greedy rows on the device instead; it compiles and `dsv41-spec gen` checks it row by row against
  the host argmax, but it has not run on a card and no route uses it yet. The second pass stays
  until the FMA small chain takes 8 rows in one launch.

In the one-card expert-cache residency a block also fetches: in the full-trunk run the cache held
3,040 of 15,360 experts beside the draft head (a 24 GB reserve for the long prefills), 23% of the
run's lookups missed (18.8 MB of H2D each), and a 6-row block asks for up to 36 experts a layer.
One-token step 31 to 44 ms on the chat and long cases, draft 4.4 ms, 6-row verification 83 to 184
ms (`receipts/out/full-sm8`): at full depth `copy` ran 1.47x, `code` 1.22x,
`code_pal` 1.21x, `gsm_think` 1.09x of plain greedy, and prose 0.59x to 0.70x. So on this
residency the depth has to be priced, and prose mostly should not draft.

## Pricing the depth (code and unit tests; not measured on a card)

`draft::RowCost` is what a decode call costs by its rows on the residency that runs it, filled by
the caller from its own steps (an exponential average per row count; a row count nobody ran is
read off the line through its measured neighbours). `draft::price_depth` verifies the depth with
the most expected tokens per millisecond: `1 + sum of the survival of each draft` (the product of
the confidence sigmoids up to it) over the call's cost at that depth plus one draft.
`draft::DraftGate` stops a sequence drafting for a stretch (8 steps, doubling to 64) while its
drafted steps yield less than one-token steps would, and tries again. The route keeps the table
(`dsv41_serve.rs`: seeded by the warm-up's one-token call, full block and draft, then corrected by
every step whose calls had the same rows). With the 9-layer resident table and a 2 ms draft the
rule verifies 5 of 5 near-certain drafts and 1 of the mean prose block. The averaging weights and
the stretch lengths are first values, not tuned: they are the first thing to measure when the
route gets a card.

Arithmetic, not a measurement: on a layout whose one-token step is T1 with the experts on one card
per layer, a 6-row call is about 2.1 T1; with a draft at 0.2 to 0.4 T1 that is about 2x plain
decode on code-like text at full depth and a loss on prose at any fixed depth, since the draft
itself costs a fraction of a step. Prose gains only if the draft gets cheaper (its five picks are
five host round trips today) or is skipped when it does not pay.

## The served route

`crates/memra-server/src/dsv41_serve.rs` carries the loop behind `MEMRA_DSV41_DRAFT=1`
(`docs/FLAGS.md`, off by default, decide-by 2026-11-24): the draft head loads beside the trunk, a
request's decode call carries its next token and the priced drafts, the rows' tokens are chosen
with the request's own sampler state, the kept rows are committed, the response's `usage.spec`
counts rounds, drafted and accepted. With the door off the route's calls, choices and memory book
are the ones it had. It compiles and its unit tests pass; it has not served a request on a card.

## What is left

1. The route's gates with drafting on: the chat-gate cells (solo greedy against native, batched,
   streamed, stop strings, a seeded sampled request equal with the door on and off) and
   `memra model verify serve`.
2. Acceptance on the benchmark set's workloads through the server, and the interleaved A/B
   (A B B A, three runs per arm) of the door off against on, decode rate for one request and for
   eight; then the door becomes the default where it wins and is removed.
3. The pricing's first values measured; the row-cost table on the layout the lane settles on.
4. The draft's host round trips (five picks, each a read-back) and the trunk head's second pass:
   a device-side chain of picks, the device argmax's first run on a card, an 8-row FMA body.
5. N ranks: the code paths are the rank group's (the stages go through `block_forward_kind`, the
   head and the Markov head split by vocabulary rows, the picks reduce across ranks on the host),
   but only one rank has run.
6. Batched verification across requests rides the lane's batched decode design; today each call
   runs alone through each block.

## What the draft path needs from a layout

The loop was built on the rank group, where every rank holds a replica of every stream. A layout
that places consecutive layers on different cards has to keep these open:

1. **A decode call of several rows.** `Runner::forward` with more than one token past the prefill:
   each layer's `attn_keys` builds `[ring | the call's rows]` and leaves the rows in
   `LayerState::pend_kv` / `pend_c`; the stream that crosses a stage boundary is `[rows][hc][dim]`
   with its `[rows][hc]` pre-mix, the same two buffers a one-token call hands over, only taller.
2. **Commit and discard per sequence.** `Runner::commit(keep)` and `Runner::discard` reach every
   layer's state (`Ctx::commit_rows`: ring and open pair) and every kv-source group's row count,
   wherever the layer lives; nothing of a call may become state before it.
3. **The taps.** The mean over the stream copies at the INPUT of layers 37, 38 and 39, for every
   row of a decode call and for the last 128 rows of a prefill (`Runner::set_taps`, `Out::taps`,
   bf16 `[rows][3 * dim]`), available where the draft stages run.
4. **A place for the head.** The three stages (7 GB of weights, a ring of 128 KV rows per stage per
   sequence), `main_proj`, the Markov head (132 MB) and the confidence row want the card that
   holds the trunk's head and embedding rows: a draft reads the embedding of its token and of the
   noise id, the trunk head's weights for its five rows, and the taps of the last trunk stage.
5. **The trunk head by rows.** Logits (or the device argmax) of rows `r0 .. r0 + rows` of a call,
   not only of its last row (`Runner::logits`, `Runner::argmax_rows`).
6. **The token history** up to the call's rows for the Engram hash (`Seq::ids`, cut by commit).
7. **A row-cost table per layout.** The route measures its own decode calls by rows
   (`draft::RowCost`); a layout only has to let a step report its time and rows.

## Files

- `oracle/`: `spec_tail.py` (the tape and replay tail), `spec_oracle.sh`, `export_tape.py` (a tape
  as raw arrays), `draft_cmp.py`, `draft_gate.py`.
- `plans/`: the `dsv41-spec` job lists (`full.plan` is the full-trunk gate, `resident9.plan` the
  9-layer run); `$LANE` stands for the run box's working directory.
- `scripts/`: the lease, build and run wrappers, `sm8_patch.py` (the small-M change as a script, as
  it was applied for the second binary), `export_receipts.py`, `serve_bench.sh` (the server A/B
  launcher, not yet run).
- `receipts/`: `out/<run>/stdout.jsonl` (one JSON line per job), `stderr.log`, per-job
  `result.json`, `rounds.tsv` (per round: position, drafts verified and accepted, draft, verify
  and commit ms, the confidence scores), `blocks.tsv`; `gate/` (the two comparisons and the
  verdict); `tapes-raw/`, `tapes-cpu-raw/` (per case: `meta.json`, draft ids, confidences);
  `logs/`; `leases.tsv` (one row per GPU lease); `binaries.sha256`.

```bash
# vendor tapes (one card), then the CPU floor over the same inputs
bash oracle/spec_oracle.sh $LANE/spec/tapes --device cuda --only capital,gsm,code,... --max-new 64 --full-logits 4
python spec_oracle.py CKPT REF CASES $LANE/spec/tapes-cpu --device cpu --replay $LANE/spec/tapes --only ...
python oracle/export_tape.py tapes tapes-raw gsm code ...
# memra: the full-trunk gates, then the 9-layer resident run
MEMRA_FULL_PREC=1 dsv41-spec CKPT --plan plans/full.plan --out out/full
MEMRA_FULL_PREC=1 dsv41-spec CKPT --plan plans/resident9.plan --out out/res9 --layers 9 --resident
# the band
python oracle/draft_cmp.py tapes-raw tapes-cpu-raw floor.json gsm code ...
python oracle/draft_cmp.py tapes-raw out/res9 memra.json gsm code ...
python oracle/draft_gate.py memra.json floor.json draft-gate.json draft-gate.md
```
