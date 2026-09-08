# GLM-5.3-Flash DFlash2 on the symmetric TP-2 walk

Status: eager implementation and pair cell prepared; pair qualification pending. The owner corrected the
contract after stage 1: use the decode split's per-row expert program, not F16
grouped prime, and implement the row/state interfaces in this lane. The original
inspection below is retained as the record of the starting constraints.

Stage 2 uses t-row KDA and MLA/DSA with rows-exact projections, rank-local replay
stashes, the decode expert partial program per row, a t-row one-shot reduction,
and unfused HC post for t>1. Anchor-only rounds use the existing symmetric t=1
layer. Root drafts/accepts and broadcasts IDs; rollback validates both ranks before
mutation and synchronizes both streams before returning. Request preconditions
fail closed to plain; an error after mutation fails the request. Graph capture is
not part of this stage.

Corrected pair gates: O1 is K=0 versus plain TP greedy tape, exact. O2 compares
TP spec K=6 against PP spec K=6 using per-row argmax and a declared numeric band,
plus accepted-token sequence identity over 160 output tokens. O3 compares t-row
verify target logits against sequential plain TP logits on identical input IDs,
bit for bit. None of these pair gates can run on the single-card tune host.

Engine inspection pin: `dcfeab7c738912a150ebbfea277112724bb99de4` (`origin/main`,
including #325). All source line numbers below refer to this pin. Branch:
`lane/glm5-tp2-dflash-20260909`. Session:
https://claude.ai/code/session_01TFyR32RLUiSejCgrPm5nNj

## Stage-1 inspection: the three hardest constraints

1. **Grouped prime is not the decode-exact expert program.**
   `hybrid_forward.rs:16296` explicitly calls it a band class and declines
   `MEMRA_MOE_GATE`, the sequential byte-identity oracle. At `:16484` and `:16546`
   it converts activations with `moe_f16g_act` and uses `moe_f16_grouped` for the
   gate/up/down contractions. The decode half at `:15235` uses the routed Q8
   rows program instead. The private tpprime3 receipt says the **16 generated
   IDs** match with grouped prime on/off. It does not establish bit equality
   of the intermediate rows, nor of a verify continuation. Reusing that receipt
   as proof of exact verification would change the contract without evidence.
2. **The symmetric layer API is one token and has no verify checkpoint output.**
   `sym_layer_step` (`hybrid_forward.rs:3236`) hardcodes `t=1` and decode KDA;
   `HyperDecodeWs` (`hyper.rs:770`) allocates one token's gates, mix matrix and
   residual. `kda_tp_partials_sym` (`glm5_tp.rs:1330`) accepts a row count but
   passes `KdaStash::None` and returns only partials. A wider buffer alone cannot
   supply the pre-round recurrent state and replay inputs needed after rejection.
3. **TP/PP equality is an oracle requirement, not a construction proof.**
   Symmetric KDA `wo` and the split expert down projection reduce half-width
   dots, whereas PP computes full-width dots. `glm5_tp.rs:1328` and
   `hybrid_forward.rs:15351` explicitly disclaim unsharded bit identity.
   Deterministic drafting and acceptance cannot make different target logits
   equal. A fixed 160-token TP/PP tape may pass; this inspection does not predict
   failure. It also cannot promise that equality for arbitrary prompts.

The owner corrected the contract after this inspection: these interfaces are
this lane's work, the expert stage must retain the decode split program, and
O2 is a TP/PP class comparison. Stage 2 implements those choices. No F16 grouped
prime enters verification and no TP/PP bitwise-logit claim is made.

## Implemented walk

- `HybridModel::glm5_tp_verify_symmetric` owns the eager rank pair for each
  verification call. It uses rows-exact HC pre, batched KDA and MLA/DSA, a
  per-token invocation of the existing split expert partial program, and t-row
  AR followed by unfused HC post. It emits root tap rows after each full layer.
- `kda_tp_partials_sym_verify` snapshots each rank and returns its partials plus
  `KdaRowsStash`. The ordinary t=1 entry is unchanged. The MLA symmetric helper
  threads `rows_exact` through projections, indexer, attention and output head
  shards. Indexer-less shards decline before speculative session creation.
- `glm5_tp_spec::Commit` validates the accepted prefix and every rank stash before
  restoration. The outer rollback validates all layers and both latent replicas
  before mutation, restores recurrent state through each owning rank, truncates
  both KV/DSA cursors, and drains both streams before publishing the new position.
- Root retains the existing DFlash2 block forward, selector, PMIN, FR-Spec and
  sampler counters. The row IDs and positions are broadcast using existing
  transport. No drafter copy executes on the peer.
- Worker admission keys on loaded shards and returns plain for unsupported
  compositions before draining the prompt. The existing K policy and K=0 plain
  route remain intact. Errors after mutation fail the request; they do not
  silently resume from a partly advanced cache.
- `MEMRA_GLM5_SPEC_TP` stays OFF, with decide-by 2026-09-23. A separate verify
  graph implementation remains deferred. Its default-OFF eligibility row is in
  section 4 of `docs/FLAGS.md`.

The exactness argument is the reuse of the TP decode arithmetic per row and
rank-local replay, not a completed pair receipt. O1 and O3 require the pair and
remain pending. See `research/glm5-tp2-dflash-20260909/PAIR-CELL.md` in private Darklanes and
`research/glm5-tp2-dflash-20260909/run-pair-cell.sh` in private Darklanes for the actual oracle and A/B entry point.

## Inputs read without rerunning the experiments

Private Darklanes sources (kept in the private repository):

- `wt-ship1m/research/glm5-1m-b200-ship-20260906/receipts/dflash2-20260908/`:
  `PP2-RESULTS.md`, `PMIN0-RESULTS.md`, `PROFILE-RESULTS.md`.
- `research/glm5-b200-mint-20260904/LANE.md`: tpprime3, the symmetric walk and
  graph history, and tpwalk14-17. The profile makes target verification the
  optimization target; changing the confidence threshold is not this lane.
- `agent-knowledge/gpu/models/glm53-flash.md` and the spec/TP law and trap rows:
  admission must use loaded model state, engagement receipts print after
  admission succeeds, and a one-card fixture is not a symmetric pair gate.

Engine catalog: `docs/FLAGS.md`, particularly the GLM spec, PMIN/PMIN0,
verify-batch, verify-graph, DFlash and TP rows. Earlier documentation describing
column-parallel TP verification predates the symmetric row-parallel walk.

## Refusal map

| Source | Actual boundary at the pin |
| --- | --- |
| `crates/memra-server/src/worker.rs:4844` | `glm5_tp_serve_boot_verdict` refuses GLM spec and dspark with TP, requires explicit `MEMRA_SERVE_SPEC=0`, and refuses hyper batching. The existing spec-TP door does not lift this worker gate. |
| `crates/memra-server/src/worker.rs:17930` | `glm5_spec_capable` independently excludes TP using the environment. Replace this eligibility check with loaded topology plus the new admission result, so set/load/unset cannot bypass it. |
| `crates/memra-engine/src/glm_spec.rs:2521` | Session admission checks actual layer sharding. `MEMRA_GLM5_SPEC_TP=1` already lifts this particular refusal; batched verification remains required. |
| `crates/memra-engine/src/glm5_tp.rs:524` | The current refusal array contains `MEMRA_HC_FUSED_PRE`, `MEMRA_HC_DECODE_WS`, and `MEMRA_MLA_DECODE_SPLIT`. It does **not** contain `MEMRA_GLM5_SPEC_TP`. There is no spec-TP array entry to delete. |
| `crates/memra-engine/src/glm_spec.rs:1238` | The existing verifier enters the root-owned range walk or PP ranges, not `sym_layer_step`. Its TP mixer branches provide useful older composition primitives, not a symmetric verify executor. |

The door must remain default OFF. A future stage-2 commit must update the existing
three-column section-4 row at `docs/FLAGS.md:141`, with `decide-by: 2026-09-23`,
both arms, rollback, the admitted loaded topology, and actual gate receipts.
Unrelated dspark and hyper-batch refusals remain independent.

## Required row walk, by operation

The actual row count is `t = drafts.len() + 1`, not always the selected policy
`K+1`: PMIN may shorten the proposal. With K=6, the walker must support every
width 1 through 7, including the zero-draft anchor-only case.

| Operation | Existing primitive and required change |
| --- | --- |
| Entry and HC | Embed `[anchor, drafts...]` once on root (`glm_spec.rs:1893`), distribute IDs/positions with `tp_transport.rs:697` (`fanout_i32`) and the entry residual with `:625` (`fanout_f32`). Both ranks hold `[t, streams, n_embd]`. Use `hyper.rs:406` `pre_exact` and `:991` onward `post` with each rank's `Glm5TpGlue`; never reinterpret one-token gates as row gates. |
| KDA | Extract a verify-capable counterpart to `glm5_tp.rs:1330`: each rank snapshots its own SSM and captures `KdaStash::Rows`, uses the PP verifier's rows-exact projections and sequential scan at t rows, returns its `wo` partial plus stash. Reuse the stash/replay contract at `:1408`, `:1415`, `:1444`. The existing root-owned `kda_tp_verify_rows` already captures it, but fans out one root input and reduces back to root. |
| MLA | `hybrid_forward.rs:11450` already runs head shards on both ranks with replicated latent planes, but hardcodes `rows_exact=false` and ordinary `matmul` for `wo`. A symmetric verify counterpart must carry `rows_exact=true` through pre/attention/`wo` and return rank partials or row-reduced outputs. Existing `mla_tp_attn_cached` at `:11536` supplies the older rows-exact TP composition. |
| DSA | `hybrid_forward.rs:10707` `mla_kpool_indices_rows` handles per-query positions, state append and pool construction. Run all t causal queries on both replicas for the first exact path. Do not broadcast one decode selection to every row. Query splitting is an additional composition, not needed for eager correctness. |
| MLA TC prime | `hybrid_forward.rs:9952` admits TC prefill only at prefill widths and with `!rows_exact` (`:9975`). Its head-shard prime receipt does not admit TC verification at t=7. Keep the PP rows-exact attention classes here. |
| Routed experts | The current grouped entry at `hybrid_forward.rs:15374` requires `prefill && t > MOE_DEV_MAX_T`. Its implementation at `:16266` consumes root input and returns a root-reduced tensor. It does not expose both rank partials to the symmetric post. First expose an exact t-row partial interface using `moe_split_rank_half` (`:15235`) or the slot program (`:15145`); preserve row routing, macro folds, slot order and the shared-expert placement of `moe_ffn_glm5_tp_split_sym` (`:15534`). A grouped tensor-core alternative needs separate numeric admission. |
| Dense layers | Preserve the three dense layers' existing computation and fan-out; both residual replicas must receive the same full branch result. |
| Add/post | General all-reduce can reduce `t*n_embd` floats. `tp_ar.rs:131` selects one block at n<=8192 and more blocks above it; **8192 is not a capacity limit**. At t=7 the mixer output is 28672 floats. The fused `all_reduce_1stage_hcpost` (`:532`) and CUDA kernel (`cu/tp_ar.cu:280`) have only hc,d and one token's post/comb indexing. Use general AR plus t-row HC post initially, or add a row-indexed fused interface. Passing `d=t*n_embd` to the existing fused API gives the wrong layout. |
| Head and taps | Root contracts completed layer rows into the DFlash tap sink (`glm_spec.rs:1685`), then runs collapse/norm/full-vocabulary target head (`:1842`). A symmetric walk omitting these taps can verify output but cannot supply the next DFlash round's context. |

Any new cross-file CUDA twin of the unfused `-fmad=false` arithmetic must retain
explicit `__fadd_rn` and `__fmul_rn`. The existing fused HC crossing already does
so at `cu/tp_ar.cu:311-315`. This preserves its local arithmetic; it does not
restore the full-K reduction order changed by tensor splitting.

## Drafter, acceptance and rollback ownership

Run DFlash2 only on root. The unsplit TP model already keeps the target embedding
and head on that engine; `glm5_head_engine` (`glm_spec.rs:2063`) selects root when
there is no PP fence. Reuse the loaded full drafter and its one KV ring. Replicating
or splitting drafter compute adds no correctness benefit and risks diverging RNG.

Keep `glm5_dflash_round_drafts` (`glm_spec.rs:4366`) unchanged: one block forward,
FR-Spec rank-to-token mapping and selector-q accounting, truncate to K, then PMIN
(`:4499`). Broadcast the resulting token IDs and row positions with the existing
integer transport before either rank starts verification. Sample/accept on root
once using the existing session Philox counters and target filtering.

The root coordinator owns a single decision `(round, t, accepted=j, keep=j+1,
bonus, pos0)`. Both rank state updates must complete before the next proposal can
consume context. The bonus is the next anchor; it is not an additional committed
KV row in this round (`glm_spec.rs:4215-4268`).

- KDA: retain pre-round state plus raw conv/scan rows on **each rank**. Replay
  exactly `keep` rows from that rank's snapshot on partial accept. Do not copy
  canonical root state over rank-specific state. Full accept retains live state.
- MLA: `glm5_verify_rollback` and `glm5_rollback_layer` (`glm_spec.rs:2110`,
  `:2151`) already restore root and peer `len`, device `len_d`, and truncate
  index pool keys (`:2193`). Validate all rank stash/plane shapes before writing.
- DSA: truncate each replica's pool-ready boundary with its latent boundary.
  Include the ring/pool-boundary cases in the rejected-tail continuation oracle.
- Drafter: keep only the first `keep` tap rows. `forward_round` does not advance
  its persistent KV length, so there is no speculative draft-block rollback
  (`glm_spec.rs:4245`).
- Stream ordering: publish completion from both peer and root streams before
  declaring rollback complete, releasing buffers, or admitting the next round.

Plain fallback must be decided before mutable session work. The worker currently
drains the prefill queue and fails a prime error loudly (`worker.rs:24139` onward).
An error after one rank advances cannot safely fall back to plain by catching
`Err`: it needs a complete pre-round transaction restore, or the request fails.
Use a shared admission verdict before draining/priming for unsupported topology,
missing drafter/rewrites, row geometry, capacity and incompatible doors. Runtime
CUDA failure is not an eligibility fallback.

## Graphs follow eager exactness

`glm5_tp_sym_graph.rs:278` sizes the residual as `streams*n_embd`, allocates one
peer position (`:298`), and owns one `HyperDecodeWs` per rank. `walk_token` runs
the first token eagerly (`:416`) before capture. Its KDA pieces and MLA
pre/middle/post split cannot be reused with wider pointers.

A later graph pool needs keys `(range, t, rank, recurrent-buffer phase)` and
stable row-sized residuals, positions, gates, KDA stashes and MLA scratch.
Each new t runs eagerly first. Capture both ranks' pieces using the decode
pair-capture approach, preserve the eager host-geometry MLA middle initially,
and rewind **both ranks** after capture/warmup before replaying real work.

The PP verify graph pool is already keyed by `(lo, hi, t)` (`glm_spec.rs:851`),
but replay updates only canonical latent/recurrent bookkeeping (`:868-885`).
It is not a TP graph state implementation. Do not pass it to the new symmetric
verify path until its checkpoint ownership includes both ranks. Grouped prime's
host CSR construction also cannot simply be enclosed in CUDA capture.

## Stage-1 qualification proposal (superseded by O1-O3 above)

Construction claims available after the required changes: one authoritative
draft/accept decision; identical broadcast IDs; rank-ordered AR results equal
on both ranks; `keep=j+1` boundaries; rank-local snapshot/replay ownership; K=0
dispatches the existing plain TP program before creating a speculative session.
These claims do not establish target-logit equality across TP and PP.

Required tests, not implemented or run in this document-only stage:

1. CPU fake executor: t=1..7, every j=0..t-1, root/peer KV and DSA cursor equality,
   rank-distinct KDA states, full accept, early rejection, invalid keep, missing
   peer stash, and refusal before mutation. Assert actual production bookkeeping
   helpers, not a duplicate arithmetic formula in the test.
2. CPU admission: door off/on, actual shard topology, set/load/unset simulation,
   no drafter, incompatible flags, unsupported widths, and K=0 plain routing.
3. Single-card ignored GPU tests: local rows/slot arithmetic and stash replay.
   Existing `glm5-tp-gate` same-device symmetric arm deliberately declines
   (`bin/glm5_tp_gate.rs:1060`); it cannot qualify the real symmetric executor.
4. Pair ignored GPU oracle 1: same pinned artifact, drafter, tokenizer, prompt
   IDs and numeric recipe; TP-2 DFlash K=6 and PP-2 DFlash K=6; 160 greedy output
   IDs must match. Record draft/accept traces and first differing target row on
   failure. Add forced rejection positions, not only naturally accepted rounds.
5. Pair ignored GPU oracle 2: TP spec door ON with `MEMRA_SPEC_K=0` versus plain
   TP-2; 160 IDs must match, zero spec rounds, same first-token eager behavior.

Future pair cell specification: c1, contexts 32768 and 131072, 512 output cap,
three fresh-boot cycles ordered TP-plain/TP-spec/PP-spec, then
TP-spec/PP-spec/TP-plain, then PP-spec/TP-plain/TP-spec. Freeze prompt bodies;
omit sampling parameters for the scored vendor-default requests. Keep PMIN=.7,
normal K policy, the same FR-Spec artifact and nonexperimental latent cache.
Require full SSE completion, usage, model/binary/artifact identity, K>0 and
accepted/drafted/round log reconciliation for both spec arms. Report TTFT,
counted decode interval and full wall separately; reject loop-inflated rows.

Stage 3 supplies the executable pair cell and its hash-checked configuration.
The earlier proposal in this section is superseded by the corrected O1-O3
contract and the detailed `research/glm5-tp2-dflash-20260909/PAIR-CELL.md` in private Darklanes runbook.

## Stage-1 evidence

Stage 1: direct source and existing receipt inspection only. No cargo, gate,
server, benchmark or CI invocation on the local rig or any GPU host. No
production host access. No new equality or performance measurement.

The original document-only commit was `395136b1a`. Eager implementation landed
in `aebbcd422`; tests, pair-cell preparation and remote validation follow in
the next stage. Validation receipts live in this lane's `receipts/` directory.

## Repository boundary

Engine base: `dcfeab7c738912a150ebbfea277112724bb99de4`. Private cell base:
`dd5b04cd1188fb2aada272c32a6465e569d63fff`. The exact numerical launch recipes,
HTTP cell driver and scheduling inputs are tracked in Darklanes on
`lane/glm5-tp2-dflash-cell-20260909`, under the same research namespace. They
are private serving material under `tools/public-boundary-policy.toml`. The
ignored engine oracle harness remains in Memra.
