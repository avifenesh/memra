# Cross-request verify batching: stage-2 seam

Date: 2026-08-09

Status: design only; no production batchdraft implementation in this lane

## Decision

Implement **verify-first, fixed-K, same-width cohorts**. The first production-shaped increment should
batch up to four greedy Step-3.7-Flash sessions with K=3 on PP2, while retaining today's solo path as
the oracle and fallback. Draft chains stay row-local in that increment; once each row has prepared
its draft, one new B x T target core processes the cohort. Cross-request draft batching is the next
increment, using the same round rendezvous.

This ordering follows the box1 evidence: verify is 93.7% of steady round time, while draft is about
5.8%. It also keeps the first patch bounded. Merely passing 16 concatenated tokens to today's
single-sequence verifier is explicitly rejected by the m-scale result.

## Required control-flow split

The current generator must be factored into a round state machine without changing solo behavior:

```text
per-row turn setup / suffix prime
            |
            v
prepare_spec_round(row)       x B
  snapshot -> K draft steps -> PreparedVerifyRow
            |
            v
verify_spec_batch(rows)
  one target weight walk at M=sum(T_i)
  row-local attention/cache/checkpoint state
            |
            v
commit_spec_round(row, slice) x B
  accept -> rollback/keep -> pending/draft state -> callback
            |
     live rows rendezvous again
```

The safe outer API is a cohort operation, not a public bag of half-mutated sessions. Proposed shape:

```rust
generate_spec_cohort_greedy(
    engine,
    rows: &mut [SpecCohortRow<'_>],
    k: usize,
) -> Result<Vec<SpecCohortResult>, SpecBatchError>
```

The worker selects and disjointly borrows sessions, as its plain batch path already does. The engine
owns turn setup, every prepare/verify/commit cycle and tail restoration so an error cannot leak a
half-published `SpecSession`. Internally, extraction into three private operations makes the batch
boundary testable:

- `SpecTurnState`: the per-row ephemeral state currently held as locals across the monolithic loop,
  including moved cache/scratch/session-tail references, output, pending/base, hidden seeds,
  `DraftGraphCtx`, RNG counters, penalty/grammar state and telemetry.
- `PreparedVerifyRow`: draft ids/statistics, verify tokens, `pos0`, base, pre-round cache snapshot,
  and the row's intended output budget. Preparation has mutated only that row's draft scratch and
  taken a recoverable trunk snapshot; it has not published accepted output.
- `VerifyBatchOutput`: flattened logits and pre-output-norm hidden stacks plus `offsets`/`lengths`,
  and one `VerifyCkpt` per row. `row(i)` must return exactly the slices the scalar commit code sees.

Initially, `prepare_spec_round` and `commit_spec_round` should be the scalar code extracted verbatim.
The existing single-session generator should call the same operations with B=1. This prevents two
independent acceptance implementations from drifting.

## Scheduler cohort contract

Add a distinct phase-a `step_spec_cohort`; do not route speculative rows through phase-c plain
decode. A fast-path cohort key must include every property that changes target math or round shape:

- model/revision and PP placement;
- fixed `spec_k`, current verify length and pending/base shape;
- numeric/exactness tier and graph/round-stream mode;
- cache capacity/ladder class where buffer geometry requires it;
- greedy, sampled or constrained mode.

Stage 2 admits only greedy, unconstrained, fixed-K rows with the round-stream experiment disabled.
Everything else calls today's scalar generator unchanged. Same absolute context position is **not**
required: the target core receives one position vector per row. Same verify length is required in the
first increment, eliminating padding and attention masks. A pending-less first round can form a
separate T=K cohort or run solo; steady rounds form T=K+1 cohorts.

The cohort is dynamic at every round boundary. A row that reaches EOS, budget, cancellation,
admission yield or an error leaves the next wave; no row waits for a finished peer. The box1
divergent arm still had B=4 for 86.8% of observed waves, so this simple shrinking-cohort policy
captures most of the available shape without fabricating work.

No global acceptance length is allowed. Each row accepts its own prefix, preserves its original
request ordering, rolls back to its own snapshot and advances its own pending token. This is the
core anti-raggedness invariant.

## Target-core seam

Add a continuation-only core alongside `decode_step_t_core`, conceptually:

```rust
decode_step_t_batch_core(
    engine,
    token_rows: &[&[u32]],
    pos0s: &[usize],
    caches: &mut [&mut Cache],
    ckpts: &mut [Option<&mut VerifyCkpt>],
) -> Result<VerifyBatchOutput, Error>
```

For fixed K, the physical hidden layout is a concatenation of rows at
`M = sum(T_i) = B * (K + 1)`. Projection, normalization, FFN/MoE and output-head operations should
stream each weight once at M. Sequence-scoped operations must split by `offsets`:

- per-row RoPE positions and sliding-window/global attention view;
- per-row KV append/length and cache capacity;
- per-row GDN convolution/SSM state and rollback stash;
- per-row MoE routing results, with no expert choice influenced by another row;
- per-row PP-stage cache ownership and one flattened stage-boundary transfer.

Return **all** verify logits and pre-output-norm hiddens, not only each sequence's last column. The
acceptance walk needs each prediction, full/partial accept needs the predecessor hidden, and
true-hidden draft refresh may consume the verified prefix. A `VerifyCkpt` must carry row offsets (or
remain a vector of genuinely independent checkpoints); it may never index a flattened column as if
all rows shared one recurrent sequence.

`step35_prime_batch_layers` supplies a useful `ts`/`offs`/`pos_ds`/per-cache skeleton, and
`step35_decode_batch_layers` supplies the served per-cache attention pattern. Neither numeric path
is inherited as correct. The new M=16 tier must be tuned and gated as a verify configuration; the
current contiguous M=16 timing is already a fail signal.

Under PP2, construct every cache through `pp::new_cache`, enter the owning stage before its cache
operations, and preserve the existing event/fence ordering. H2D/D2H publication remains on the CUDA
owner thread.

## Draft batching follow-on

Once verify-first is exact and faster, use the existing round rendezvous to batch draft position
`j` across live rows:

```text
for j in 0..K:
    one B_j x 1 MTP-head weight walk
    row-local draft KV / grammar / sampling decision
    remove rows whose p-min or grammar rule ended the chain
```

Keep one `DraftGraphCtx` and stable device buffers per session at first. A single multi-row captured
graph is a later optimization only after pointer stability and row-reordering gates exist. Draft
sampling counters and grammar clones remain row-local. This extension changes no target-core or
commit contract.

## Exactness risks

The implementation review must account for each of these explicitly:

1. **Sequence boundaries:** no causal attention, KV append, SWA view or GDN scan can cross a row
   offset; each row's absolute position is authoritative.
2. **Rollback:** snapshots and `VerifyCkpt` rebuild data must be indexed by row and layer. Full,
   partial and zero acceptance must restore exactly the scalar row state.
3. **Pending/base semantics:** pending is emitted but absent from `committed`/cache. Mixing T=K and
   T=K+1 rows without explicit lengths shifts every target prediction.
4. **Hidden pairing:** full/partial acceptance uses the predecessor verify hidden to seed the next
   MTP chain. A flattened off-by-one can silently couple adjacent requests.
5. **Numeric configuration:** changing M changes matmul tiling and floating-point association. B=1
   must be buffer-identical to scalar; B>1 needs the repository's numeric-config gates and greedy
   token identity, never an assumption that concatenation is exact.
6. **PP ownership:** stage-local KV/recurrent state, streams, events and boundary copies must remain
   on their owning devices. A correct-output peer-read regression is still a failure.
7. **Sampling and constraints:** Philox counters, filters, penalties, grammar state and draft-mask
   clones are session-local. Until distribution/event-count gates pass, these modes fall back.
8. **Publication ordering:** target results for all rows must validate before any row commits. A
   failed shared forward cannot leave earlier rows published and later rows rolled back.
9. **Lifecycle:** cancellation, EOS, max-token overshoot, admission yield, session affinity, cache
   parking/resume and OOM parking can all change the next cohort without changing surviving rows.
10. **Telemetry:** rounds/drafted/accepted counts and SSE callbacks are attributed and ordered per
    request; batching cannot turn a cohort into one accounting unit.

Recent batch-speculative work independently identifies position ids, masks and KV synchronization
as the central ragged-batch hazards, and its same-length EXSpec grouping is directionally consistent
with this first increment ([Batch Speculative Decoding Done Right](https://arxiv.org/html/2510.22876)).
Its reported decoding-equivalence tolerance is not memra's gate: memra still requires its own
per-request exactness battery. Current vLLM documentation also exposes dynamic draft-token counts,
which reinforces keeping adaptive-K rows out of the first rectangular cohort
([vLLM dynamic speculative decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/dynamic_speculative_decoding/)).

## Gate plan

### G0 — feature containment

- New path default off behind one experimental `MEMRA_SPEC_BATCH=1` door.
- Current scalar path remains the B=1 oracle and runtime fallback.
- No sampled, constrained, adaptive-K or round-stream admission in the first increment.

### G1 — target-core identity

- B=1, T=1..9: compare logits, full hidden stack, cache lengths/content and checkpoint material to
  `decode_step_t_core` from cloned initial state.
- B=2..4, K=1..8: compare every demultiplexed row to independent scalar calls at equal state.
- Cover pending and pending-less rows, different absolute positions, SWA boundaries, near-capacity
  caches, both PP device orders and every promoted verify numeric tier.
- Assert finite outputs and sentinel-free argmax; prove a recorded B>1 core call actually occurred.

For B=1, require byte-for-byte state/output identity. For B>1, any batch-shape floating-point change
falls under explicit numeric-config jurisdiction: per-row argmax/token and state-transition identity
must pass the corpus gates; no tolerance is inferred from a green compile.

### G2 — speculative round oracle

Add `spec-batch-gate` that clones session state, then runs scalar and cohort engines from the same
prompt. At every round compare draft tokens, target predictions, accepted length, bonus, committed
tokens, pending state, cache position, draft-scratch length, KV and recurrent rollback state.

Fixtures must force full, partial and zero acceptance and cover synchronized and divergent rows,
cohort shrink B=4->3->2->1, different context depths and cache ladder rungs. Sweep K=1..8.

### G3 — required repository battery

- `kernel-check` ALL GREEN for every affected kernel/configuration.
- `run-gen` argmax MATCH on the affected Step model.
- `run-spec` K=1..8 self-consistency PASS with batching both enabled and disabled.
- Repeat under the designated target PP layout; CI compile-only is not a substitute.

### G4 — server isolation

Replay c=4 identical and divergent request sets with `MEMRA_SPEC_BATCH=0/1`; require byte-identical
greedy text per request plus equal stop reason, usage, acceptance telemetry and cache/session tail.
Add staggered arrival, EOS in one row, max-token overshoot, disconnect, admission yield, pool
resume/rewind, OOM/error injection and mixed plain/spec traffic. A row leaving the cohort must not
change any survivor's output.

### G5 — sampled and constrained admission

Only after greedy ships experimentally: verify per-session Philox event counts and reproducibility,
distributional exactness, filters/penalties, grammar masks and illegal-bonus replacement. Until each
mode passes, its cohort predicate remains false.

### G6 — performance promotion

- Interleaved scalar vs cohort measurements, at least N=5, with raw logs, exact artifacts, clocks,
  temperatures and concurrent-GPU state.
- Report target-core time, whole speculative-phase time, client throughput and latency separately.
- Require the actual B x T call to beat B scalar verifies at B=2,3,4; an m=16 result matching this
  lane's 146.2 ms proxy is a hard no-go.
- Validate on the RTX PRO 6000 Blackwell target trajectory, then repeat correctness, memory and
  throughput gates on the local 5090 proof rig before any default flip.

If the gate wins, make it the default and keep `MEMRA_SPEC_BATCH=0` only as the rollback seam. If it
is flat or negative, remove the experimental dispatch and preserve these raw receipts as the result.
