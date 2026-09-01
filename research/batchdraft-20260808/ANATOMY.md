# Cross-request batchdraft: current anatomy

Date: 2026-08-09

Source base: `cbe25b75e95f9aed8863771b625e12c35b016286`

Diagnostic runtime commit: `3248e4f91ab8dbe892d7b27c7f0fd30abd2c2009`

## Finding

Speculative verification is already token-batched *within one request*: one round sends that
request's pending token plus its draft tokens through a `T = K + 1` target forward. It is not
request-batched. The server completes an entire speculative burst for session A before it enters
session B, and the engine API accepts exactly one mutable `SpecSession`. The ordinary B-row decode
batch explicitly excludes every session with `spec.is_some()`.

```text
worker tick
  phase a: spec_order
    session A -> step_session -> complete spec burst -> return
    session B -> step_session -> complete spec burst -> return
    session C -> step_session -> complete spec burst -> return
    session D -> step_session -> complete spec burst -> return

  phase b: prefill work

  phase c: only spec.is_none()
    B plain sessions -> one B x 1 decode_step_batch
```

The missing seam is between per-row draft preparation and target verification. Today that boundary
is private to a monolithic single-session generation call.

## Worker path

| Stage | Current code | Ownership / consequence |
|---|---|---|
| Session state | `crates/memra-server/src/worker.rs:1700` (`Session`) | One request owns `spec_k`, `Option<SpecSession>`, sampler, grammar, output and telemetry. |
| Tick policy | `worker.rs:2360-2364` | The scheduler documents spec as phase `(a) ... burst solo`; plain batched decode is phase `(c)`. |
| Spec selection | `worker.rs:2705-2734` | `spec_order` selects `spec.is_some()` rows, optionally sorts cold-first, then iterates them. |
| Serial call | `worker.rs:2734-2769` | Each loop iteration calls `step_session` and waits for the whole call to return before touching the next row. |
| Burst setup | `worker.rs:4794-4914` | `step_session` drains the suffix, builds sampling/constraint state and a request-local SSE/admission-yield callback. |
| Engine entry | `worker.rs:4915-4923` | It calls `generate_spec_session_constrained` or `generate_spec_session_sampled` with one `&mut SpecSession`. |
| Plain batch exclusion | `worker.rs:3055-3060` | The ordinary decode candidate predicate requires `active[i].spec.is_none()`. |
| Plain batch call | `worker.rs:3133-3153` | Disjoint `Cache` borrows are gathered and passed to `decode_step_batch_sampled_lean_masked`; this split-borrow pattern is reusable for a spec cohort. |

`step_session` does not expose a “one draft round is ready” result. Its return boundary is after the
engine has generated up to `MEMRA_SPEC_BURST` output tokens, emitted round callbacks, updated the
sampler, and committed the burst to the server's `generated`/`fed` vectors. Consequently, the
scheduler never simultaneously holds prepared verify rows from multiple sessions.

## Engine path

The public entry points at `crates/memra-engine/src/spec.rs:3147-3210` all converge on
`generate_spec_session_constrained(..., sess: &mut SpecSession, ...)`. The relevant round is:

1. At `spec.rs:4397-4399`, capture the one session's position and take its pre-round cache
   snapshot.
2. At `spec.rs:4401-4712`, build up to `K` draft tokens autoregressively. The graph arm replays one
   T=1 draft graph per position; the eager fallback calls `mtp_head_forward_dev` once per position
   (`spec.rs:4597-4626`). Both mutate that session's `MtpScratch`.
3. At `spec.rs:4715-4726`, build `verify_tokens = pending ++ draft`; steady fixed-K greedy rounds
   therefore have `T=K+1`.
4. At `spec.rs:4727-4741`, allocate one `VerifyCkpt` and call `decode_step_t_core` with one `pos`,
   one `&mut Cache`, and that one token row.
5. `decode_step_t_core_stream` is the single target-verify funnel (`spec.rs:1486-1542`). Under PP-N
   it delegates to `decode_step_t_core_ppn`, still for one sequence.
6. At `spec.rs:4743-4784`, run target argmax for the row's columns and synchronize on its compact
   prediction readback. Sampling/grammar acceptance then walks the same row.
7. At `spec.rs:5188-5214`, account for that session; at `spec.rs:5220-5445`, commit accepted tokens
   and either keep the verified prefix or roll the cache/recurrent state back.
8. At `spec.rs:5536-5585`, restore `DraftGraphCtx`, RNG counters, pending-token and last-hidden
   invariants to the same `SpecSession`; only then does the public call return.

### State that is scalar today

`SpecSession` (`spec.rs:322-368`) owns one each of:

- trunk `Cache` and draft `MtpScratch`;
- committed tokens, last hidden, next prediction and pending token;
- sampled-spec Philox counters;
- persistent `DraftGraphCtx` and turn checkpoint;
- acceptance telemetry.

Each round also owns a cache snapshot, draft tokens/statistics, position/base, grammar clone,
prediction buffers and optional `VerifyCkpt`. `VerifyCkpt` (`spec.rs:687-698`) is per layer and
retains GDN convolution/SSM rebuild material for that row's accepted prefix. None of these objects
has a row id or row-offset table because the API has never needed one.

## Why four requests cannot share a verify today

There are four independent blockers, not just a scheduler predicate:

1. **No common rendezvous.** The worker invokes complete bursts serially; session A has already
   accepted, rolled back and advanced before session B drafts.
2. **A scalar engine transaction.** The generator borrows one `SpecSession` for setup, every round,
   callbacks and tail restoration. There is no owned prepared-round object that can wait for peers.
3. **A scalar target core.** `decode_step_t_core` takes one token slice, one absolute start position,
   one cache and one checkpoint. Its logits/hidden outputs carry columns, not sequence offsets.
4. **Row-local exactness is resolved immediately.** Argmax/readback, grammar/sampling, accepted
   length, cache rollback, GDN state rebuild, draft-scratch truncation and pending-token publication
   all finish before control returns. A shared call needs to demultiplex these before publishing any
   row.

Removing only `spec.is_none()` from the ordinary batch would be wrong. That path is B x 1, consumes
plain `Cache` state, has no draft scratch or verify checkpoint, and cannot represent K+1 target
columns or speculative rollback.

## Nearby reusable seams—and their limits

| Existing seam | What it proves | Why it is insufficient |
|---|---|---|
| `decode_step_batch_sampled_lean_masked` (`decode_batch.rs:434`) | B independent sessions can share one B x 1 weight walk while their caches and sampling metadata stay isolated. | It exposes only one token/position per row and no full hidden stack or rollback checkpoint. |
| `step35_decode_batch_layers` (`decode_batch.rs:1330`) | Step35 projections run at `m=B`; attention, KV length, RoPE/SWA position and cache remain per session. | Again B x 1 only. Its served B>1 numeric configuration is not automatically the verify configuration. |
| `step35_prime_batch_layers` (`hybrid_forward.rs:1617`) | The needed layout skeleton exists: `ts`, `offs`, per-row position tensors and caches, with projection work at `m=sum(T)`. | The public Step35 prime path rejects any `cache.pos != 0` (`hybrid_forward.rs:1853-1867`) and its epilogue returns only the last row logits/hidden per sequence. |
| Generic `prime_cache_batch` (`hybrid_forward.rs:2006`) | Other model families already support continuation concatenation with per-sequence stateful mixers. | It is explicitly a prefill numeric configuration (`hybrid_forward.rs:2020-2021`), not a verify-exact contract, and Step35 routes away from it. |

The implementation seed is therefore the *shape and state-isolation pattern* from batch decode and
concat prime—not either public call unchanged. A cross-request verifier needs its own continuation
core, all-column outputs, per-row checkpoints, and an exactness gate.
