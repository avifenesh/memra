# WP-B day 24: memra#476 (the predictive book omits prefill workspace and draft state) and memra#524 (`/readyz` before warmup)

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`. Day start: `git merge --no-ff origin/main e2e9e294a`
(#616, integ25; merge `fb0c0a9a0`; the only `crates/` change is C's `hash-micro` diagnostic bin), then
`tools/check-conflict-markers.sh` `OK` (ruling 27), pushed. Every push of the day whose range touches engine source is
refused `UNQUALIFIED` by the #589 hook on a plain push and runs as `MEMRA_RELEASE_QUALIFICATION_MODE=development git
push` (`UNQUALIFIED DEVELOPMENT`, logged in `.git/memra-gate-skips.log`). **No qualification is claimed anywhere in this
record**; every GPU cell is `executed-not-qualified`; no timing is compared across cards.

Rig discipline as on every prior day: every GPU command on the local RTX 5090 Laptop GPU runs under `flock -w`
`/tmp/memra-5090.lock` (a foreign `memra-server`, not this lane's, held the lock and 6,562 MiB of the card when the day
started; it was neither inspected nor signalled; this lane waits for the lock), CPU-heavy work runs under `systemd-run
--user --scope -p CPUQuota=1200% -p MemoryMax=28G`, `nvidia-smi --query-compute-apps` is read before and after every
cell, and no process this lane did not start is touched.

## 1. memra#476: the accounting gap as arithmetic (written before any change)

Sources read at `fb0c0a9a0`: `crates/memra-server/src/worker.rs` (`AdmissionCostModel` and `estimate`,
`prefill_workspace_bytes`, `cost_after_prefix_restore`, `admission_reserve`, `admission_required`, the admission block
from the `[admission] request cost:` line through `admission_book.admit`, the retire seam, `run_boot_calibration`,
`effective_free_bytes`), `crates/memra-server/src/admit_predict.rs` (`context_cache_bytes`, `kv_hat_ring`,
`AdmissionBook`, `DerivedShadowBudget`), `crates/memra-engine/src/hybrid_forward.rs` (`PrimeWorkspaceShape`,
`HyperPrimeWorkspaceShape`, `prime_workspace_shape`, `prime_slabs_get`).

Notation: `ctx(n) = context_cache_bytes(bpt, ring_bpt, ring_rows, n)` (the exact KV allocation for `n` rows on the
request's path, plain or spec), `P` = exact prompt tokens after tokenization, `L` = the predicted completion (the causal
per-tenant p95, `CompletionHistory::lhat`), `C` = `ctx_cap` (the request's context cap: `prompt + max_tokens`, or the
server window when `max_tokens` is omitted), `A` = `activation_bytes` (the learned fixed residual high-water),
`W(n) = prefill_workspace_bytes(n)` (the hyper shape plus the non-hyper `PrimeWorkspaceShape` charge at `n` prompt rows,
each `admission_bytes`: `call_row_bytes x min(n, chunk rows) + prompt_row_bytes x n`; 0 when the model publishes no
shape or `MEMRA_ADMIT_PREFILL_WORKSPACE=0`), `D` = `draft_state_bytes` (spec path only: the DFlash plane at `C` on a
dspark model, else the measured draft-graph capture high-water `draft_session_admission_bytes()`), `R` = the restored
prefix rows when a retained-prefix plan is taken.

**What the physical gate charges per admitted request** (the `cost` the VRAM gate compares to effective free):

```text
cost(r)     = ctx(C) + W(P) + A + D                                  (estimate + the draft-state line)
cost'(r)    = ctx(C) + W(P - R) + A + D        when a retained-prefix plan is taken (cost_after_prefix_restore)
required(r) = cost(r) + reserve                                      (admission_required)
reserve     = admission_reserve(spec_capable, cost, transient_floor, MEMRA_ADMIT_RESERVE_MB)
            = floor                            on a spec-capable path
            = min(cost, floor)                 on a plain path
floor       = max(boot-calibrated transient floor, SPEC_SHRINK_RESERVE)
```

At `active.push` the real book takes `booked_kv_bytes = cost` (the restore-adjusted value when a plan was taken) and the
retire seam subtracts the same number. `reserve` is never booked per request: it is the shared transient class (capture
arenas, verify activations, prime chunk slabs) and is charged once on top of the in-flight sum at every admission.

**What the predictive book charges per admitted request** (`shadow_kv_hat`, both the verdict site and the booking site):

```text
kv_hat(r)   = ctx(P + L + 8) + A                                     (kv_hat_ring; CTX_HAT_SLACK = 8)
verdict     = reject-kv  iff  kv_hat(r) + sum(in-flight kv_hat) > budget
budget      = effective_free(boot) - prefix_cache_budget - floor     (DerivedShadowBudget, unless MEMRA_ADMIT_PREDICT_BUDGET_MB)
```

**The gap, per request**, is the difference of the two books:

```text
cost(r) - kv_hat(r) = [ctx(C) - ctx(P + L + 8)]  +  W(P) (or W(P - R))  +  D
                        by design                    NOT BOOKED             NOT BOOKED
```

The first bracket is the predictor's purpose (it books the predicted context, the physical gate the cap) and is not a
defect. The second and third terms are the omission memra#476 names: the prefill workspace at the request's prompt
length and path, and the per-session draft-graph state, are charged by the physical gate and by the real book but by
neither the predictive verdict nor the predictive book. The predictor's budget already subtracts the shared `floor`
once, so the shared transient class is accounted on the predictive side; what is missing is exactly the two per-request
terms.

**Where the real allocation happens.** `ctx(C)`: the session's KV planes, allocated at `C` when the session is created
(`new_session` / `Cache::new_planned`), live for the session. `W`: the non-hyper prime keeps its trunk transients in a
per-device retained slab pool sized by the CALL's row count (`prime_slabs_get(t)` grows `t_cap` monotonically and is
shared by every prime on that device), plus the per-prime `hiddens` stack (`prompt_row_bytes x P`); so the
`call_row_bytes` term is a device high-water that the physical gate charges per request (conservative by construction:
concurrent primes share one slab), and the `hiddens` term is genuinely per prime. `D`: captured at the session's first
spec burst (draft-chain graphs, capture-retain keepers, q slots), live for the session. Graph growth of a NEW decode
shape (a batch width the boot calibration probe did not exercise) is in neither book and in neither the physical nor the
predictive per-request charge: it is covered only insofar as the calibrated `floor` (one B=1 spec-shaped generation
through the real path, pool high-water read) exceeds what later shapes add. The cell below measures whether the device's
real delta stays within `booked_real + floor`; if it does not, that overshoot is the shared-growth term and is not fixed
by a per-request booking.

**Bounded fix, stated before the receipt.** One cost function is already the source of both books' KV and `A` terms
(`AdmissionCostModel`). The predictor's `fixed_bytes` argument becomes `A + W(P or P - R) + D` from the same
`prefill_workspace_bytes` and the same `draft_state_bytes` the physical `cost` was built from at that very site, so
`cost(r) - kv_hat(r) = ctx(C) - ctx(P + L + 8)` exactly. No new numeric program, no new flag, no new formula: the
predictor stops re-deriving a narrower charge and reads the physical gate's own terms. The `reserve` stays out of the
per-request book (charged once in the budget). The "outstanding" refinement (release the workspace charge when the prime
completes, keep the persistent terms) is wider than the predictor plus a test module: it needs a prime-completion seam in
both books and a retire-side accounting change; it is named, not done, here.

### 1.1 Pre-registered cell (local RTX 5090, before and after; the same script, the same sequence)

Server: the merged tree's `target/release/memra-server`, `MEMRA_COMPAT=openai`, one model `q9` =
`Qwen3.5-9B-NVFP4-MTP-GGUF.gguf` (SHA-256 `52c9cceb...`, the artifact day 23 verified; embedded NextN drafter, spec
path on), `MEMRA_ADMIT_PREDICT_SHADOW=1` (log only: nothing is refused, so the sequence is identical before and after),
`MEMRA_TTFT_TRACE=1`. Client sequence, fixed: after `/readyz` is 200, one 64-token warm request (settles the first
request's transients), an idle baseline of 8 samples, then (a) four concurrent requests of about 6,000 prompt tokens
each with `max_tokens=96`, (b) one request of about 12,000 prompt tokens with `max_tokens=96`, (c) the same four as (a)
again (their prefixes now retained: the restore-adjusted `W(P - R)` arm). Sampler: `nvidia-smi
--query-gpu=memory.used` and `/metrics` (`admission_booked_bytes`, `cuda_driver_free_bytes`) every 250 ms for the whole
sequence. Server log parsed for every `[admission] request cost:`, `[admission] per-session draft-state charge:`,
`[admission] retained prefix plan:`, `[admit-predict]` and `[ttft]` line; `Overloaded`, `CUDA_ERROR_OUT_OF_MEMORY`,
`out of memory` counted.

Rows produced, verbatim: per request `prompt`, `predicted_completion`, `kv_hat`, `booked_real` (both from the
`[admit-predict]` line), `prefill-workspace` and `fixed` (from the request-cost line), `draft-state`, and
`gap = cost - kv_hat`; per 250 ms sample `t`, `memory.used`, `delta` (over the idle baseline), the real book
(`admission_booked_bytes` summed), the shadow book (the in-flight sum of `kv_hat` from the admit lines and the `[ttft]`
completion times). The two headline numbers: `max_underbook_shadow = max over samples of (delta - shadow book)` and
`max_underbook_real = max over samples of (delta - real book)`.

Pre-registered assertions for the AFTER cell (numbers with reasons, fixed here, not fitted):

- **A (arithmetic, per request): `booked_real - kv_hat == ctx(C) - ctx(P + L + 8)`, tolerance 0 bytes**, on every
  request of the sequence, including the restored arm (c). Reason: after the fix both books read the same
  `prefill_workspace_bytes` and `draft_state_bytes` at the same site, so the only residual is the by-design KV bracket,
  which the CPU test computes from the model's coefficients; any nonzero remainder is a term one book has and the other
  does not.
- **B (device, shared term): `max_underbook_real <= floor`**, where `floor` is the boot-calibrated transient floor the
  `[admit-cal] boot calibration done:` line prints. Reason: `reserve` is the one term the real book does not carry per
  request and it is bounded by the calibrated floor by construction of the calibration; a breach names unbooked shared
  growth (a new graph shape), which is a separate defect from #476's per-request terms. B is measured before and after
  (the fix does not move the real book); it is recorded, not a gate on the fix.
- **C: zero `Overloaded`, zero OOM lines** in either cell (shadow mode; the physical gate is what admits).

The BEFORE cell's `gap` column is expected to read `W(P) + D` above the KV bracket; that expectation is stated, not
asserted. If the before cell does not run (lock never free within the bounded wait, model absent), the arithmetic above
stands on its own and the stop is recorded.
