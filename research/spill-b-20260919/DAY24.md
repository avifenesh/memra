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

## 2. The BEFORE receipt (local RTX 5090, `rtx5090-day24/before/`, binary from the merged tree `fb0c0a9a0`)

Runner `run-day24-cell.sh before` (lock acquired at once; `compute-apps-before.csv` empty), client `day24-client.py`,
parser `day24-parse.py`; every input file is in the cell dir (`server.log` stamped per line, `samples.csv` at 250 ms,
`client.jsonl`, `REPORT.txt`; `REPORT-v1-metrics-column.txt` is the first parse, whose real-book column read the
throttled `/metrics` snapshot, 0 for most of the run and one stale 6.96 GB value: the parser was corrected to
reconstruct the real book from the admit lines' `booked_real` deltas, exact where the next admit is the very next
in-flight entry, the MB-rounded request-cost line otherwise, flagged `~`; no tolerance moved). Server boot line:
`[admit-predict] shadow armed: budget_bytes=14534558848 budget_src=derived(effective_free_bytes=16493823104 -
prefix_cache_budget_bytes=348651520 - admission_reserve_bytes=1610612736) ... enforce=false`; `[admit-cal] boot
calibration done: model="q9" route=mtp transient floor 1536MB (static was 1536MB; measured 1266MB; ... 1.1s)`, so
`floor = 1610612736`. Ten requests, ten `200`, `Overloaded/OOM lines = 0`.

Per-request rows (verbatim from `REPORT.txt`; `cost_exact` is the physical cost the real book took, from the next
admit line's `booked_real` delta; `kv_hat` the predictive charge from the same request's admit line):

```text
warm P=66    cost_exact=None       cost_used=71000000    kv_hat=2305152    real/shadow=30.80x
a0   P=6006  cost_exact=1702520120 cost_used=1702520120  kv_hat=193525048  real/shadow=8.80x
a1   P=6295  cost_exact=1722630196 cost_used=1722630196  kv_hat=208900148  real/shadow=8.25x
a2   P=6564  cost_exact=1731567988 cost_used=1731567988  kv_hat=213430644  real/shadow=8.11x
a3   P=6837  cost_exact=None       cost_used=1741000000  kv_hat=217983412  real/shadow=7.99x
b0   P=13362 cost_exact=None       cost_used=2014000000  kv_hat=340841332  real/shadow=5.91x
c0   P=6006  cost_exact=1727256888 cost_used=1727256888  kv_hat=218736952  real/shadow=7.90x
c1   P=6295  cost_exact=1736282936 cost_used=1736282936  kv_hat=223028024  real/shadow=7.79x
c2   P=6564  cost_exact=1744684344 cost_used=1744684344  kv_hat=227022136  real/shadow=7.69x
c3   P=6837  cost_exact=None       cost_used=1753000000  kv_hat=231075640  real/shadow=7.59x
```

The decomposition (A rows, `residual~ = (cost - kv_hat) - bpt x (C - (P + L + 8))`, the line's 1 MB grain):

```text
a0 P=6006 L=64 C=6166 path=plain bpt=14848 kv_hat=193525048 cost~=1703000000 gap~=1509474952 bracket=1306624 residual~=1508168328 W~=1508000000 A~=103000000 D~=0
b0 P=13362 L=96 C=13522 path=spec bpt=16704 kv_hat=340841332 cost~=2014000000 gap~=1673158668 bracket=935424 residual~=1672223244 W~=1628000000 A~=116000000 D~=44000000
A: max |residual~| = 1672223244 bytes
```

Reading: the residual IS `W + D` (a0: 1,508,168,328 against `W~ = 1,508 MB`, `D = 0`; b0: 1,672,223,244 against
`W~ + D~ = 1,628 + 44 MB`), as section 1 derived. The book-level line: at the fourth concurrent admit
`[admit-predict] ... prompt=6837 ... kv_hat=217983412 booked_bytes=615855840 booked_real=5156718304 inflight=3`: the
predictive book read 616 MB for three in-flight 6k-token requests whose real book was 5,157 MB (8.37x; memra#476's
historical 3.2x is a different model and shape and is not reproduced here, per its acceptance list). The request-cost
lines that fed it, verbatim: `[admission] request cost: model="q9" ctx=6166 path=plain = 14848 B/token x ctx + 1508MB
prefill-workspace + 128MB fixed = 1727MB` (and `ctx=13522 path=spec = 16704 B/token x ctx + 1628MB prefill-workspace +
116MB fixed = 1970MB`, `[admission] per-session draft-state charge: model="q9" +44MB (measured capture high-water)`).
The `fixed` term (`A`) learned during the cell (0, 103, 114, 115, 116, 128 MB), as the residual learner is designed to.

Device rows (every fourth 250 ms sample; `delta` over the idle baseline `8697 MiB`, n=8; the books reconstructed):

```text
epoch_ms       used  delta       real_book   shadow_book under_real  under_shadow inflight
1790025361021  9593  939524096   6897718304  833839252   -5958194208 105684844    4     (a burst)
1790025366021  9913  1275068416  6897718304  833839252   -5622649888 441229164    4
1790025369021 10969  2382364672  2014000000  340841332   368364672   2041523340   1     (b0)
1790025371021 11129  2550136832  2014000000  340841332   536136832   2209295500   1
1790025373021 11129  2550136832  0           0           2550136832  2550136832   0     (between bursts)
1790025375021 11129  2550136832  6961224168  899862752   -4411087336 1650274080   4     (c burst)
max_underbook_real   = 2550136832 at inflight 0;  while inflight > 0: 822879944
max_underbook_shadow = 2550136832 at inflight 0;  while inflight > 0: 2331399880
B: max_underbook_real=2550136832 <= floor=1610612736: FAIL       C: Overloaded/OOM lines = 0: PASS
```

Two readings, both stated as measured. (i) `memory.used` never falls after a burst: the engine's pool keeps freed
blocks mapped, so between bursts the device delta is the pool's high-water (2,550 MB after the 12k prompt) while both
books are 0. The pre-registered headline is the max over ALL samples, so B reads FAIL on the before and, since the fix
does not move the real book, on the after too; the tolerance is not moved after the fact. The labeled second row
(in-flight samples only) is the one that speaks to per-request booking: the real book under-books by 823 MB at most
(inside `floor`), the shadow book by 2,331 MB (outside `floor`; the `W + D` omission). (ii) During the four-way burst
the real book (6,898 MB) is far above the device delta (1,275 MB): the physical gate charges `W`'s `call_row_bytes`
term per request while the prime slab pool is one retained per-device allocation shared by every concurrent prime
(section 1), so the physical book OVER-books concurrency by about 5.6 GB here. That is a conservative direction and a
different defect from #476 (it refuses fits, it never admits misfits); it is named for the lead, not fixed by this lane.

## 3. The fix (`crates/memra-server/src/admit_predict.rs`, `crates/memra-server/src/worker.rs`, `docs/FLAGS.md`)

`admit_predict::RequestCharge { context_hat_bytes, activation_bytes, prefill_workspace_bytes, draft_state_bytes }`
with `from_physical_cost(cost, ctx(C), A, D, ctx(P + L + 8))` (W is the remainder the cost carries beyond `ctx(C) + A +
D`, saturating) and `total()`. Both predictor sites in the admission block (the verdict, and the booking at
`active.push`) now compute `shadow_kv_hat = RequestCharge::from_physical_cost(cost, context_cap_bytes, activation_bytes,
draft_state_bytes, kv_hat_ring(P, L, bpt, ring_bpt, ring_rows, 0)).total()`, where `cost` is the very value the real
book takes (`s.booked_kv_bytes = cost`): the verdict reads the cold cost (the restore and eager arms have not run yet
at that seam, so the verdict is never below the book), the booking the final restore- or eager-adjusted one.
`context_cap_bytes = model.context_bytes(admission_cap, estimate_spec)` is read from the cost model beside `cost`. No
new numeric program (the context term still comes from `context_cache_bytes` through `kv_hat_ring`), no new flag, no
new formula: the predictor stops re-deriving a narrower charge and reads the physical gate's own terms. The shared
`reserve` stays in the budget (charged once). The `[admit-predict]` line shape is unchanged (`verdict_line_locks_fields`
holds); `admit_predict_shadow_wiring` holds (one book-admit, one book-retire, unchanged text). `docs/FLAGS.md` rows
`MEMRA_ADMIT_PREDICT_SHADOW` and `MEMRA_ADMIT_PREDICT_ENFORCE` state the new `fixed` meaning in the same commit.

CPU tests (all pass; verbatim block in section 3.1 once the run completes): `admit_predict::tests::
request_charge_books_every_physical_term_once` (cost - total == the bracket; total - legacy kv_hat == W + D),
`request_charge_follows_the_restore_and_eager_adjusted_cost`, `request_charge_without_workspace_or_draft_is_the_legacy_kv_hat`
(no published shape, plain path: bit-identical to the pre-fix book), `request_charge_saturates_instead_of_wrapping`;
`worker::tests::predictive_charge_books_the_physical_cost_terms_on_every_path` (two `AdmissionCostModel`s, flat and
ring geometry, plain and spec, cold / retained-prefix / eager compositions: `W` read from the cost equals the model's
own `prefill_workspace_bytes(P)` or `(P - R)`, `D` equals the charge, and `cost - total == ctx(C) - ctx(P + L + 8)` on
every path; plus the bare model regression).

### 3.1 CPU test run (verbatim, `checks/cpu-tests-2.log`; the first run `cpu-tests.log` failed on a fixture whose `cap` was below `P + L + 8`, so `cost - total` underflowed in the test, not in the code: `cap` raised to 6,200, nothing else changed)

```text
test admit_predict::tests::kv_hat_ring_below_at_above_ring_rows ... ok
test admit_predict::tests::kv_hat_ring_flat_model_equivalence ... ok
test admit_predict::tests::kv_hat_ring_saturation_on_overflow ... ok
test admit_predict::tests::kv_hat_ring_refuses_ring_bpt_greater_than_total - should panic ... ok
test admit_predict::tests::request_charge_books_every_physical_term_once ... ok
test admit_predict::tests::kv_hat_ring_refuses_zero_ring_rows_with_positive_ring_bpt - should panic ... ok
test admit_predict::tests::request_charge_follows_the_restore_and_eager_adjusted_cost ... ok
test admit_predict::tests::request_charge_without_workspace_or_draft_is_the_legacy_kv_hat ... ok
test admit_predict::tests::request_charge_saturates_instead_of_wrapping ... ok
test admit_predict::tests::verdict_line_locks_fields ... ok
test health::tests::loading_is_not_live_and_not_ready ... ok
test worker::tests::retained_restore_cost_keeps_context_draft_residual_and_reserve_paid ... ok
test worker::tests::admission_cost_scales_with_each_requests_context ... ok
test worker::tests::predictive_charge_books_the_physical_cost_terms_on_every_path ... ok
test worker::tests::readiness_follows_the_boot_calibration_probe ... ok
test worker::tests::admit_predict_shadow_wiring ... ok
test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 764 filtered out; finished in 0.02s
```

Not done, named: the "outstanding" refinement (release `W` from both books when the prime completes) needs a
prime-completion seam and a retire-side change in both books; and the physical book's per-request charging of the
shared slab (section 2 (ii)).

## 4. The AFTER receipt (`rtx5090-day24/after/`, binary `build-fix/`, same runner, same sequence, same tolerances)

Ten requests, ten `200`, `Overloaded/OOM lines = 0`; same boot lines (`budget_bytes=14534558848`, `floor 1536MB`,
calibration `1.2s`); idle baseline `8697 MiB` (n=8), peak `11129 MiB`, both equal to the before cell.

```text
warm P=66    cost_exact=None       cost_used=71000000    kv_hat=69999504   real/shadow=1.01x
a0   P=6006  cost_exact=1702520120 cost_used=1702520120  kv_hat=1701213496 real/shadow=1.00x
a1   P=6295  cost_exact=1722630196 cost_used=1722630196  kv_hat=1721323572 real/shadow=1.00x
a2   P=6564  cost_exact=1731567988 cost_used=1731567988  kv_hat=1730261364 real/shadow=1.00x
a3   P=6837  cost_exact=None       cost_used=1741000000  kv_hat=1739286964 real/shadow=1.00x
b0   P=13362 cost_exact=None       cost_used=2014000000  kv_hat=2012955268 real/shadow=1.00x
c0   P=6006  cost_exact=1727256888 cost_used=1727256888  kv_hat=1726425400 real/shadow=1.00x
c1   P=6295  cost_exact=1736282936 cost_used=1736282936  kv_hat=1735451448 real/shadow=1.00x
c2   P=6564  cost_exact=1744684344 cost_used=1744684344  kv_hat=1743852856 real/shadow=1.00x
c3   P=6837  cost_exact=None       cost_used=1753000000  kv_hat=1752379192 real/shadow=1.00x
A: max |residual~| = 479880 bytes (tolerance 0 at the line's 1 MB grain, so |residual~| <= 2e6 passes)
B: max_underbook_real=2550136832 <= floor=1610612736: FAIL        C: Overloaded/OOM lines = 0: PASS
while inflight > 0: max_underbook_real=822879944  max_underbook_shadow=823711432
```

**Assertion A holds to the byte where the physical cost is exact**: `a0: cost_exact - kv_hat = 1702520120 -
1701213496 = 1306624 = 14848 x (6166 - (6006 + 64 + 8))`, the context bracket and nothing else; `c0: 1727256888 -
1726425400 = 831488 = 14848 x 56`; at the fourth concurrent admit `booked_bytes=5152798432 booked_real=5156718304`,
a difference of `3919872 = 3 x 1306624` (before: `615855840` against the same `5156718304`). The residual on the
MB-rounded rows is inside the line's grain (max 479,880 bytes; sign both ways, as rounding gives). B is unchanged from
the before cell (the fix does not touch the real book; the in-flight row now reads the same 823 MB for both books,
because the two books now differ only by the bracket). C holds.

**Honest scope**: one model (Qwen3.5-9B NVFP4, GDN trunk, `PrimeWorkspaceShape` published, no ring geometry), one
card, shadow mode. The retained-prefix arm (`W(P - R)`) did not trigger on this card (a plan is taken only when the
cold cost does not fit; here everything fit), so that path is covered by the CPU test alone
(`cost_after_prefix_restore` then `from_physical_cost` reads `W(P - R)`); the enforcing door was not exercised (it
would now refuse on the fuller charge, which is the intended change and needs its own cell against a budget arm).

## 5. memra#524: the readiness gap, written from the code at `fb0c0a9a0`

Order inside `worker::run`: `[worker] Engine ready` (line 16015, the ENGINE constructed, before any model loads; this
is the line the issue's incident pairs with the sym-graph capture), model load, `run_boot_calibration(...)` (17172),
`ready_tx.send(Ok(...))` (17311), `health.mark_ready()` (17316, PHASE_LOADING to PHASE_IDLE; `/readyz` reads
`WorkerHealth::ready`, which is `Err` in PHASE_LOADING: `health::tests::loading_is_not_live_and_not_ready`). So on
the ARMED path the one warmup the server runs, the boot calibration probe (one spec-shaped 4096-token generation
through the real serving route: chunked prime, draft-chain capture, sampled verify, then the pool high-water read),
completes BEFORE `/readyz` can say ready. The gap is elsewhere:

- The probe is skipped, and then NOTHING warms before ready, when `MEMRA_ADMIT_CALIBRATE=0`, when spec serving is
  disabled (`serve_spec_enabled()` false: a plain-only deployment), or when `MEMRA_ADMIT_RESERVE_MB` is set; and when
  the probe FAILS the static floor serves and the process is cold. On those paths graph capture (the glm5 TP sym-graph
  is captured lazily inside the forward walk, `hybrid_forward.rs:3725` `walk_graphed`, so at the first request) and the
  first prime both happen after `/readyz` is 200.
- Even on the armed path the probe warms ONE route and shape (B=1 spec, one 4096-row chunk); a first plain request, a
  batch width above 1, or a chunk width the probe did not walk still captures or grows after ready.
- `/health` has no `phase=warming`: the phases are LOADING, IDLE, BUSY, DEAD.

Measured on this rig, both cells (`REPORT.txt`, section "readiness"): `/readyz` first 200 to the warm request's
completion `1437 ms` (before) and `1426 ms` (after) at a 500 ms poll grain, of which the client's own wait was about
1,000 ms; the warm request's `[ttft]`: `prime_wait_ms=0.793 prime_ms=75.805 decode_wait_ms=0.000 first_decode_ms=77.197`
(before) and `prime_wait_ms=0.679 prime_ms=76.002 ... first_decode_ms=77.187` (after): the first request after ready
primed and decoded in 77 ms because the probe (`1.1s` / `1.2s`) had already warmed the route. N=1 per cell, warm
rig, not a timing claim; it is the receipt that on the armed path the first request does not pay the warmup.

**Bounded fix taken**: none in code, because the task's bound is "readiness waits for the warmup steps the server
itself runs, no new work added to warmup", and on the armed path it already does; on the skipped paths the server runs
no warmup, so there is nothing for readiness to wait for without adding work. What lands: the CPU test
`worker::tests::readiness_follows_the_boot_calibration_probe` (anchored on the comment-stripped production text of
`run`: the probe call precedes `ready_tx.send(Ok(`, which precedes `health.mark_ready()`; and the probe generates
through the real routes), so a reorder is a red test. `health::tests::loading_is_not_live_and_not_ready` already
covers the not-ready read before `mark_ready`.

**Pre-registered, not built (wider than a day): `tools/health-fault-gate.sh`**, a fast-gate cmd cell on the local
5090, every arm asserting HTTP codes and ledger rows, each with a red twin (fault injected, assertion fires):
(a) hung canary: a `PATH`-shadowed `nvidia-smi` that sleeps past the canary deadline once during warmup, then answers;
assert no `mark_gpu_fault` latch, `/readyz` 200 after warmup, `/health` phase names the warmup (needs the
`phase=warming` state, a health.rs change: new phase constant, `set_phase` at the probe's start, `mark_ready` after).
(b) wedged worker: an injected stall door (exists as `MEMRA_HEALTH_PROGRESS`'s red arm? to be confirmed in health.rs
before the gate is written); assert 503 within `MEMRA_HEALTH_STALL_S`, respawn, `/readyz` 200 again.
(c) OOM at prime under a tiny `MEMRA_ADMIT_DRIVER_HEADROOM_MB`; assert park/requeue, no 5xx to peers, peers' streams
complete. (d) client disconnect mid-stream; assert the session retires within one tick, ledger row
`client_disconnected`, peers unaffected. (e) SIGTERM with N streams open; assert new requests 503 + `Retry-After`,
`/readyz` 503, streams finish, exit 0 before `MEMRA_DRAIN_S`, zero-debit rows for anything killed.
Plus (f), from this day's finding: a plain-only boot (`serve_spec_enabled()` false) must either run a plain warmup
before ready or `/health` must say so; today it says ready with a cold route. Effort: the gate plus the warming phase
is a two-day lane (the arms exist as mechanisms; the red twins and ledger assertions are the work).
