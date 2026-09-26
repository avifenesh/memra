# WP-B day 43: O13, the spec pool's exact-extension miss after an overshooting final burst

OWED.md O13. DAY41 2.1 and 2.2 read the default spec route resuming 34 of 60 exact-extension turns on the 27B (R2 FAIL
on `keep`, both sittings, both orders). Placed in the code: `SpecSession::committed` keeps the final round's accepted
drafts past the request budget ("INCLUDING overshoot: spec commits accepted drafts past max_new"), so the parked
session is longer than the public stream the client sends back, and the exact probe
(`prompt.starts_with(&e.sess.committed)`) misses whenever the last round overshot. The DSpark engine already clamps its
commits at the budget. This day builds the MTP twin behind a default-OFF door and measures what it recovers. Every cell
is `executed-not-qualified`; no default moves.

## 1. Pre-registration

Committed and pushed before any day-43 code and before any day-43 cell. Nothing in section 1 changes after a number is
seen; a failed clause is recorded as it reads and a revision is a new, dated addendum pushed before its code.

### 1.1 The arm (`MEMRA_SPEC_BUDGET_CLAMP`, default unset; decide-by 2026-10-10)

Unset: today's program, byte for byte. `1`, on the Qwen MTP session route (`step_spec`'s qwen arm, `SpecSession`):

- **The worker** sets `SpecSession::budget_room = Some(request_room)` before each burst (the request's remaining public
  budget, `s.budget - s.generated.len()`, not the burst's cadence target, which may be 1 under prime fairness). Unset,
  the field stays `None`.
- **The engine**, in session mode, greedy, unconstrained, after the accept decision of a round (after the grammar
  truncation, which it never meets because it requires no constraint): with `room = budget_room - out.len()`, when
  `n_acc + 1 > room`, `room >= 1` and `base + room - 1 >= 1`, it truncates the round at slot `na = room - 1` exactly
  as the grammar truncation does: `n_acc = na`, `bonus = draft[na]` (the verify's own argmax at that column, because
  the draft there was accepted). The round then commits through the ordinary partial-accept path (KV truncated to
  `pos + base + na`, the recurrent state rebuilt from the round's `VerifyCkpt`, the bonus pending), which is the path
  every rejection takes. The public stream is the same tokens; the session's `committed` after the park's pending flush
  is exactly the prompt plus the public stream.
- **Receipt:** one line per firing, `[spec] budget clamp: round truncated at <na> of <n_acc> accepted (<room> of the
  request's budget left; model <m>)`.
- **Outside the arm, stated:** sampled spec (the rejection sampler's Philox counters would differ from today's for the
  truncated columns), constrained requests, the round-stream arm (`MEMRA_SPEC_STREAM=1`, experimental; the clamp does
  not apply there), a first round with no pending token and `room == 1` (`base + room - 1 == 0`), and the Gemma, GLM,
  DSpark (already clamped) and step35 tensor-parallel spec engines.

### 1.2 One numeric program per request

Within the request: the drafts, the verify batch (its tokens and width) and the accept decision are unchanged; the
clamp only declines to commit accepted columns past the budget, which were never public. The committed rows are the
same verify columns a rejection at that slot keeps (the partial-accept path), so the parked state is a state today's
program already produces on every rejected round. Across requests: a later turn that resumes the clamped session
resumes on the verify-produced rows through the pool's exact probe, the same program `keep` resumes on today whenever
the last round did not overshoot; its residual against a cold prime is the near-tie class DAY41 prices (O11), not a
new one.

### 1.3 CPU before the cards

A census test: the door is read only where the worker sets `budget_room` on the qwen spec arm; the engine reads the
field only in the session-mode greedy unconstrained accept path; unset leaves `budget_room` `None`. A pure-function
test of the truncation rule (`room`, `n_acc`, `base` over the edge cases of 1.1).

### 1.4 The cell (`day41-client.py` RX shape, unchanged)

The spec route only (the plain route does not overshoot), `MEMRA_PREFIX_CACHE_MB=0`, one binary, arms `unset` and
`clamp` (`MEMRA_SPEC_BUDGET_CLAMP=1`), orders O1 (unset then clamp) and O2 (clamp then unset), plus `offprev` (the tip
with the door's commit reverted, spec route, RX): 5 boots per card. Lengths 6,144 and 30,720 on both cards plus 122,880
on the target card; G in {32, 256}; N = 5 conversations per (L, G). Runner `day43-run.sh` (the day-41 runner with the
door as the arm), reader `day43-read.py`.

### 1.5 Clauses

- **C1 exactness within a request.** Every turn-1 row and every cold-twin row has the same completion digest on both
  arms in each order (fresh prompts: no resume can differ).
- **C2 the clamp fired and the parks match.** On `clamp`, at least one `[spec] budget clamp:` line per boot, and every
  resumed turn's `cached_tokens` equals the previous turn's prompt plus its completion tokens.
- **C3 resumes.** On `clamp`, at least 80% of RX turns 2 and 3 resume (`cached_tokens > 0`).
- **C4 health.** No `CUDA_ERROR_OUT_OF_MEMORY` line, no 503, no crash line on any boot.
- **C5 door OFF.** Every `unset` RX row equals `offprev`'s.

Readings, no bound: resumed fraction per arm; resumed-turn TTFT p50 and p95 per arm (turns 2 and 3); resumed-against-cold
flips per arm (the near-tie residual, reading only); generated tokens over boot wall time; clamp firings per boot.

### 1.6 Price

Code: about 0.5 agent-day. Cells: the 5090 about 1.5 h, the target card about 3 h (5 boots, the 122,880-token cold twins
dominate).

## 2. Results

Written after the runs. Section 1 is unchanged.
