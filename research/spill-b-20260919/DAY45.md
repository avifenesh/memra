# WP-B day 45: O4, the outstanding-only `W` release at prime completion

OWED.md O4, named in DAY24 (sections 1 and 3: "release the workspace charge when the prime completes, keep the
persistent terms ... needs a prime-completion seam in both books and a retire-side accounting change"). Both admission
books (the predictive shadow book, `AdmissionBook.shadow_booked_bytes` from `shadow_kv_hat`, and the real book,
`booked_bytes` from `booked_kv_bytes`) carry each session's prefill workspace `W` for its whole life, though the
workspace is live only while the session primes. The predictive door (`MEMRA_ADMIT_PREDICT_SHADOW` and `_ENFORCE`,
both default OFF) reads the shadow book, and `/metrics` publishes the real one (`admission_booked_bytes`). Under
`MEMRA_ADMIT_BY_MEMORY` the booked reading already counts `W` only for still-priming sessions (`pending_prime`), so
this item is the two books' side. No default moves.

## 1. Pre-registration

Committed and pushed before any day-45 code and before any day-45 cell. Nothing in section 1 changes after a number is
seen; a failed clause is recorded as it reads and a revision is a new, dated addendum pushed before its code.

### 1.1 The arm (`MEMRA_ADMIT_W_RELEASE`, default unset; decide-by 2026-10-10)

Unset: today's program and today's books, byte for byte. `1`:

- **At the admit seam** each session records `booked_w_bytes`, the prefill-workspace term of its own charge
  (`RequestCharge::from_physical_cost(cost, ..).prefill_workspace_bytes`, the one term both books already carry).
- **The prime-completion seam, one site:** at the tick top, before admission, every active session whose prime has
  completed (`prefill_done`) and whose `W` is still booked releases it from both books (`AdmissionBook::release`,
  model row) and reduces its own `booked_kv_bytes` and `shadow_kv_hat` by it, once. Every prime path reaches
  `prefill_done`, so one site covers them all.
- **Retire side:** the retire seam subtracts the session's (reduced) charges as today, so both books stay exact by
  construction: a session that retires before its prime completes still carries `W` and retires it.
- **Receipt:** `[admit-book] w-release id=<id> model=<m> bytes=<W> booked_real=<after> booked_shadow=<after>`.
- **Stated limits:** a step-OOM park that re-admits a session books it again at re-admission (today's rule); the
  prime-slab high-water (the shared slab, DAY39) is not a per-session term and is not touched.

### 1.2 One numeric program

The arm changes only accounting reads: the predictive door's verdicts (when armed) and the published booked figure.
No token path reads either book, so outputs are unchanged by construction; E3 below measures it.

### 1.3 CPU before the cards

Unit tests: `AdmissionBook::release` (both books, saturating, inflight unchanged), the per-session release is
idempotent, a release then retire equals a retire of the full charge. A census test: the door is read at the admit
seam and the one release site only.

### 1.4 The cell (`day45-client.py`, one boot = one arm)

`MEMRA_ADMIT_PREDICT_SHADOW=1` (log only: no verdict refuses, so both arms serve the same sequence), the prefix cache
at its default. Sequence: 4 warm requests of 512 tokens; then a burst of B requests released together (distinct
prompts of L tokens, `max_tokens = 64`); after the burst drains and 5 s idle, one probe request of 512 tokens. 5090: the
9B at `MEMRA_CTX=65536`, B = 32, L = 6,144. Target card: the 27B, B = 64, L = 30,720. Arms `off` (door unset) and
`on` (`=1`), orders O1 (off, on) and O2 (on, off): 4 boots per card.

### 1.5 Clauses

- **W1 exact books.** On both arms, the probe's `[admit-predict]` line reads `booked_bytes=0` and `booked_real=0`
  (nothing in flight is booked), and the final `/metrics` `admission_booked_bytes` is 0.
- **W2 one release per session.** On `on`, every admitted burst and warm request prints one `w-release` line before its
  retire, with `bytes` equal to the `W` printed for it at admission (`[admit-book] w-booked id=<id> bytes=<W>`, printed
  on both arms), and no session releases twice.
- **W3 identity.** Every request's completion digest equals the other arm's in each order.
- **W4 health.** No `CUDA_ERROR_OUT_OF_MEMORY` line, no 503, no crash line.

Readings, no bound: the peak shadow `booked_bytes` and `booked_real` across the burst's `[admit-predict]` lines per arm;
the count of `verdict=reject-kv` shadow verdicts per arm (what the enforcing door would refuse, O6's input); the time
from each session's admission to its release.

### 1.6 Price

Code: about 0.3 agent-day. Cells: about 40 min on the 5090, about 1.5 h on the target card.

### 1.7 Addendum A (2026-09-26, while coding, before any cell)

The release site is after the tick's command drain and before admission, not at the loop head: a lane-A census window
pins the loop head's first statements, and the drain-then-admit point is the same seam for the admission that
follows (an admission reads the books after the release either way). No clause, bound or reading changes.

## 2. Results

Written after the runs. Section 1 is unchanged.
