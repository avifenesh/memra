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

### 2.1 The target card (the tenth sitting, one RTX PRO 6000 Blackwell Workstation Edition at 600 W, 2026-09-26 13:51 to 14:32Z)

Binary built on the box from `21b081ee1`, sha256 `97d2021d...aa540c0c6`; receipts at `pro-single-day45/box/` (107 files,
the box manifest checked, the binary by hash only). The boot-derived predictive budget reads `budget_bytes=65920413392`.
Verbatim:

```
DAY45 W4 card=pro6000 boot=O1-off oom_lines=55 crash_lines=0 r503=53 -> FAIL
DAY45 W1 card=pro6000 boot=O1-off probe_booked=0 probe_booked_real=0 metrics_booked=0 -> PASS
DAY45 READING card=pro6000 boot=O1-off predict_lines=68 peak_booked_shadow=231005913608 peak_booked_real=231194498056 shadow_reject_kv=47
DAY45 W4 card=pro6000 boot=O1-on oom_lines=55 crash_lines=0 r503=53 -> FAIL
DAY45 W1 card=pro6000 boot=O1-on probe_booked=0 probe_booked_real=0 metrics_booked=0 -> PASS
DAY45 READING card=pro6000 boot=O1-on predict_lines=68 peak_booked_shadow=231005913608 peak_booked_real=231194498056 shadow_reject_kv=47
DAY45 W2 card=pro6000 boot=O1-on w_booked=69 w_release=11 twice=[] bytes_mismatch=[] never_released=['cmpl-1a4bf5d0a3310c03a06b4e3159eb623d', 'cmpl-23dd300f87f0282c34f953f2ada46c52', 'cmpl-27b175e645c90837e0676855049a8927', 'cmpl-2d451e437d86201e87379c00550246d3'] -> FAIL
DAY45 W4 card=pro6000 boot=O2-off oom_lines=55 crash_lines=0 r503=53 -> FAIL
DAY45 W1 card=pro6000 boot=O2-off probe_booked=0 probe_booked_real=0 metrics_booked=0 -> PASS
DAY45 READING card=pro6000 boot=O2-off predict_lines=68 peak_booked_shadow=231005913608 peak_booked_real=231194498056 shadow_reject_kv=47
DAY45 W4 card=pro6000 boot=O2-on oom_lines=55 crash_lines=0 r503=53 -> FAIL
DAY45 W1 card=pro6000 boot=O2-on probe_booked=0 probe_booked_real=0 metrics_booked=0 -> PASS
DAY45 READING card=pro6000 boot=O2-on predict_lines=68 peak_booked_shadow=231005913368 peak_booked_real=231194497816 shadow_reject_kv=47
DAY45 W2 card=pro6000 boot=O2-on w_booked=69 w_release=11 twice=[] bytes_mismatch=[] never_released=['cmpl-01b57d15a858c30c3edfc465a7a8698d', 'cmpl-024f302e88e494c79bd90fd8b4a0f146', 'cmpl-07d03b7ca147010b84e26eed69f4be0c', 'cmpl-0a92c980009220e0eec841d6c8a719ea'] -> FAIL
DAY45 W3 card=pro6000 order=O1 rows=69 differ=['burst-5', 'burst-7'] -> FAIL
DAY45 W3 card=pro6000 order=O2 rows=69 differ=[] -> PASS
```

**Clauses, as they read, and where each failure sits (placed before any fix):**

- **W1 PASS on all four boots**: the probe reads `booked_bytes=0 booked_real=0` and the final `/metrics` booked figure
  is 0 on both arms; both books are exact at idle.
- **W4 FAIL on all four boots, both arms, identically** (55 OOM lines, 53 of the 64 burst requests end `503`
  overloaded). Placed: without an admission door the physical gate admitted all 64 sessions of 30,720 tokens (the real
  book peaks at 231 GB against a 66 GB predictive budget; the shadow door would have refused 47), and the batched prime
  then ran out of memory (`[prime-batch] failed after a partial pipeline wave (DriverError(CUDA_ERROR_OUT_OF_MEMORY,
  ...)); dropping tainted sessions`). That is today's program without `MEMRA_ADMIT_BY_MEMORY` on a burst past the card
  (the door's own subject, O3), the same on both arms; the W release is not in it. The cell's shape was wrong for this
  item: it overloads the card before the books matter.
- **W2 FAIL on both `on` boots** (`w_booked=69 w_release=11`, no double release, no byte mismatch). Placed by the ids:
  the 58 never released are the 53 sessions the prime-batch OOM dropped before their primes completed, and the 5 short
  requests (the 4 warm and the probe) that primed and retired inside one tick. Both exits retire the full charge at the
  retire seam by design (1.1: "a session that retires before its prime completes still carries `W` and retires it");
  W2's wording did not name them. W1's exact books confirm the retire side.
- **W3 FAIL in O1, PASS in O2.** Placed: `burst-5` and `burst-7` are `503` on one arm and `200` on the other (the OOM's
  drop set differs between boots); the 15 requests that are `200` on both arms have equal digests.
- **Readings:** the peak books are equal on both arms (231.0 and 231.2 GB) because the burst's admissions all precede
  its first prime completion; the release moves the books only after that, which this shape never reached.

### 1.8 Addendum B (2026-09-26, after 2.1, before any code of the revision)

- **The cell runs with `MEMRA_ADMIT_BY_MEMORY=1` on both arms**, so the burst is shaped by the memory door (defer or
  the typed 429) and no prime runs past the card; the door's decisions do not read the two books, so both arms serve
  the same admissions. The predictive shadow stays on (log only).
- **A second wave:** 10 s after the first burst's last admission, a second wave of B/2 requests (distinct prompts, the
  same L and `max_tokens`), so admissions arrive after some primes completed and the release can move the shadow
  verdicts. Readings add, per wave and arm, the `reject-kv` shadow verdicts and the booked figures at admission.
- **The retire receipt:** at retire, a session whose `W` is still booked prints `[admit-book] w-retire-unreleased
  id=<id> bytes=<W> reason=<prime-incomplete|same-tick>` (the door on only). W2 then reads: every booked id prints
  exactly one of `w-release` or `w-retire-unreleased`, bytes equal, no id twice.
- **W3 compares the requests that are `200` on both arms** and prints the status mismatches as a reading.
- W1 and W4 unchanged. Cells: both cards, both orders, 4 boots each.
