# WP-B day 46: O6, the enforcing predictive door on the fuller charge

OWED.md O6. DAY24 made the predictive book read the physical gate's own terms (the "fuller charge": `A + W + D` beside
the re-keyed context, `booked_real - kv_hat` equal to the context bracket to the byte) and ran it in shadow mode only:
"the enforcing door was not exercised (it would now refuse on the fuller charge, which is the intended change and needs
its own cell against a budget arm)". This day is that cell, with O4's workspace release (DAY45) as a second enforcing
arm, since a lifetime `W` in the book refuses on workspace that is no longer live. `MEMRA_ADMIT_PREDICT_ENFORCE`
stays default OFF; no default moves and no budget value is chosen.

## 1. Pre-registration

Committed and pushed before any day-46 code and before any day-46 cell. Nothing in section 1 changes after a number is
seen; a failed clause is recorded as it reads and a revision is a new, dated addendum pushed before its code. This day
is text only until DAY37 addendum G's repro and the ninth and tenth sittings have read (the lead's order, 2026-09-26).

### 1.1 The arms (existing doors; no new code is planned)

One binary (the lane tip with DAY45's `MEMRA_ADMIT_W_RELEASE`), the budget at its boot-derived default
(`MEMRA_ADMIT_PREDICT_BUDGET_MB` unset; the boot line's `budget_bytes` is recorded):

- `shadow`: `MEMRA_ADMIT_PREDICT_SHADOW=1` (log only, the control: every request runs).
- `enforce`: `MEMRA_ADMIT_PREDICT_ENFORCE=1` (a `reject-kv` verdict becomes the typed 429 with `Retry-After`).
- `enforce-wrel`: `MEMRA_ADMIT_PREDICT_ENFORCE=1 MEMRA_ADMIT_W_RELEASE=1`.

`MEMRA_ADMIT_BY_MEMORY` unset on every arm (the predictive door alone). Orders O1 (`shadow`, `enforce`,
`enforce-wrel`) and O2 (the reverse): 6 boots per card.

### 1.2 The cell (`day46-client.py`, one boot = one arm)

DAY24 1.1's sequence (one 64-token warm request; (a) four concurrent requests of about 6,000 prompt tokens,
`max_tokens=96`; (b) one of about 12,000; (c) the four of (a) again), then 5 s idle, then a burst of B requests
released together (distinct prompts of L tokens, `max_tokens=64`), then 5 s idle and one 512-token probe. 5090: the 9B
at `MEMRA_CTX=65536`, B = 32, L = 6,144. Target card: the 27B, B = 64, L = 30,720; the sequence's prompt lengths are
DAY24's on both cards.

### 1.3 Clauses

- **P1 typed refusals.** On both enforcing arms every refused request is a 429 with `Retry-After` in 1 to 60 and a
  `[admit-predict] ... verdict=reject-kv ... enforce=1` line with the same id; no other non-200 on any arm.
- **P2 no OOM.** No `CUDA_ERROR_OUT_OF_MEMORY` line, no parked prefill or step OOM, no 503, no crash line on any boot.
- **P3 within the budget.** On both enforcing arms, at every admitted request's `[admit-predict]` line,
  `booked_bytes + kv_hat <= budget_bytes`.
- **P4 identity.** Every request admitted on an enforcing arm has the completion digest of the same request on
  `shadow` in the same order (admission chooses who runs, not what they output).
- **P5 the release reaches the door.** On `enforce-wrel`, every admitted session prints one `w-release` line before its
  retire (DAY45 W2's rule), and the probe's line reads `booked_bytes=0 booked_real=0` on every arm (the books are
  exact at idle).

Readings, no bound: the burst's 200 and 429 counts per arm; the sequence's admitted count per arm (the day-24 rows);
the time to each 429; the admitted requests' TTFT p50 and p95; the peak `booked_bytes`; the `budget_bytes` the boot
derived. Each median states N and the 250 ms regime.

### 1.4 What the reading decides

Nothing moves a default. The reading is the owner's input for the enforcing door: how many of the burst the fuller
charge admits, and how many more the workspace release admits, with no OOM. If P2 fails on an enforcing arm, the
cause is quoted and the door's charge is revised under a new pre-registration.

### 1.5 Price

Code: about 0.2 agent-day (the client and reader; no engine or server change). Cells: about 1 h on the 5090, about
2 h on the target card.

### 1.6 Addendum A (2026-09-26, from DAY45 2.1 and 2.2, before any day-46 code or cell)

Two facts from DAY45 on the target card change this cell before it is built. No bound changes.

- **The shadow arm OOMs at this burst by construction.** DAY45 2.1: with no admission door, the physical gate admits
  all 64 sessions of 30,720 tokens and the batched prime runs out of memory (55 OOM lines, 53 of 64 end `503`), the same
  on both arms. `shadow` here has no door either, so P2 as written would fail on the control for a reason already
  placed. P2 therefore applies to the two enforcing arms, and on `shadow` the OOM, 503 and crash counts are a reading
  (the before). P1's "no other non-200 on any arm" also becomes the enforcing arms only. P4 compares the requests that
  are `200` on both boots of the pair, and prints the status mismatches as a reading (DAY45 addendum B's W3 rule).
- **A burst released together is booked before any prime completes.** DAY45 2.2: a 30,720-token burst session's
  `w-release` lands 217 to 459 s after its booking, so every admission of that burst reads the same books with or
  without the release, and `enforce` and `enforce-wrel` would admit the same burst. So the cell gains a second wave:
  when the burst's first request completes (its prime has completed, so its `W` has been released on
  `enforce-wrel`), B/2 more requests (distinct prompts, the same L and `max_tokens`) are released together. Then 5 s
  idle and the probe, as before. Readings add, per wave and arm, the 200 and 429 counts and the `booked_bytes` at the
  wave's first admission. The second wave's admitted count on `enforce-wrel` against `enforce` is the release's value.
- **P5 uses DAY45 addendum B's W2 rule:** on `enforce-wrel` every booked id prints exactly one of `w-release` or
  `w-retire-unreleased` (bytes equal, none twice), and the probe reads `booked_bytes=0 booked_real=0` on every arm.
- Everything else of section 1 stands. The day stays text only until the ninth sitting (DAY44) has read.

## 2. Results

Written after the runs. Section 1 is unchanged.
