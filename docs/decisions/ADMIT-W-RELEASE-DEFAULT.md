# The prefill workspace leaves the admission books at prime completion on its receipted cards (2026-11-04, Max-Q 2026-11-05)

**Chosen:** on the full-power RTX PRO 6000 Blackwell Workstation Edition and, since 2026-11-05, the
Max-Q Workstation Edition (each from its own cells), each session's prefill workspace term `W`
leaves both admission books (the predictive shadow book and the real book
`admission_booked_bytes`) once its prime completes, instead of staying booked for the session's
whole life; the retire seam subtracts the reduced charges, so both books stay exact. The default
is keyed by device name once per worker (`w_release_for_device` in
`crates/memra-server/src/worker.rs`). `MEMRA_ADMIT_W_RELEASE` is deleted.

**Rejected:**

- Keeping the door default-off past its decide-by (2026-10-10). The owner's 2026-11-04 ruling: a
  winner with a positive, correct receipt becomes a default keyed to the card it came from, and
  its switch goes.
- One default for every card. Only the two Workstation editions ran the release cells. The RTX
  5090 class's halves never ran (`rtx5090-day45/` holds only its chain script, and
  `rtx5090-day46c/` ran the `shadow` and `enforce` arms but not `enforce-wrel`), so the 5090, the
  Server Edition and every other card keep the whole-life booking until a cell on them passes.
- Deleting the release because only the default-off predictive door reads the books. The real
  book is also the published `admission_booked_bytes` figure, and the receipts pass; the ruling
  flips a passing door rather than tying it to another door's decision.

**Measurement that settled it:** one RTX PRO 6000 Blackwell Workstation Edition at 600 W, the 27B.

- `research/spill-b-20260919/DAY45.md` 2.2 (addendum B: the memory door on both arms, a 64 x
  30,720-token burst, then a second wave of 32), both orders, 4 boots: every clause PASS. W1 both
  books 0 at idle on every boot; W2 each of the 53 booked sessions printed exactly one
  `w-release` (48) or same-tick `w-retire-unreleased` (5) line, bytes equal, none twice; W3 53
  rows `200` on both arms with equal digests; W4 no OOM, 503 or crash. On that shape the release
  landed after the second wave was refused, so it moved no verdict there.
- `research/spill-b-20260919/DAY46.md` 2.2 (addendum C: the enforcing predictive door,
  `enforce` against `enforce-wrel`, the second wave released at the burst's first `200`), both
  orders: P1 to P5 PASS on all four enforcing boots, no OOM, every admitted line within the
  65.88 GB budget, requests `200` on both arms with equal digests. The value reading: by the
  second wave the release took the book from 60.60 GB to 20.47 GB, and the enforcing door
  admitted 11 of 32 second-wave requests with the release against 1 of 32 without it (39
  against 29 over the boot).

The Max-Q row is `research/spill-b-20260919/DAY45.md` 2.3 (`pro-maxq-day45-46/`), pre-registered
in DAY45 addendum C and DAY46 addendum D before any Max-Q boot: both halves replayed at their
deciding trees and binaries on one Max-Q card (index 2) beside training on the box's other cards, at 280 W
for all but the first minutes of each part (the runner set 300 W; the box's cap timer restored
280 W), the same 27B, arms, orders and client. Every clause PASS, 12 of 12 and 24 of 24: both books
0 at idle, each of the 54 booked sessions one `w-release` (49) or same-tick retire (5) line, equal
digests, no OOM or 503, typed 429s, every admission within the 65.98 GB budget. The value reading, on a shared box
with no bound: the enforcing door admitted 12 of 32 second-wave requests with the release against
1 of 32 without it, the book at the second wave 20.47 GB against 60.60 GB, in both orders.

**Scope:** the books change and nothing that produces tokens reads them, so outputs are
unchanged by construction (W3 and P4 measured it). The verdict-moving effect needs the enforcing
predictive door (`MEMRA_ADMIT_PREDICT_ENFORCE`), whose own default is a separate decision.
