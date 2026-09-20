# Proposed typed GPU-cache budget refusal

Status: proposal only; no change to `MEMRA_MOE_SLOTS` parsing or behavior.

## Observed seam

`MoeSlotCache::new` converts forced slot count to bytes, caps it to the
hardware headroom, then applies `(budget_bytes / (max_block_bytes + 8)).max(8)`.
Thus forced 0–7 slots still allocates eight, even when the admitted byte budget
cannot hold them. Saturating multiplication and invalid-string fallback also
cannot attest a strict caller budget. The existing host-bank one-record refusal
is a different gate and does not establish GPU admission safety.

## Proposed API

Introduce `CacheBudget::LegacyAuto` and `CacheBudget::Strict { bytes: u64 }` at
the typed constructor boundary, with `try_plan_slots(layout, budget)` returning
an allocation plan or `CacheBudgetRefusal` **before the first CUDA allocation**.
Keep legacy env translation unchanged until separately decided/documented.
The explicit bank qualification installer should request the strict mode via a
typed argument, not reinterpret an existing environment variable.

- Checked addition/multiplication for every exact capacity plus eight-byte tail.
  Overflow returns `InvalidLayout`/an explicit overflow refusal; never saturate.
- Enforce requested bytes and measured hard-headroom independently. Never raise
  the caller ceiling to fit eight slots. Uniform layout requires eight slots
  under the existing algorithm; return `BelowMinimum { requested, minimum }`
  below `8 * (max_block_bytes + 8)`. Do not invent a smaller staging fallback.
- Mixed layouts require a separately proved minimum per capacity class and the
  same per-record shape contract. Refuse unsupported class minima rather than
  silently replacing them with max-sized uniform slots.
- Reserve the exact allocation-plan bytes through the governor before CUDA
  allocation; unwind partial allocations and reservation on allocation failure.
  Keep the original error (budget refusal is not CUDA OOM).
- On refusal, no slots, source reads, pending H2D, or published bank records may
  exist. A collector wrapper may normalize only that exact typed refusal to
  exit 2; unexpected errors remain errors.

## Acceptance for the follow-up

CPU boundary tests: zero, minimum minus one, exact minimum, minimum plus one,
maximum integer overflow, hard-headroom below minimum, and mixed-class refusal.
Check allocation counts and charge restoration on each injected allocation
failure. Native target-card minimum-minus-one must refuse before dispatch;
exact minimum must run generation/speculation with unchanged tapes and
explicit evictions. Run normal legacy default and forced-slot controls as
regressions. If any MEMRA flag behavior changes, update its `docs/FLAGS.md` row
in the same commit, with an owner decision and retained raw gates. This proposal
neither changes a default nor claims those follow-up gates ran.
