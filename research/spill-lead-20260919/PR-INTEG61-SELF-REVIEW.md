# Self-review: integ61 (C days 59 to 62: MEMRA_MOE_PREFETCH=1 decided on the target card, the door's gap attributed, I11 and I12)

Author's review of the full diff `main..lane/spill-integ61-20260925`, posted as a PR comment per the owner rule.

## What the diff is
- Engine and tier, behind the default-absent CLI door `--experts-via-tier`: I11 changes 1 to 5 (the ticket metadata
  allowance computed once per record, among others) and I12 (finished leases retire only where a lease is taken).
  Change 6 read flat on the CPU and was reverted before any card cell.
- `run_gen.rs`: the log-only `--moe-dispatch-clock`.
- `docs/FLAGS.md`: the `MEMRA_MOE_PREFETCH=1` row now carries decide-by 2026-10-04 and its deciding cell.
- No new `unsafe`, no new `MEMRA_*` read.
- Research: C's DAY59 to DAY62, the target-card receipts (`pro-single-day61/`, 635 files); the integ61 record (ruling
  56), both batteries, this file. The branch is a fast-forward over main `7bf457384`.

## What I checked
- The I11 changes move work off the per-ticket path without changing what a lease carries; I12 moves the retire point,
  not the condition (a lease finishes only after its copy event completes).
- The dispatch clock is log-only and off unless the flag is passed; its cost on the door (0.047 ms per token) is
  recorded `over_bound` against its 0.044 bound, as it reads.
- One numeric program per request: the prefetch arm passed the tape, speculative and serving identity gates on the
  target card; the door's identity checks are unchanged.
- The verdict lines in the record are copied from DAY59 to DAY62. The tuned door still loses to the prefetch arm, and
  the record says so plainly; `MEMRA_MOE_PREFETCH=1` qualifying as the target card's naked default is the owner's call.

## Batteries
- CPU battery 15 of 15 rc=0 (server 917, engine lib 567, portable 384 with 0 skipped, tier 297, pytest 87).
- The local 5090 still needs a reset; the GPU battery ran on BOX12 (an RTX PRO 6000) with the same 9B model under the
  pair lock: every cell green (serve-smoke 0 failed, identity 12 ok, fault default and plain 229 ok each, hit OFF/ON
  61 and 68, admit-mem-burst, spec-ctx-edge, engine span 10 and worker 18).

**Hygiene:** no provider name, host, id or price in tracked files. No em dash in authored lines.

## Push regime
Engine source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced,
logged). Every other hook ran. No tag: default-absent CLI door and a default-OFF flag only. Revuto: if capped or
unavailable, this comment is the review.
