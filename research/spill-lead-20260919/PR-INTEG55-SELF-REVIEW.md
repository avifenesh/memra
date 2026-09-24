# Self-review: integ55 (B day 35: memra#680's remaining term, the prefix seed booked under the door)

Author's review of the full diff `main..lane/spill-integ55-20260924`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-server/src/worker.rs`: `session_pending_seed_rows`, `seed_entry_bytes`, `prefix_seed_bytes_at` and
  `pending_seed_bytes`. The armed-only `pending_seed` joins `pending_prime` in the one booked reduction
  (`less_pending(pending_booked, ..)`). Three CPU tests.
- `crates/memra-server/src/admit_memory.rs`: the `pending_seed=` field on the memory line, and two tests.
- `docs/FLAGS.md`: the door row gains the seed term.
- Research: B's DAY35.md, the receipts `rtx5090-day35/` and `pro-single-day35/box/`, STATE, the INDEX rows, and the
  day-36 pre-registration (`88bb3b1e2`, which decides nothing here). Also the integ55 record section (ruling 50),
  both batteries and this file.

## What I checked
- Door OFF cannot move. `pending_seed` is 0 unless `admit_memory_cfg.armed`. The census test pins it computed once
  and joined to the single `less_pending` call. `pending_seed_bytes` also returns 0 when the prefix cache has no
  budget.
- No double count. A session books its seed only while `seed_prefix` and `seed_at` are set. The publish path clears
  both whether it publishes or refuses (`s.seed_prefix = false`, `s.seed_at.take()`), so a published entry is counted
  once, as resident.
- The sizing follows `prefix_snapshot_bytes`'s own arithmetic: per-row KV, recurrent state once, and latent rows
  where present, including the TP peers. Vision sessions and sessions without a cache book nothing, matching the
  publish path's refusals.
- The verdict lines in the record are copied from the two day-35 SUMMARY files. The red leg reproduces on BOX4 (3
  parked prefill OOMs on main). The green legs are 0 OOM twice, with G-BOOK PASS. The 5090 has no red for this term,
  which is pre-registered.
- One numeric program per request: V-ID-FIX, V-ID and V-OFF are 16 of 16.
- Checks on `38b35dc5e`: CPU battery 15 of 15 rc=0. RTX 5090 (binary `cd97b015` hashed after serve-smoke):
  serve-smoke, the serial span and worker cells, identity, fault default and plain (160 ok each), hit OFF/ON, the #680
  gate (37 x 200, 27 typed 429s, no OOM, every admit within its booked free) and spec-ctx-edge, all green.
- The review fix `34a4b7f23` (revuto's finding): `prepare_snapshot` evicts inside the budget before a seed
  allocates, so the booking is capped at `prefix_cache_budget_bytes() - px.total_bytes`. The cap can only lower
  the term. A cold burst books as before: B's BOX4 addendum reads G-NOOM 0 twice with the identity terms 16/16, and
  the r2 5090 battery on `34a4b7f23` is all green.
- No provider name, host, id or price in tracked files. No em dash in authored lines.

## Push regime
Engine source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced,
logged). Every other hook ran. No tag (default-OFF door). Revuto: if capped or unavailable, this comment is the review.
