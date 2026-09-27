# Self-review: integ53 (B day 33: memra#680 fixed under `MEMRA_ADMIT_BY_MEMORY`)

Author's review of the full diff `main..lane/spill-integ53-20260923`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-server/src/worker.rs`:
  - `pending_prime_bytes` (the prefill workspace of sessions still priming) and `AdmissionHeadroom::less_pending`,
    applied to every headroom reading in the admission block.
  - A `verdict=admit` `[admit-mem]` line on every admission.
  - `prefill_oom_parkable` / `park_prefill_oom` on both prefill arms.
- `crates/memra-server/src/admit_memory.rs`: `booked_device_free` and the `pending_prime=` field on the memory line.
- Every new behavior is gated on the door (`admit_memory_cfg.armed`), which is OFF by default.
- `tools/admit-mem-burst-gate.sh` and its client, wired into `tools/local-ci.sh` behind `MEMRA_CI_ADMIT_MEM_BURST`,
  with FLAGS and TESTING rows.
- Research: B's DAY33.md, the receipts `rtx5090-day33/`, STATE.md, and the INDEX row. B day 34's work lands in a
  later integration.
- The integ53 record section (ruling 48), the CPU and 5090 batteries, and this file.

## What I checked
- Door OFF cannot move:
  - `pending_prime` is 0 unless the door is armed, and `less_pending(0)` returns the reading unchanged.
  - `prefill_oom_parkable` checks `armed` first.
  - V-OFF reads 16 of 16 rows equal with no `[admit-mem]` line.
- The reduction applies only to the primary device's reading, the device the prefill workspace lands on.
- The park fires only for a session with `generated` empty and `tokens_emitted` 0, under the existing
  `step_oom_retries` bound (`step_oom_parkable`), so a client never gets a duplicated or partial stream.
- One numeric program per request: V-ID-FIX is 16 of 16 equal on every green shape.
- The diagnosis lines in the record match DAY33 section 1.1, which quotes file:line at `c3eb41d12`.
- Checks on `9f335ac48`:
  - CPU battery 15 of 15 rc=0.
  - RTX 5090, binary `5c557664` hashed after serve-smoke: serve-smoke, the serial engine and worker span cells,
    identity, fault default and plain (160 ok each), hit OFF/ON, the #680 burst gate (40 x 200, 24 typed 429s,
    no OOM, 40 admit lines within their booked free memory) and spec-ctx-edge. All green.
- PR shape:
  - The PR says `Refs #680`, not `Closes`: the PRO 6000 boots are owed until BOX3 is restored.
  - No provider name, host, id or price in tracked files. No em dash in authored lines.

## Push regime
Engine source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced,
logged). Every other hook ran. No tag: the change sits behind a default-OFF door. Revuto: if capped or unavailable,
this comment is the review.
