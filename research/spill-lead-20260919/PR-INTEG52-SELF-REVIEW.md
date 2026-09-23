# Self-review: integ52 (A day 32: the H2D half of Move 2 owed item 1, design H)

Author's review of the full diff `main..lane/spill-integ52-20260923`, posted as a PR comment per the owner rule.

## What the diff is
Engine and server, all behind the default-OFF door `MEMRA_KV_HOST_CONTRACTS`:
- `tier_transfer.rs`: `submit_h2d_spans` / `take_h2d_spans`.
- `pinned_host.rs`: `enqueue_to_device_f32`.
- `lib.rs`: `Engine::alloc_f32_uninit`.
- `worker.rs`: the log-only owner-segment field, the resident `Arc<Vec<f32>>` form, the `Fill` job, the parked
  `Filling` phase, and the settle's take-back.
- `memra-tier`: `conformance/h2d_span.rs` and its binding.

Other files:
- `docs/FLAGS.md`: `contract-promote-spans` on the existing `MEMRA_KV_HOST_FAULT` row, with no new name.
- `docs/TESTING.md`.
- `tools/kv-host-contract-fault-gate.sh`: the `promote-span-refusal` cell.
- Research: A's DAY32.md, STATE.md, OWNER-THREAD-OFFLOAD.md, the receipts `pro-single-day32/box/` and
  `rtx5090-day32/`, and the INDEX row.
- This PR: the integ52 record section (ruling 47), the CPU and 5090 batteries, and this file.

## What I checked
**`submit_h2d_spans`**
- Every refusal before the enqueue is whole: no copy stream, a bad ticket state, a non-H2D batch, spans already
  attached, an empty batch, a length mismatch or unwritten source, a destination owned by another stream, or a
  failed fence. In each case every span comes back.
- The copy stream waits on a fresh owner-stream event before the first copy, so each destination's allocation is
  ordered before it.
- An enqueue or event error after the first enqueue keeps every span and quarantines the ticket. This matches day
  30's D2H rule 5.

**`take_h2d_spans`**
- Returns `NotReady` until every item's and every span's event is observed and the owner stream's wait is installed.

**`enqueue_to_device_f32`**
- Refuses a length mismatch or an unwritten source before the FFI call.
- Keeps the destination's write record alive across the enqueue.
- Its `unsafe` contract names who keeps the source and destination alive until the event.

**`alloc_f32_uninit`**
- An uninitialized device destination, like the engine's existing `uninit` helpers. It is handed out only after
  the copy has written the whole range and landed.

**The `Filling` step**
- Runs through the existing contract settle callers and the `Poll` / `Block` wait.
- A fill past `HOST_HASH_DEADLINE`, or a helper that is gone, latches the tier, so a parked request cannot wait
  unbounded.
- B3 shows every fill landing at its first poll.

**Not re-derived**
- I did not re-derive every line of the 1591-line `worker.rs` diff. Its source censuses, the twelve-cell fault gate
  (160 ok on both cards) and B1 carry it.

**A's verdicts**
- The B1 to B5 lines in the record are copied from DAY32 section 4.
- The DAY28 clause 1b reading comes from DAY32 section 5 and is recorded as it reads. Day 32 did not re-register
  that clause, and no threshold moved.

**Checks**
- CPU battery on `e009df6d3`: 15 of 15 rc=0.
- RTX 5090, binary `de9a0888` hashed after serve-smoke's build, one hold: serve-smoke, the engine span cells
  (7, serial), the worker cells (16, serial), identity default ON, fault default and plain (160 ok each), and hit
  OFF/ON. All green.
- BOX3 is not rerun on the merged tree. The merge adds only main's #677, #681 and records, and the box was reclaimed
  after A's sitting. Its loss is recorded in the section.
- No provider name, host, id or price in tracked files. No em dash in authored lines.

## Push regime
Engine source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced,
logged). Every other hook ran. No tag: this is an engine change under a default-OFF door with no board move.
Revuto: if capped or unavailable, this comment is the review.
