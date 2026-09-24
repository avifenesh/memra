# Self-review: integ54 (A days 33 and 34: design F, the same-tick fill, and design K, the H2D completion checksum on the hash helper)

Author's review of the full diff `main..lane/spill-integ54-20260924`, posted as a PR comment per the owner rule.

## What the diff is
All of it is behind the default-OFF door `MEMRA_KV_HOST_CONTRACTS`:
- `tier_transfer.rs`:
  - design F: `SpanFillTask` and the copy stream's `span_fill_on_copy_stream` host function;
  - design K: `H2dSourceView`, `DeferredSums` and `supply_h2d_checksums`.
- `pinned_host.rs`: `enqueue_to_device_f32_after_fill`.
- `worker.rs`: the probe submitting in its own tick, the timeline fields, and the landing poll that waits for the
  helper's digests.
- `memra-tier`: `conformance/h2d_deferred_checksum.rs` and span rule 6, with their bindings.
- The fault gate's `digest` cell.
- Research: A's DAY33.md and DAY34.md, the receipts on both cards, STATE, OWNER-THREAD-OFFLOAD and the INDEX rows.
  The integ54 record section (ruling 49), both batteries, and this file.
- No new `MEMRA_*` name.

## What I checked
F's host function:
- It takes back the `Box<SpanFillTask>` leaked for exactly that call, copies each resident plane into its staging
  target with `copy_nonoverlapping` (lengths checked at attach), and drops only host `Arc<Vec<f32>>` planes. No CUDA
  call runs on the driver's callback thread.
- `catch_unwind` keeps a panic from crossing the FFI boundary, and a failed launch reclaims the box.
- The span copies are enqueued after the host function on the same stream, so they read the staging only after the
  fill.

K's view:
- `H2dSourceView` is a `Send` raw read-only view. The engine keeps each source until its view comes back through
  `supply_h2d_checksums`: the ticket cannot land, retire its sources or retire while a view is out.
- An entry dropped unretired leaks rather than frees, so a dead helper leaves no dangling pointer, only a latched
  tier.
- The digest is the same `checksum` over the same bytes, and the `digest` fault cell shows a corrupted byte is still
  refused.

Verdicts:
- The verdict lines in the record are copied from DAY33 and DAY34. F's decision was owed at branch time. Lane A
  day 35 settled it (`cbf52cc16`, `DAY35 F DECISION -> KEEP`), and the record quotes it; it lands in the next A
  integration.
- The 5090 A/B ran with K on both arms, 30 boots in one hold, 30 of 30 replays PASS. F wins by +7.81 and +7.44 ms
  e2e against pair noise of 5.57 and 5.03 ms. F is flat on BOX4, so a per-card win keeps it.

Batteries:
- CPU battery 15 of 15 rc=0.
- RTX 5090 (binary `72efd587`, hashed after serve-smoke): serve-smoke, the engine span cells (10, serial), the
  worker cells (17, serial), identity, fault default and plain (160 ok each), hit OFF/ON, admit-mem-burst and
  spec-ctx-edge. All green.
- BOX4 is not rerun on the merged tree. The merge is a fast-forward over main `25bbb91f5`, and A's sitting ran this
  lane's code.

Hygiene: no provider name, host, id or price in tracked files. No em dash in authored lines.

## Push regime
Engine source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced,
logged). Every other hook ran. No tag: this is an engine change under a default-OFF door. Revuto: if capped or
unavailable, this comment is the review.
