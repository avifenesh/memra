Author's review of the full diff `main..lane/spill-integ73-20260927`, posted as a PR comment per the owner rule.

## What the diff is
- `worker.rs` (lane A, T-H'): the hash helper fills and hashes on up to 8 threads (`host_hash_threads`, `host_scoped_map`). It is T-H's code re-applied, adopted on the target card by DAY65's T-H' cell. The chain cell's helper fell 82.8 to 29.8 ms and its wall 370.8 to 313.8 ms, per order, with every gate green.
- `worker.rs`, `parallel.rs` (lane B, DAY49 addendum F): `MEMRA_BATCH_OOM_RECOVER` resolves per card class at worker boot. Unset means ON on the RTX PRO 6000 Blackwell class and OFF elsewhere. `0` is the rollback seam (decide-by 2026-10-11), `1` forces it on, and the boot prints `[batch-oom] recover=... source=...`. The decision record is `docs/decisions/BATCH-OOM-RECOVER-DEFAULT.md`.
- `worker.rs`, engine (lane B, memra#476, DAY51): every model pre-grows its FA partition pools right before boot calibration, sized from the ModelPlan's full, sliding-window and MTP layers at the served context and `max(wave cap, spec K+1)` rows. Boot cost: 129 MiB for the 9B at 65,536 and 774 MiB for the 27B at 262,144 on the target card. That is the reachable envelope, which a batch would otherwise reach mid-life. A failed grow is loud and non-fatal.
- Lane B: DAY44 R1 as the owner accepted it (addendum D); DAY48's physical debt line on every admission under `MEMRA_ADMIT_PREDICT_VG_DEBT=1`, ending in the request id.
- Records: lane A's DAY65, DAY68 and the T-H' sitting; lane B's DAY44, DAY48, DAY49 and DAY51.

## What I checked
- The batch-OOM default is resolved once from the device name at worker boot, before the serve loop's two reads. A read before the init would resolve with no card default. The boot line prints the armed value beside its source, so a mismatch would show in the log.
- T-H' is the sitting's program byte for byte (the rebase applied with no conflict), and its own CPU cells are green, including the red arm.
- The pre-grow contributes nothing for MLA, GatedDeltaNet and Kimi layers. It takes the ladder's maximum over every key length up to the served context, because the split count is not monotone across rung edges.
- The fixture pin holds. No provider name, host, id, price or city in the added lines.

## Batteries
- CPU battery 16 of 16 with CI's gates job (server lib 993, engine lib 601).
- GPU battery on an RTX PRO 6000 WS, running after lane B's DAY51 cell: integ72's cells (every serving gate now boots with the pre-grow and the batch-OOM default), the pause and tier gates, and the health gate arms g, h and j. Its results follow as a records commit.

**Hygiene:** no em dash in authored lines.

## Push regime
Engine and server source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development`. No tag. Revuto: if capped or unavailable, this comment is the review.
