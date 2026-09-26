# WP-F resumable state (2026-09-25, stopped: box pending; CPU prerequisites complete)

- Lane `lane/spill-f-20260919`, worktree `wt-spill-f`, branch contains `origin/main` `5228ff0cd`
  (no newer main at the last fetch). Upstream unset; pushes use
  `MEMRA_RELEASE_QUALIFICATION_MODE=development` (engine files in range; logged).
- Plan of record: `M1-PREREG.md` (A proof, B0 to B6 cells, C routes, B2 amendment). Ledger: `OWED.md`.
  CPU-side registration of the prerequisites: `CPU-PREREG.md`.
- Landed CPU-verified today: proof tool (`M1-PROOF-CONTROLS.md`); OWED 7 direct over-read
  (`owed7/`); OWED 8 stage counters (`owed8/`); OWED 9 storage-bench timing (`owed9/`); OWED 10
  sampler (`owed10/`); OWED 11 cache regimes (`owed11/`); OWED 12 B3 runner (`owed12/`); OWED 13
  B2 driver plus `kv-handoff-gate` seam (`owed13/`); OWED 14 collector patch and test, not
  applied (`owed14/`, lead routes to D); B0 runner (`b0/`).
- Box: the lead rents rank 1 of `LANE-LOCAL-NVME.md` with a 600 GB local volume at `/scratch`
  after lane A releases it (about 17:00Z), runs the proof first, hands it over only on
  `M1-PROOF verdict=PASS`.
- On the box, in order: bootstrap (CUDA 13 toolchain, `apt-get install -y fio`), build
  `run-gen`, `run-spec`, `memra-server`, `kv-handoff-gate`, `storage-bench`, `h2d_probe` at one
  commit; stage and re-hash the artifact on `/scratch/spill-f`; generate the B2 prompts and check
  the manifest; then B0, OWED 7's pinned-pool GPU test, B1, B3 (cold, warm, bounded), `run-spec`
  cells, B2 (1 GiB, 8 GiB), B4, B6. `direct16` enters B3 only after the pinned-pool test passes.
- Private receipts (retained, not scratch): `~/.local/share/memra-lane-f-private/`.
- Scratch: none. No process of this lane is running.
