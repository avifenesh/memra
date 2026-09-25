# WP-F resumable state (2026-09-25, stopped at NEED BOX)

- Lane `lane/spill-f-20260919`, worktree `wt-spill-f`. Resynced 2026-09-25: the branch was fully
  contained in `origin/main`, fast-forwarded to `5228ff0cd` (no conflicts). Upstream unset so the
  pre-push range is measured against `origin/main`; pushes of this branch carry main's engine
  files relative to the old remote tip, so they go out with
  `MEMRA_RELEASE_QUALIFICATION_MODE=development` (logged by the hook).
- Plan of record: `M1-PREREG.md` (proof chain A, cells B0 to B6, routes C). Ledger: `OWED.md`.
- Done: proof tool `m1-nvme-proof.py` (24/24 fixture, 7/7 live controls, `M1-PROOF-CONTROLS.md`);
  this rig's artifact store and root filesystem proven `nvme-local-direct`; direct-arm alignment
  census of the pinned artifact (0 of 31,488 slices O_DIRECT-admissible); B3 arm lock and prompt
  frozen in `m1-prereg/`.
- **NEED BOX** for the M1 proof and cells: one RTX PRO 6000, docker instance with a host-local
  volume at `/scratch`, whole machine, 600 W. Ranked offers and quotes: private
  `LANE-LOCAL-NVME.md` (worktree root, excluded through the common `info/exclude`).
- CPU work that does not need the box, in order: OWED 7 (aligned over-read in the direct worker),
  8 (per-stage spill counters), 9 (`storage-bench` io_ns), 10 (host/storage sampler), 11 (cache
  regimes), 12 (cell runner), 13 (B2 driver), 14 (collector proof binding, D-owned file).
  B3's `direct16` arm stays refused until 7 lands; B1 through the collector waits on 14.
- Private receipts (not scratch, retained): `~/.local/share/memra-lane-f-private/m1-proof-controls/`.
- Scratch: none. The read-only search dumps under `/tmp/spill-f-m1/` were removed at stop.
