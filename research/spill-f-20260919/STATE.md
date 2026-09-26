# WP-F resumable state (2026-09-26, BOX27 campaign complete; box released to the lead)

- Lane `lane/spill-f-20260919`, worktree `wt-spill-f`. Box binaries were built once at
  `ffff2d89a`; every later checkout on the box carried that engine source unchanged.
- BOX27 done in full, receipts in `box27/` (9,569 box files mirrored and verified against a
  box-side full manifest, `MANIFEST-FULL.sha256`; binaries by hash only; volume id sanitized,
  originals private; large sample files stored gzip with uncompressed hashes in
  `EXPORT-MANIFEST.json`). Results and verdicts: `box27/RESULTS.md`.
- Resync note: a rig reboot interrupted this session at about 20:40Z on 2026-09-25; on resume the
  local tip and origin matched (`c288d19c2`), the box's b3-bounded cell had kept running, and it
  was left untouched until it finished.
- Box scratch removed (`/scratch/spill-f`, `/root/wt-f`, helper scripts, `/tmp/f-*`); the
  background sampler stopped by its recorded pid; no lane process or compute app left. The lead's
  files and the receipt directories remain for the destroy.
- Next item: the 5090 halves (OWED 19, 23): the B3 subset and G2 on the local RTX 5090, whose
  storage is already proven (`M1-PROOF-CONTROLS.md`). Open candidates: OWED 17 (mapped
  pinned-host arm), 18 (handoff O_DIRECT arm), 20 (above-RAM artifact), 21 (flag to the owning lane).
- Private receipts: `~/.local/share/memra-lane-f-private/box27/`. Local scratch: none.
