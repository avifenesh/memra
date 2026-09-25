# WP-F resumable state (2026-09-25, on BOX27; box campaign in progress)

- Lane `lane/spill-f-20260919`; build commit on the box `ffff2d89a` (engine source verified
  unchanged at every later checkout). Box checkout `/root/wt-f`; receipts
  `/root/spill-receipts/f-box27/`; private proof `/root/f-private/m1-proof.json`; frozen
  binaries and the staged artifact under `/scratch/spill-f/`.
- Done on BOX27: proof mirrored (PASS); B0 (io_uring refused by seccomp); OWED 7 GPU gate
  (12/12); B1 (360/360 scored under the amended read gate, buffered wins everything); B3 cold
  (unscored: host-level foreign I/O); B3 warm (unscored registered; read-gate rescoring: mmap
  arms 1.19x winners). Mirrored and pushed: parts 1 to 4 (`box27/MANIFEST-part*.sha256`).
- Running: B3 bounded (touched balloon, about 2.8 h from 19:47Z). A background 1 s device sampler
  runs with its pid in `$R/background-sampler.pid` (stop it by that pid only).
- Next, in order: run-spec for worker16 and direct16; B2 at 1 GiB and 8 GiB; B4 (worker16 plus
  the B3 winners in their regime); B6; a second-window rerun of B3 cold; then mirror everything,
  remove `/scratch/spill-f` and `/root/wt-f` scratch, stop the background sampler, report
  `BOX27 RELEASED`.
- Private receipts and sanitized originals: `~/.local/share/memra-lane-f-private/box27/`.
