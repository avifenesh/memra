# WP-A day 9 resumable state

- Lane: `lane/spill-a-20260919`; worktree `/Users/avifen/tiyuvta/wt-spill-a`.
- Integration replay merged/pushed: `12122467a014502bfcd050ec0e120932068ff3fb`.
- Final native build/source: `1adf2be3d9f8ea82596dbc5917407e35d597c939`.
- Native binary SHA-256: `3eae930dab5504c89bc5f7f18473c9d5762b4f4bceed87a5d7dd139ef4782be4`.
- **PASS v1.3 device_hand_back native CUDA**.
- **PASS v1.3 transfer_source_retirement native CUDA**.
- Both unchanged canonical schedules are directly invoked with real pending
  CUDA producers; source/destination graphs and host lifetime are independent.
- Existing native schedules, early-drop/mutable-reuse regression, six exact
  4 KiB–256 MiB roundtrips and zero-governor drain: PASS.
- Final receipts: `day9/native/final/attempt-00/`; source/binary identity in its
  parent. Two failed development attempts retained alongside, not relabelled.
- Final collector exit 0; canonical `/tmp/memra-gpu.lock`, `--rig pro-single`,
  250 ms telemetry (56 samples), 600/600 W, empty compute snapshots before/after.
- All 86 remote raw-file hashes matched; final offline replay and six tamper
  controls PASS. `day9/replay.json`, `day9/hash-audit.json`.
- CPU tier tests: 196 PASS; Mac/Linux-target tier+KV checks, CPU clippy
  `-D warnings`, fmt, diff and flags census PASS. Linux-target docs-stub
  engine/gate check PASS. Broader Mac engine check remains blocked by existing
  Linux-only libc APIs in `cpu_experts.rs`/`spill_pread.rs` (raw log retained).
- No A native process/tmux remains. Remote source checkout `/root/wt-a` is clean.
  Final receipt root `/root/spill-receipts/a-day9-final/`; earlier attempt roots
  retained. Temporary git bundles and replay scratch were removed.
- Native results are development correctness only: collector `qualification:false`.
  No serving/default promotion, actual CUDA graph-execution qualification, or
  physical context-loss recovery claim.
- `V13-BINDING.md` maps every frozen step. No frozen-contract gap remains for
  these two bindings. `day9/RESULTS.md` records the full evidence and failures.
- `IO-BASELINE.md` includes numeric +5% screening targets, but io_uring remains
  DEFERRED: fresh five-AB/five-BA same-box control and composed pipeline/tail gates
  still needed. No NVMe ancestry/spill-speed claim.
- Next for lead: integrate the full lane and rerun affected composed/native tier
  callers. Immutable pre-ack H2D reuse is preserved, but the earlier D2H ticket
  retains the host charge until its acknowledgement; D's old comment that H2D
  source retirement alone destroys that host is now stale.
- Access remains lead-owned existing `~/.ssh/cm/box3` socket; check first, never
  create a fresh connection. Retain this active lane for lead integration.
