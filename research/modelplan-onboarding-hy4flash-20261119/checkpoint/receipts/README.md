# Step 7 receipts

- `verify/final`: `memra model verify checkpoint <artifact dir at Avifenesh/Hy4-flash-NVFP4@f9eac06cf022> --against hy_v4_flash --oracle <tape set> --native-runner hy4f-run` on one GB10 (sm_121a), 2026-11-20 12:35 to 12:57: `verified=Checkpoint`.
- `verify/red-*`: the same command with `red-runner.sh` and one wrong program each; every run fails (`run.log`).
- Binaries (`bins.sha256`): built for sm_121a from a tree whose `crates/` equals this commit's except three doc-comment lines of `crates/memra-cli/src/checkpoint_tape.rs` (removed after the build); `hy4f-run` is the binary the checkpoint receipt names (`native_runner_sha256`).
- The oracle and band tapes ran on CPU on the same machine (torch 2.15.0+cpu, 20 threads, transformers PR #50903 at b865850d0532); `tapes/*.tape.json` record each run's seconds and the pinned modeling blob.
