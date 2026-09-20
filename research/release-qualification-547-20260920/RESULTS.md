# Content-bound release qualification — #547

Status: CPU implementation and controls complete; native qualification pending. These results
are not GPU evidence or permission to merge/tag. Independent review and the coordinator's
fresh native build/capture are next. No rig was rented or GPU work performed by this lane.

Checked code source: `b2c1e8b236189f8dddd43e459480b01d7157109b`; accepted main
`cf4f317e68b9cedea4f4e195caee83a598db421e` was integrated without conflicts. The exact
changed source bytes are listed in `source-sha256.json`. This publication adds CPU records only.

## CPU checks

- `python3 tools/test_release_qualification.py`: 14 tests passed. Temporary fabricated records
  exercise binding/refusal only, including source changes, all runtime dependency classes,
  stale binaries/models/rig, missing/duplicate/failed/narrowed cells, the fixed skip ceiling,
  a fresh checkout, real pre-push behavior, and safe publication. They do not mint native proof.
- `bash tools/test_flags_guard.sh`: 12 assertions passed, including the explicitly retired
  perf waiver. The qualification arm's real-hook cases live in the first suite.
- `python3 tools/test_release_battery_coverage.py`: 12 tests passed, including the original
  #546 controls and raw-output persistence before verdict parsing on success/failure.
- `python3 -m unittest tools.test_check_hardware_gate`: 6 tests passed. The initial direct
  invocation lacked the repository package import path; that setup failure is retained.
- Flags, docs registry, publication/stub/architecture censuses, perf-board generation check,
  `cargo fmt --all -- --check`, Python syntax and `git diff --check` passed.
- ShellCheck passes with the three existing diagnostics excluded: SC2034 in release-battery,
  SC2016/SC2181 in the inherited flags fixture. No new ShellCheck diagnostic was observed.
- `test_release_battery_card.sh` explicitly skipped because this Mac has no nvidia-smi.

The integrated source inventory includes all 48 crate paths added/modified between the
initial base and accepted main, including engine Cargo/build scripts, KV, tier, worker and
GGUF source dependencies. A scan of the generated source-inventory JSON against the public
boundary patterns had zero matches. No runtime implementation was edited by this lane.

## Native queue request

After independent review, use a fresh build at the final pushed head, then the existing
non-serving rig's physical GPU0 through `/usr/local/bin/memra-gpu-run`. CUDA UUID, the
lease and the battery's NVML GPU0 query must agree; other-card selection remains #264.
Run the normal release roster with the actual named kernel oracle files, seal only after
the wrapper records successful cleanup, preserve the exact ELF archive off the rig, and
bank the manifested evidence. Commands and profile requirements are in
[RELEASE-QUALIFICATION.md](../../docs/RELEASE-QUALIFICATION.md).

The tag workflows require valid records and exact binary equality for their OS profiles.
A fresh CI build does not inherit GPU qualification from a matching source revision.
No source/binary/model/numeric/hardware/cell proof has been inferred from these CPU tests.
