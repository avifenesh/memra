# Final arm B successful-build provenance

- Classification: derived-default candidate after MTP/NextN geometry correction
- Source commit: `13b4918ee5cc69b73bd045c036440d065303fd9a`
- Budget behavior: trunk-geometry-derived naked default, free-VRAM-clamped
- Instrumentation: same refusal counters/publication and one-time warning as arm A
- Build worktree: clean lane worktree at the exact source commit
- Build command class: `cargo build --release -p memra-server -j 1`
- TMPDIR: `/home/avifenesh/projects/wt-cx-budgetsize/target/lane-tmp/candidate-13b4918ee`
- CARGO_TARGET_DIR: `/home/avifenesh/projects/wt-cx-budgetsize/target/candidate-13b4918ee`
- Build result: success in 4m16s
- Frozen binary: `target/bench-binaries/cx-budgetsize/arm-b-13b4918ee-memra-server`
- Binary size: 52,789,376 bytes
- Binary SHA-256: `29f1a64e8935bfc5b97ea1e9b6cf02e5fd4b562c05dd844b2b6566a53f9b77a8`
- Raw compiler output: `candidate-build-after-mtp-fix.log`

The prior `772235e52` binary remains frozen only to identify the excluded live diagnostic that
found the MTP/NextN overcount. It is not arm B in the final protocol.
