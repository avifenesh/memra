# Arm B successful-build provenance

- Classification: derived-default candidate
- Source commit: `772235e526b64f6ed2f02aa7cca853b9d858e299`
- Budget behavior: geometry-derived naked default, free-VRAM-clamped
- Instrumentation: same refusal counters/publication and one-time warning as arm A
- Build worktree: clean detached exact-commit checkout
- Build command class: `cargo build --release -p memra-server -j 1`
- TMPDIR: `/home/avifenesh/projects/wt-cx-budgetsize/target/lane-tmp/candidate`
- CARGO_TARGET_DIR: `/home/avifenesh/projects/wt-cx-budgetsize/target/candidate-772235e52`
- Build result: success in 4m19s
- Frozen binary: `target/bench-binaries/cx-budgetsize/arm-b-772235e52-memra-server`
- Binary size: 52,789,328 bytes
- Binary SHA-256: `8cf97fb0771caee87ac73b86186c7127ca91d3942c1dad3212f00d33d49e4840`
- Raw compiler output: `candidate-build.log`
