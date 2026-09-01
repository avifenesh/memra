# Arm A successful-build provenance

- Classification: deliberately metrics-instrumented v0.81.3 budget baseline
- Source commit: `093a214a9e1bc7170dd655bb417b0fd7fc6d13c8`
- Budget behavior: naked v0.81.3 256 MiB default
- Instrumentation: refusal counters/publication plus the one-time first-refusal warning
- Build command class: `cargo build --release -p memra-server -j 1`
- TMPDIR: `/home/avifenesh/projects/wt-cx-budgetsize/target/lane-tmp/baseline`
- CARGO_TARGET_DIR: `/home/avifenesh/projects/wt-cx-budgetsize/target/baseline-instrumented`
- Build result: success in 3m25s
- Frozen binary: `target/bench-binaries/cx-budgetsize/arm-a-093a214a9-memra-server`
- Binary size: 52,775,640 bytes
- Binary SHA-256: `ec0c2fed4aa25fa904ab072fc2af53cee34dbee7c352d0eefb257c52f88a2a2f`
- Raw compiler output: `build-success.log`

The older `build.log`, `build-retry-serial.log`, and `build-retry-project-tmp.log` are retained as
environmental-failure receipts only. They did not produce the frozen benchmark binary and are not
candidate verdicts.
