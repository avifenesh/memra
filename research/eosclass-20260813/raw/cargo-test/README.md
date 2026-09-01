# Cargo unit-test receipt

Source runtime fix: `7cd4561a6bb6221785f62168432fcbda2ce80e28`.

Passing unit targets:

- `DOCS_RS=1 CARGO_BUILD_JOBS=1 cargo test -p memra-engine --lib`: 83 passed, 0 failed,
  1 GPU-dependent test ignored.
- `DOCS_RS=1 CARGO_BUILD_JOBS=1 cargo test -p memra-server`: 221 passed, 0 failed.

`cargo-test.log` retains an excluded broad invocation,
`cargo test -p memra-engine -p memra-server`. Under `DOCS_RS=1`, Cargo also attempted to link all
engine gate binaries and stopped before tests because `decode-dc-gate` had undefined CUDA FFI
symbols (the exact linker output is retained). The two target-specific commands above avoid that
unsupported no-CUDA bin-link shape and exercise all library/server unit tests.
