# AD-556-1: preserve Step RoPE source extents

The reviewed head `36019232d1479a81166a2fb51cb3d341562682f4` accepted a compact
32-value factor tensor during preflight, but its GGUF contract required 64 values.
The complete fixture reproduces that real binding failure in [before.log](before.log).

Step config now retains the source tensor shape from GGUF headers. The pack selects
full-head storage only when the source declares that valid extent; compact sources
retain the canonical consumed-width contract. Preflight retains the accepted shape
alongside the validated factors. Neither path pads, truncates, or substitutes bytes.

The synthetic two-layer fixture includes every required tensor. Compact `[32]` and
full-head `[64]` forms pass preflight, raw and normalized contract binding, and the
actual `memra model inspect` process. `[1]`, `[48]`, `[65]`, `[2,16]`, `[32,1]`, and
`[64,1]` fail both preflight and binding. The engine test calls `artifact_costs`, the
binding entry used by automatic placement before CUDA capacity queries; it also
asserts that full storage costs exactly 128 additional bytes. This fixture proves
admission and accounting, not model-scale numerical correctness or support status.

## Local verification

Rust/Cargo 1.97.1 on aarch64 macOS. [manifest.json](manifest.json) hashes the final
changed source files and logs. `after.log` is the initial focused confirmation;
`cpu-suites.log` is the final source run after adding source-shape assertions.
The public copy redacts one Rust-generated test executable basename caught by the
build-fingerprint policy; all test output is unchanged. The manifest records the
original log hash, and the unredacted bytes remain in private task evidence.

| Command | Result / evidence |
|---|---|
| `cargo test -p memra-gguf -p memra-cli` | [Pass](cpu-suites.log): GGUF lib 290 reported passes, 1 ignored; CLI lib 11 passes; CLI process integration 1 pass; GGUF dequant bin 7 passes |
| `MEMRA_GGUF_SKIP_BUDGET=12 python3 tools/skip-census.py run --budget-var MEMRA_GGUF_SKIP_BUDGET --min-passed 290 -- cargo test -p memra-gguf --lib` | [Pass](skip-census.log): 12 artifact-dependent skips explicitly named; these are included in libtest's 290 reported passes |
| `cargo test -p memra-cli --test step_rope_contract` | [Pass](cli-after.log): eight source shapes through the actual CLI |
| `cargo clippy -p memra-gguf -p memra-cli --all-targets -- -D warnings` | [Fails](clippy-strict.log) only on the pre-existing macOS `AsRawFd` unused import in `source.rs:20` |
| Same clippy command with `-A unused-imports` | [Pass](clippy-macos.log); supplemental local lint check, not a strict-clippy pass |
| `cargo fmt --all -- --check`; `git diff --check` | Pass |

The initial lint failure for the test's large shared contract error is retained in
`clippy-before.log`; the final test preserves that diagnostic with a scoped allowance.
CI now invokes `cargo test -p memra-cli`. Its existing unfiltered engine lib suite
runs `auto_placement_binds_the_actual_step_rope_extent`; that test and strict Linux
clippy remain pending CI at publication. No local CUDA compilation is claimed.

## Native and integration gates still required

The [prior native receipts](../native-20260920/RESULTS.md) remain bound to
`fd7ce385b7861e6b96b98682906b5e8ff6ae31c2` and its preserved executable hashes.
They do not qualify this changed source or a combined integration binary.

On the coordinator's integrated source, rerun the focused Step factor gate, affected
Step kernel cells, Step GGUF PP-2 argmax and K=1..8, and official Step FP8 PP-3
verify-class argmax and K=1..8. Re-run the applicable dense/Ornith checks if integration
changes their shared admission or numeric paths. The compiler, engine, server, CLI,
formatting, and diff gates also apply to the combined tree. Preserve the #542/#544
behavior during replay. Remaining named release-oracle gaps stay with #546/#484.

This review fix changes no kernels or performance defaults. Native execution,
current-main integration, merge, and release are outside this CPU-only repair run.
