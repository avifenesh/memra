# CPU verification receipt — 2026-09-19

Repository `avifenesh/memra`, implementation commit
`dc8c8ed45fabbb069c0a76c6531170ba272afda4` (`feat(kv): prototype exact tier identities and fenced restore state machine`).
Parent base `c5a33b14ff7b6c75a3cd808dbc0ee4f10aa8b33f`.
Mac CPU-only; no CUDA/GPU, model, serving, network or deployment test.

| Check actually executed | Result | Raw receipt |
|---|---|---|
| `cargo fmt --all -- --check` | exit 0 | raw/fmt.log |
| `cargo check -p memra-kv --offline` | exit 0 | raw/check.log |
| `cargo test -p memra-kv --offline` | exit 0; 38 passed, 0 failed/ignored, 0 doctests | raw/test.log |
| `cargo clippy -p memra-kv --offline --all-targets --no-deps` | exit 0, no memra-kv warnings | raw/clippy.log |
| `bash tools/check-flags.sh` | exit 0; 864 literal reads, no uncovered names | raw/flags.log |
| `git diff --check` | exit 0 | raw/diff.log |

Seventeen new unit tests exercise deterministic hash identity, all identity dimensions,
parent chains, valid-vs-padding bytes, q8_0/q5_1 records, opaque f32/FP8/FP4 layouts,
all/trailing page completeness, owner aliases, malformed/overflow metadata, advisory lookup
race, program/layout substitution, disk-vs-GPU readiness, missing/duplicate/partial/rejected/
short/corrupt/stale completions, active rollback during I/O, mandatory headroom, eviction,
write/prefetch policy, measured-frontier arithmetic, cancellation and shutdown quarantine.
The other 21 existing ring/TP transaction tests also passed unchanged.

One pre-existing Darwin warning is preserved: `memra-gguf/src/source.rs:20` unused
`std::os::fd::AsRawFd`. No out-of-scope source edit to suppress it.

Development checks first passed 35/35. While extending active guard/shutdown tests, two
compile errors occurred and were corrected before the final receipt: an over-broad fixture
edit added `active` to EvictionCandidate (E0560), and the Option-backed shutdown refactor
missed one multiline TransferEngine.poll receiver (E0599). These were compile failures,
not GPU/runtime results; final checks above ran after both corrections.

The first shell command combining formatting and tests with `set -o pipefail` was blocked
by the harness as a shell-variable-dump pattern and did not execute. Checks were then run
directly, with final raw logs captured by Python subprocess before any summarization.
No secret/config file was read and no guard was disabled.

## Scope and reproducibility

Cargo resolved only one lockfile change: adding the already-resolved sha2 dependency edge
to memra-kv. Root Cargo.lock is lead-owned, so this auto-generated change was saved as
`Cargo.lock.patch` and the local root file returned to the exact base bytes. Apply this
fragment with the source at integration. An unlocked `cargo check/test --offline` reproduces
the edge automatically; `--locked` is NOT claimed on the branch until lead applies it.

No HostPrefixCache/server/runtime callsites changed, no env reads or CUDA kernels added.
No stage in this receipt proves G1/GPU readiness or active KV materialization. Shared
contracts, global governor injection and all CELLS.md GPU rows remain pending.

Worktree intentionally remains open for the requested multi-round engagement. No push,
merge, PR, tag, deployment or live verification. No scratch directories outside worktree
were created. About 0.25 agent-hours for this round; the 10 agent-day WP budget remains
almost entirely for frozen-contract integration and real hardware qualification.
