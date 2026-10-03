# Atomic update alias maintenance

Replace twelve atomic `fetch_update` calls with `try_update`. Keep the writer census recognizing both spellings. The closure bodies, memory orderings, integer types, error adapters and counter owners are unchanged.

The pinned Rust implementation forwards `fetch_update` directly to `try_update`. [Pinned Rust source](https://raw.githubusercontent.com/rust-lang/rust/8bab26f4f68e0e26f0bb7960be334d5b520ea452/library/core/src/sync/atomic.rs).

## Evidence scope

The before source is `9595cea51efe1fb70b9f922acf5798a0584ce08d`. The tested after source is `89507cd5addc3ebd756d35f959842849731d7e37`, composed on `4ba0d0cb0a5773ebfd9367c69851963612a94a78`. Thirteen source and compiler/config inputs bridge the historical baseline to that base byte for byte.

The real package tests passed the same 530 unique CPU test identities in both arms: 411 Tier/KV and 119 server controls, with zero failures, ignored tests or skip markers. Their raw logs and identity/hash inventories are retained. These cover normal ownership, permits, queue cancellation, admission rollback, route underflow and contention.

The emitted-code check binds all twelve source sites to actual atomic operations. It retains CAS orderings, retry branches, integer widths, bounds, error branches, target identities and constants. Arc support code is not counted as a site witness.

Forty-two real emitted IR function pairs compare equal. The three complete linked test witnesses and their inspected direct/relocated code targets compare equal across 127 retained function rows. Every instruction offset, resolved target, referenced constant and six-entry jump-table destination stays bound to raw bytes or its relocation witness. Further callees retain identity/offset and raw hashes; this does not claim complete transitive-code equality.

The copied exact expressions pass 18 boundary/error/contention controls per arm. Four compiling overflow/underflow/queue/byte faults fail their unchanged oracle tests, then the restored 18 pass. Actual raw CAS-order and overflow-constant IR mutants are rejected by the comparator, while restored and comment-only controls compare equal.

The separate installed Rust 1.101.0-nightly consumer diagnostic emits twelve deprecated-call warnings before and zero after. The earlier real stable1.99 package verification reported eleven production-call warnings. The twelfth call is cfg-test-only. These are distinct observations; current hosted package verification still gates the merge.

Copied-expression controls exercise the exact twelve atomic expressions on local same-width counters, including checked exhaustion, underflow, reset, queue/byte bounds and contention. Six copied exhaustion adapters preserve the `map_err` or panic behavior. Their local `Error::Overflow` stand-in and receiver adapters are declared in the source map. They complement the real package tests and emitted site witnesses; they do not run whole production constructors.

## Diagnostics and exclusions

Rust 1.97.1 is the pinned code-generation and package-test toolchain. It reports no deprecated-call warnings in either arm. A separate installed consumer-toolchain diagnostic checks the twelve exact call expressions. It does not change a compiler pin or build default. Current hosted package verification is required before merge.

The packaged server's explicit DEGRADED build identity remains named in issue [979](https://github.com/avifenesh/memra/issues/979). It is not suppressed or repaired by this alias change.

No CUDA source, native math, compiler/build default, model artifact/default, numerical tolerance or required native gate changes. The CPU evidence establishes this scoped alias maintenance result. It grants no model, runtime, serving or release qualification and authorizes no deployment. There are no GPU performance measurements or performance claims.

The first linked comparison exceeded its 150-function transitive bound. The next attempt exposed duplicate local ELF-symbol selection, BSS handling and an over-wide data read. Failed outputs remain in the private lane record. The corrected comparison names its direct-callee limit; it does not claim equality of the entire transitive executable.

publicity: skipped: maintenance release

Raw `.log` output is carried in reversible `.log.raw.json` envelopes. `RAW-LOG-TRANSPORT.json` binds original names, original byte hashes and public envelopes. No raw output bytes were trimmed or changed.
