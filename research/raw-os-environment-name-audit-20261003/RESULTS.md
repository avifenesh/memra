Boot auditing discarded every non-UTF8 OS key before classifying its name. A raw key beginning with an owned MEMRA family therefore bypassed the refusal, while an unowned raw MEMRA key bypassed the warning.

The audit now checks the exact ASCII namespace and generated owned-family prefixes on raw OS-name bytes. Owned malformed names refuse in default mode. Warn mode downgrades that refusal. Unowned malformed names join the existing deduplicated warning summary. Diagnostic rendering happens after classification, and values remain unread.

Valid UTF8 names keep their original retired-before-wildcard, legal-name, unknown-family and audit-switch behavior. Off mode and an absent registry remain unchanged. The real build generator emits the same 992 legal names and the identical registry SHA256.

The original and candidate executables compiled the actual audit source against that real registry. Each executed 30 process controls covering default, Warn and Off modes, malformed owned/unowned keys, unrelated keys, all 992 legal names with uninterpretable values, retired names and prefix/replacement-character controls. Six absent-registry process controls preserved the inert path. Nine actual Rust unit methods passed, including all five retained methods. The complete engine CPU suite passed 669 tests with zero failures and 80 ignored. The server CPU suite passed 1,093 tests with zero failures and 28 ignored. Strict release all-target Clippy passed with warnings denied. The framework passed 124 tests at floor 124, and the registry covers 14 packages.

The current coverage verifier admitted independently asserted CPU edges and refused missing-owned-refusal and skipped-mandatory-control copies. [PROOF.json](PROOF.json) retains both versions' raw process outputs, source/helper/executable hashes, registry identity, package counts, log hashes and mandatory-control messages. Candidate code: 7c67d11e0e5fa31fcd9d78b5755035aadb5fc4d4. Baseline: 462ae029f4efe9e7296d7dbad4e3afc4af026b0e.

Scope: CPU boot environment name admission. No model/server or GPU execution, native math, generated native program, compiler/default, artifact, qualification tolerance or required native-gate change. This does not qualify a native model or serving binary. GPU selection remains shadow-only.

publicity: skipped: maintenance release.
