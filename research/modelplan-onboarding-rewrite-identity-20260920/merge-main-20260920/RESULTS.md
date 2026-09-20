# Main integration validation

The only textual conflicts were additive documentation in docs/TESTING.md and
research/INDEX.md. Both lanes were retained. No #542 runtime file conflicted. The
source-comparison record confirms all Rust files changed by #542 match the exact
62113383 source used for the native gate.

Upstream adds engine module exports for banked_residency and ple_rows_tier, a gated
Qwen4Exp PLE path, and a kv-tier gate. Those incoming paths were retained unchanged.
The affected compiler/CLI/tier/KV CPU suites pass; the reference suite retains only the
previously reproduced macOS #548 mismatch. All61 tier Python tests, contract fixture
pins, formatting, flags/docs/board checks, and engine/server Linux cross-target
clippy/test typechecks pass. The latter uses DOCS_RS placeholders and is not GPU evidence.

The native receipt remains historical at62113383 and its exact archived ELF hashes.
No claim transfers to a newly compiled merged executable. The PR remains draft, with
whole-model/serving/MTP/pipeline/performance gates outside the scoped native result.

Whitespace validation is scoped to the PR diff against current origin/main. Incoming
upstream raw logs contain pre-existing whitespace; their original bytes were preserved
instead of rewriting another lane's evidence.
