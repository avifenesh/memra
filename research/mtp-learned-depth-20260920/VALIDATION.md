# Validation receipts

Original local runtime source: `a37e30a6dd9637ece727c040f12a115dd6d96517`.
The published patch and blob manifest reconstruct it from a public base; see `SOURCE.md`.

- CUDA release build: `cargo build --release -p memra-engine --bin mtp-depth-study --bin gemma-depth-study`, `MEMRA_CUDA_ARCH=120a`, CUDA 13.1; `receipts/metadata/build-initial.log` and `build-cold-v2.log`.
- Four pure controller tests: historical `crates/memra-engine/src/learned_depth.rs` extracted from the runtime commit, `rustc --edition 2024 --test`; `receipts/metadata/controller-tests.log`.
- Three native Qwen history/template tests: `cargo test --release -p memra-engine --bin mtp-depth-study`; `receipts/metadata/qwen-history-tests.log`. These parser functions are unchanged by the subsequent cold-priming driver correction.
- Six receipt/schedule tests: `python3 -m unittest discover -s research/mtp-learned-depth-20260920 -p 'test_*.py' -v`; `receipts/metadata/runner-tests.log`.
- Plain-target greedy MTP oracle K=1..8: after unpacking, exact command and environment in `receipts-expanded/qwen-mtp-oracle-v1/identity.json`, output in `oracle.log`.
- Four-arm short/code-context greedy gates: `qwen-mtp-{short,code}-v2` and `gemma-mtp-{short,code}-attempt-1-v4`. Each directory retains the exact runner, `.command.json`, identity, token tapes and summary.
- Full selected-data audits: unpack first, then the two `analyze.py` invocations in `RESULTS.md`; outputs in `receipts/metadata/qwen-audit.log` and `gemma-audit.log`. Recomputed analysis JSON exactly matches the saved analysis files.
- Closing-tree checks: Rust format, flag census, performance-board freshness, whitespace, and public-boundary check. The publication diff contains no applied runtime change; the prototype is an archival patch.
- Exported source patch: `git apply --cached` reconstructed all eight runtime blobs exactly against public base `7326f0e176326bb9b445720068cc502ea132ffad` using an isolated temporary index.

The initial conditional-cache short gate and three GPU-interrupted partial timing
sets are retained but excluded. Selection is explicit in the final ledgers.

Captured model text, command logs and telemetry preserve their exact whitespace
through the scoped receipt attributes. They are evidence, not reformatted source.
Authored code and documentation keep normal whitespace checks. The publication stores the original captures as checksummed archives.

Publication also verifies all archive hashes, four safe-extraction tests, and exact
runtime-blob reconstruction from the public base. The combined ten-test output is
in `receipts/metadata/publication-tests.log`. The request-position diagnostics run
the full audit first and reconcile their segment totals with the overall metrics.

The full public-boundary unit suite passes 53 tests, including two new regressions
for pinned binary data skipped by the fast prefilter. Both tests failed on the old
implementation. Content and drift verification now use the full matcher for pinned
paths; policy patterns and raw archive bytes are unchanged. Output is recorded in
`receipts/metadata/boundary-regression-tests.log`.
