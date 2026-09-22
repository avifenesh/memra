# CPU validation — compiler/access foundation

2026-09-20, Rust 1.97.1, macOS aarch64. No CUDA execution or model qualification is claimed.
`raw/sources.sha256.json` identifies the source files used by this increment.

| Check | Result | Raw record |
|---|---|---|
| `cargo test --locked -p memra-gguf --lib bound_source` | 12 passed, 0 failed, 0 ignored | `raw/bound-tests.log.gz` |
| `cargo test --locked -p memra-gguf -p memra-cli` | GGUF library: 307 completed, 2 explicit ignores; CLI: 11 passed; Step CLI contract: 1 passed; GGUF-inspect: 7 passed | `raw/compiler-suite.log.gz` |
| GGUF library skip census, budget 12 | 307 completed, 12 artifact-dependent early returns explicitly reported, 0 failures | `raw/gguf-skip-census.log` |
| Clippy, GGUF + CLI, all targets, `-D warnings` | passed | `raw/clippy.log` |
| Linux target type check, engine library/bins/tests + server | passed using `DOCS_RS=1` in a separate target directory; documentation CUDA stubs, not a native build | `raw/linux-typecheck.log` |
| Broader CUDA-free suite + skip census | failed at the existing Qwen3.5 reference bit-equality test; 65 other reference tests passed | `raw/cpu-skip-census.log` |
| Exact failing reference test on clean base `dc598deb4` | same failure and identical left/right values; isolated control checkout/build removed afterward | `raw/base-reference.log`, `raw/reference-control.json` |
| `cargo fmt --all -- --check`, `git diff --check` | passed | no output on success |

Commands for the census and cross-target checks:

```sh
MEMRA_GGUF_SKIP_BUDGET=12 python3 tools/skip-census.py run \
  --budget-var MEMRA_GGUF_SKIP_BUDGET --min-passed 280 \
  -- cargo test --locked -p memra-gguf --lib

MEMRA_GGUF_SKIP_BUDGET=12 python3 tools/skip-census.py run \
  --budget-var MEMRA_GGUF_SKIP_BUDGET --min-passed 400 \
  -- cargo test --locked -p memra-gguf -p memra-reference \
  -p memra-tokenizer -p memra-validate -p memra-sampling --lib

DOCS_RS=1 cargo check --locked --target x86_64-unknown-linux-gnu \
  --target-dir target/541-linux-typecheck \
  -p memra-engine --lib --bins --tests -p memra-server
```

The standalone GGUF census uses a floor for that single crate; the existing combined CI floor
and skip budget are unchanged. The broad suite is not reported as passing. The library summary
includes the artifact-dependent test functions that returned early; the census states them
separately rather than treating them as exercised artifact gates.

The reference failure is
`tests::qwen35_fixture_executes_mixed_gdn_and_full_attention_state`, at
`crates/memra-reference/src/lib.rs:8147`. `reference-control.json` compares its raw differing
values to the unchanged base. No tolerance or reference code was modified.

See [remaining implementation](README.md#remaining-implementation-before-541-acceptance).

The two gzip logs preserve the complete original bytes, including their terminal blank lines.
