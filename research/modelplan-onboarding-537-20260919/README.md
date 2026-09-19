# Loader semantic rejection (#537)

The canonical compiler now rejects declared window, RoPE, or text activation semantics that
its typed program does not implement. Dense, hybrid and external MTP load selection require
an accepting pack before reading model weights. Pack refusal cannot fall through to the
generic compiler.

Normalization retains `hidden_act` / `hidden_activation`, scalar HF and GGUF RoPE types, and
Gemma-style per-attention RoPE types. Source preflight requires a tensor or normalized factors
when Llama3 scaling is declared, including when automatic placement is disabled. Step HF load
now consumes its already-derived frequency factors instead of silently using no factors.
The Step pack validates auxiliary dtype, extent, byte length and finite positive values
before allocation; the engine uploads the same validated buffer, without reading it again.
Existing Step window/checkpoint factors, Gemma
window/proportional RoPE/GELU-tanh, and Qwen4Exp YaRN keep their typed programs. OLMoE and
MiniMax-M3 receive explicit packs for their existing programs so removing the fallback does
not exclude them. Both packs remain at `support: None`; no checkpoint or serving qualification
is inferred from registration or synthetic fixtures.

## Source and scope

- Base: `b3487a03b0ee3f833c1157e7b7d68f2cb35a3843`.
- Branch: `codex/537-model-semantics`.
- Host: macOS arm64, Rust 1.97.1. Exact toolchain, source hashes and CPU test executable hash:
  [source-manifest.json](source-manifest.json).
- No kernels, hardware defaults, flags, or published performance numbers change. This is a
  host config/plan admission change plus faithful consumption of normalized Step HF RoPE factors.

## Verification

| Command | Result | Raw evidence |
|---|---|---|
| `cargo test -p memra-gguf --test model_semantics` on base plus the original three regressions | 0 passed, 3 failed; window, llama3 RoPE and GELU each compiled as full attention/no factors/SiLU | [before.log](raw/before.log) |
| `cargo test -p memra-gguf --lib model_plan::semantics_tests` | 12 passed | [after.log](raw/after.log) |
| `cargo test -p memra-gguf -p memra-runtime --lib` | 289 gguf passed, 1 ignored; 1 runtime passed | [gguf-runtime.log](raw/gguf-runtime.log) |
| `cargo test -p memra-gguf -p memra-reference -p memra-runtime` | Initial gguf suite passed; reference 65 passed, 1 failed | [cpu-suites.log](raw/cpu-suites.log) |
| Base: `cargo test -p memra-reference --lib qwen35_fixture_executes_mixed_gdn_and_full_attention_state` | Same bitwise fixture failure and identical output bits on untouched base; tracked separately in #548 | [reference-base.log](raw/reference-base.log) |
| `cargo clippy -p memra-gguf --lib --tests` | Passed; existing macOS unused `AsRawFd` import warning in source.rs | [clippy.log](raw/clippy.log) |
| `cargo fmt --all -- --check`, `git diff --check` | Passed | [fmt.log](raw/fmt.log) |
| `cargo test -p memra-engine --lib` | Could not build: `spawn nvcc: ... NotFound` | [engine-build.log](raw/engine-build.log) |
| `DOCS_RS=1 cargo check -p memra-engine -p memra-server --tests` | Diagnostic type-check attempt also blocked: Linux-only libc APIs and missing MMQ archive hash in documentation build | [host-type-check.log](raw/host-type-check.log) |

The initial three regressions were first run as an integration test, then moved into
`model_plan::semantics_tests` so the existing unfiltered `--lib` CI job always runs them.
Later cases cover the shared load entry point, nested text activation precedence, supported
dense plan equality, legacy packs, hybrid configs, GGUF RoPE metadata and per-attention Gemma
RoPE. Step GGUF tests cover trunk/MTP geometry and required factor presence, including
normalized HF factors; the source test refuses any attempted model-weight read during
preflight. The MiniMax wrapper test pins dense/routed/shared placement, routing, norms and
partial RoPE. Existing Step HF, Qwen4Exp YaRN and DeepSeek-V4 fixtures additionally assert
load-plan acceptance and their exact RoPE operations. Existing pack/compiler, tensor-contract and source fixtures
remain in the full gguf suite.

## Outstanding native qualification

No designated non-serving GPU rig was available to this task. No CPU test, documentation
build, or hosted compile check is GPU qualification. Keep the PR draft and the issue open until
the required native gates have current receipts: engine/server suites and the designated
non-serving RTX PRO 6000 pair battery (`kernel-check`, affected-model `run-gen` argmax, and
`run-spec` K=1..8), including Step HF Llama3 factor consumption, under the existing
`/tmp/memra-gpu.lock` ownership protocol. No merge, tag,
deployment, or support-state promotion is authorized by this record.
