# OpenAI opt-in streaming usage (#529)

Implementation revision: `4f635204e15015ca498edef8231c341c9c4087bf` in `avifenesh/memra`.
This is API serialization qualification, not a numerical-kernel or performance result.

## Contract decision

The issue's original premise that streams carried no usage was incorrect: both
OpenAI dialects already put usage on their choices-bearing finish chunk. Preserve
that byte shape when `stream_options.include_usage` is absent or false. True opts
into null usage on every content/finish chunk and one additional empty-choices
chunk containing the full usage object before `[DONE]`. Errors do not fabricate a
successful terminal usage chunk. No new environment variables or runtime numeric
programs were introduced.

## CPU/library qualification on CUDA-capable development hardware

Hardware: one rented RTX 5090, CUDA 13.1. All commands ran in an isolated checkout
and target directory. Raw output is retained in `cuda/`.

| Command | Result |
| --- | --- |
| `cargo test --release -p memra-server --lib` | 714 passed, 0 failed, 5 ignored |
| `cargo clippy --release -p memra-server --all-targets -- -D warnings` | Passed |
| `cargo build --release -p memra-server --bin memra-server` | Passed |
| `cargo fmt --all -- --check` (macOS) | Passed after formatting |
| `git diff --check` (macOS) | Passed |

The three new tests cover strict opt-in usage, legacy byte compatibility, and
terminal failures for both OpenAI dialects. The strict test compares the entire
usage object against the blocking collector on identical synthetic worker events
(including elapsed time, cached tokens and speculative acceptance), checks receipt
completion counts, and covers zero/nonzero completion lengths. It uses the chat
blocking collector as the usage twin for both streams: both non-stream OpenAI
builders call the same `usage_json` helper. Live checks exercise each actual route.

The five existing ignored tests were not run: the manual loopback proxy fixture,
CUDA host-arena fixture, two-device host-cache roundtrip, DFlash retained-state
fault matrix, and >=32 GiB-free mixed-load admission cell. This is not a
workspace-wide clippy result or a multi-GPU qualification.

## Local compile limitation

Initial `cargo test -p memra-server --lib` and fallback
`cargo check -p memra-server` both exited 101 on macOS before executing tests:
`memra-engine/build.rs:314`, `spawn nvcc: ... No such file or directory`.
The missing fallback executable was `/usr/local/cuda-13.1/bin/nvcc`.
Raw logs are in `cpu-macos/`; the initial failed formatting check is retained there
too. These are infrastructure failures, not passing local compilation evidence.

## Live gate protocol

`live_usage.py BASE_URL MODEL OUT_DIR` uses `curl -N` to retain raw SSE bodies and
checks both `/v1/chat/completions` and OpenAI-mode `/v1/completions`, each with true,
absent, and false options, for short uncached prompts and long cache-hit prompts.
It warms an identical prompt twice per route/regime, uses
`temperature: 0`, `seed: 42`, `max_tokens: 8`, and compares accounting against a
non-stream twin at the same warmed cache state. No concurrent request is issued. The short regime asserts zero cached tokens;
the long regime requires positive cache reuse before comparing all stream shapes.

Full live usage-object equality is not a valid assertion across separate requests:
`elapsed_s` measures their individual durations. The live gate compares every
other usage field and preserves all elapsed values. Exact equality including
`elapsed_s` is instead demonstrated by the synthetic-event test. Live metering
plugin/ledger rows are not qualified; receipt/count parity is covered by the CPU
mock-receipt test.

## Live result

PASS: opt-in and legacy SSE shapes, terminal failure CPU tests, and live cold/cached accounting parity on both OpenAI routes.

The live gate ran with Gemma-4-12B-IT QAT Q4_0 under
`flock /tmp/memra-5090.lock`, held across server startup, readiness, all requests,
and shutdown. Server configuration:

```sh
MEMRA_COMPAT=openai MEMRA_ADDR=127.0.0.1:18092 MEMRA_CTX=4096 \
MEMRA_SERVE_SPEC=0 \
MEMRA_MODELS="g=/data/models/gemma4/gemma-4-12b-it-qat-q4_0.gguf" \
/root/target-include-usage/release/memra-server
```

Gate command: `python3 live_usage.py http://127.0.0.1:18092 g OUT_DIR`.
`/readyz` returned ready before requests. Gate exit was 0; the server then drained
and exited (`SERVER_STOPPED`). This was an isolated development smoke server, not
a public deployment.

| Route | Regime | Prompt | Completion | Cached prompt |
| --- | --- | ---: | ---: | ---: |
| chat/completions | short | 25 | 8 | 0 |
| chat/completions | cache hit | 905 | 8 | 905 |
| completions | short | 13 | 8 | 0 |
| completions | cache hit | 893 | 8 | 893 |

Each row passed non-stream versus true/absent/false stream accounting parity:
12 SSE responses and 4 non-stream twins, plus 8 warmup requests. `[DONE]` was last
on every stream. The initial short-prompt-only pass is retained separately rather
than discarded. All raw response hashes in both manifests were checked after
copying the receipts.

- Full matrix: `cuda/iu-live/`, `cuda/iu-live.log`, `cuda/iu-server.log`.
- Initial short-prompt pass: `cuda/iu-live-initial-short/` and matching logs.
- Runtime binary SHA-256: `454671822a7aa3fb58fa79aeec4be865feff855689ae6e4496b3bcd6af8f470a`.
- GGUF SHA-256: `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b`.
- Response fingerprint: `memra-0.138.0-7e852c710439`.

No live speculative decode, live failure injection, metering-plugin ledger,
full cache-meter fanout gate, other model, or multi-GPU battery was run. Those are
not implied by these CPU tests and bounded live serialization checks.
