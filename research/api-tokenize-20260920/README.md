# `/v1/tokenize` and `/v1/detokenize` (#531)

Implementation tip at qualification: `ae3562fb6` on the rented box, `7b7331b2b` (the same five
commits rebased onto main `fdb781362`) on the local rig. This is API and accounting
qualification, not a numerical-kernel or performance result.

## Contract

Token inspection is CPU-only and shares chat completions' bearer auth, 192 MiB body ceiling
and body-admission pools. The messages form renders through chat's semantic builder and the
same render-to-ids function chat's HTTP-side prompt accounting uses, so `count` equals the
`prompt_tokens` a chat completion bills for the same messages. Raw prompts add no BOS so
detokenize round-trips. OOV ids and unknown models reuse the existing named 400s. The HTTP
tokenizer map now loads on every boot from the worker's own source resolver and vocabulary
verifier instead of only under prepaid enforcement. Schema: `docs/API-SURFACES.md`.

## Rented RTX 5090 (CUDA 13.1), tip ae3562fb6: `cuda-box-5090/`

| Command | Result | Log |
| --- | --- | --- |
| `cargo test --release -p memra-server --lib` | 718 passed, 0 failed, 5 ignored | `tok.log` |
| `cargo clippy --release --all-targets -- -D warnings` | passed | `tok.log` |
| `cargo build --release -p memra-server` | passed | `tok.log` |
| live check, gemma-4-12b-it-qat-q4_0, port 18090, rig lock held | `LIVE_CHECK_PASS` | `tok.log`, `tok-server.log` |

`tok-attempt1.log` and `tok-attempt2.log` are the two earlier runs on prior commits of this
branch; each failed `tokenize_messages_count_equals_chat_prompt_accounting` once (716/1 and
717/1) before the omitted-tools and BOS fixes landed. They are kept as raw evidence, not as
passing runs.

## Local RTX 5090 Laptop (CUDA 13.1), rebased tip 7b7331b2b: `cuda-local-5090/`

| Command | Result | Log |
| --- | --- | --- |
| `cargo test --release -p memra-server --lib` | 718 passed, 0 failed, 5 ignored | `build-test-clippy.log` |
| `cargo clippy --release --all-targets -- -D warnings` | passed | `build-test-clippy.log` |
| `cargo build --release -p memra-server` | passed | `build-test-clippy.log` |
| live check, same model, same port, rig lock held | `LIVE_CHECK_PASS` | `tok-live.log`, `tok-server.log` |

Both live runs return identical ids: the messages form counts 15 tokens and the chat
completion bills `prompt_tokens: 15`; `"Hello, world!"` tokenizes to four ids and round-trips
exactly; id `4294967295` returns the named `prompt_ids` 400 with the supplied `x-request-id`
echoed. `MEMRA_SERVE_SPEC=0`, `MEMRA_CTX=4096`, `MEMRA_COMPAT=openai`.

## Live gate protocol

`tok-live.py` boots the lane's `memra-server` binary on the 12B GGUF, waits for `/readyz`,
then issues: `/v1/tokenize` (messages) and a one-token `/v1/chat/completions` on the same
messages (counts must match), `/v1/tokenize` (raw prompt) followed by `/v1/detokenize` on
the returned ids (text must match), and `/v1/detokenize` on an OOV id (400, `param`
`prompt_ids`, request id echoed). The two copies differ only in binary, model and log paths.

## Limits

One model family, one card class. The five ignored server tests were not run. Not a
workspace-wide clippy result and not a multi-GPU or performance qualification.
