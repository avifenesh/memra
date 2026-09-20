# `stop_token_ids` on chat and completions (#532)

Implementation tip at qualification: `74abaf40a` (the lane's five commits rebased onto main
`fdb781362`) on the local rig. API and accounting qualification, not a numerical or performance
result.

## Contract

`stop_token_ids: [u32]`, default empty, at most 16, vocabulary-validated at intake through the
`prompt_ids` gate (named 400, `param: "stop_token_ids"`). The first raw sampled or accepted id
that matches ends generation with `finish_reason: "stop"`; that id and the rest of its
speculative burst are neither emitted nor counted in `usage.completion_tokens`. Explicit stops
win over EOS; an unlisted EOS keeps its ordinary accounting. Schema: `docs/API-SURFACES.md`.

## Local RTX 5090 Laptop (CUDA 13.1), tip 74abaf40a: `cuda-local-5090/`

| Command | Result | Log |
| --- | --- | --- |
| `cargo test --release -p memra-server --lib` | 716 passed, 0 failed, 5 ignored | `build-test-clippy.log` |
| `cargo clippy --release --all-targets -- -D warnings` | passed | `build-test-clippy.log` |
| `cargo build --release -p memra-server` | passed | `build-test-clippy.log` |
| `serve_check.py`, gemma-4-12b-it-qat-q4_0, port 18091, rig lock held | `PASS` | `serve-check/result.json`, `serve-check.log` |

Gate protocol (`serve_check.py`): boot the lane binary in native mode, take the greedy token
tape of one prompt (`temperature 0`, `seed 42`, `max_tokens 64`, `MEMRA_SERVE_SPEC=0`), pick
the first interior id that has not occurred earlier in the tape, and record the k-token prefix
run. Reboot in OpenAI mode and assert: the baseline text matches native; `stop_token_ids=[tape[k]]`
returns exactly the prefix text with `finish_reason: "stop"` and `completion_tokens == k`; a
stop on `tape[0]` returns empty text with zero completion tokens; the streamed form yields the
same prefix, a stop finish chunk and the same usage; when the baseline ends in EOS, listing the
EOS id explicitly returns the same text with `finish_reason: "stop"` and one fewer completion
token (the unlisted EOS bills an empty-text token; the listed one is excluded).

Result: baseline `<|channel>thought\n<channel|>1, 2, 3, 4, 5, 6, 7, 8, 9, 10`, 34 tokens ending
in EOS; k = 2, stop id 107 (`\n`); stopped text `<|channel>thought`, 2 completion tokens; first-id
stop empty with 0; stream prefix identical; explicit EOS 33 tokens. Each response is retained
with its sha256 in `result.json`.

### First run, set aside: `serve-check-gemma3-markup/`

The gate first used gemma-3 turn markers (`<start_of_turn>`, `<end_of_turn>`) plus a literal
`<bos>`. In the gemma-4 vocabulary those marker names are plain text (ids 105 and 106 are
`<|turn>` and `<turn|>`), and the completions route adds BOS itself, so the model saw broken
markup and a double BOS and produced `<|channel>_0_0_0...`. The gate's id-level assertions still
passed on that tape (k = 2, stop id 236771), which is why the run is kept, but it is not the
receipt: the prompt was wrong, not the engine. Verified separately with `prompt_ids` equal to the
chat template's ids, which produce `1, 2, 3, 4, 5, 6, 7, 8, 9, 10` on the same route.

## Rented RTX 5090, 202cec78 (one commit before the tip)

716 server tests, clippy clean, release build. Its serving gate did not run because the minimal
CUDA image has no `ss`; the tip commit removes that dependency. Raw logs stayed on the box.

## Limits

One model family, one card class, speculative arms off (`MEMRA_SERVE_SPEC=0`; the spec-burst
behaviour is covered by the pure-function partition tests, no draft head is staged for the 12B on
this rig). Not a performance claim.
