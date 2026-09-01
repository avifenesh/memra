# Invalid scored attempt — incomplete native token receipt

This run is retained as raw diagnostic evidence and contributes no scored matrix cell.

The first forced-spec request completed with HTTP 200 and reported `n_tokens: 2048`, but its
native `tokens` array contained only 803 ids. `request.py` emitted the following failure to the
live stderr stream:

> native response n_tokens does not match tokens array

That first harness revision did not route stderr through `tee`, so the quoted line itself is not
in `request-rep1.log`; the mismatch is preserved directly in `rep1/request-meta.json` as
`n_tokens: 2048` and `token_array_len: 803`, and the checked-in request source contains the exact
invariant message. The harness now tees stdout and stderr together before any scored restart.

The server log proves that this was a true speculative request (`MEMRA_SPEC_GATE=0` and repeated
`[spec-acc]` receipts, ending at cumulative acceptance `1245/2406=0.517`). Source inspection then
found that a speculative round can stream several committed tokens as one `Event::Token` text
delta carrying only the final id; the blocking native response had treated that event cadence as
the complete token-id receipt. The 6,316 response-text bytes were therefore paired with only 803
event ids. No first-corruption-token index can be computed honestly from this response.

The later line

> `[worker] spec pending flush failed (launch_pdl quantize_q8_1: CUDA_ERROR_DEINITIALIZED); dropping session`

occurred during owned-server shutdown after the HTTP response and is not labeled as a generation
failure. The GPU lock was released by the block's exit trap.

The receipt path is fixed before any restart, and scored work resumes under a fresh run id.
