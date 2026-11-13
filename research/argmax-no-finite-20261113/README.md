# A logits row with no finite value has no token (2026-11-13)

Verdict: fixed. A logits row whose every value is NaN or -inf has no argmax. The device argmax wrote
0x7fffffff for it, and the CUDA-graph decode chain fed that id straight into the device embed
gather: a read 2^31 rows past the table. The host argmax and the host sampler returned token 0, a
real token nothing chose. Now the id is named (`TOKEN_NONE`), every kernel that indexes memory by
a device token id bounds it, and the host refuses the id where it reads it. One request fails
with "argmax found no finite logit"; its batch peers go on.

## What changed

Kernels (in-range ids compute exactly what they did):

| kernel | an id past its table |
|---|---|
| `embed_gather_u32`, `embed_gather_u32_t` | NaN row (new `n_rows` bound, from the table length) |
| `gather_row_bf16_f32` (dspark markov) | NaN row (new `n_rows` bound) |
| `u32_map_k` (FR-Spec trim map) | stays `TOKEN_NONE` (new `map_len` bound) |
| `prob_of_token_partial_f32` (draft p-min) | p = 0, so a p-min gate cuts the draft there |
| `softmax_gather_f32`, `softmax_gather_filtered_f32` (sampled spec) | p = 0, so the walk rejects that draft |
| `residual_sample*_f32` fallback | a row with no value above -3.4e38 gives `TOKEN_NONE`, not 0 |

The argmax kernels are unchanged: they already wrote 0x7fffffff for such a row.

Host: `forward::argmax` and `memra_sampling` (greedy, greedy with penalties, sampled) return
`TOKEN_NONE` for a row with no candidate. `check_token` / `check_tokens` refuse an id at or past
`n_vocab`. They run where the host reads a device id: the decode and graph loops, the chain
decode, the spec accept walks (gemma, dflash/dspark, glm5, eagle, the MTP boundary), run-gen, and
the server's per-request token acceptance. An accept walk checks only `vam[..=m]`, the values it
read: a NaN draft row poisons only the verify rows after it, and the walk never reads those. So in
the gemma, dflash/dspark, glm5 and eagle walks a broken draft is a rejected draft (eagle proposes
id 0 for a draft row with no finite value; the verify decides), not a failed request. The MTP draft
chains keep their existing traps (`guard_vocab_token`, memra#87), which fail the request on a draft
sentinel. `spec::guard_vocab_token` now calls `check_token`. The diagnostic bins (`run-hybrid`,
`run-safetensors`, `run-dense`) print NaN for such a row instead of indexing with the id.

## Evidence

`gpu-tests-round1/` (RTX 5090 Laptop GPU, driver 615.71.09, AC power, one gpulock lease): the test file
`crates/memra-engine/tests/argmax_no_finite_gpu.rs`, one test per kernel, run on origin/main
7a43ac1e56 (an old-API copy of the test: the same id as a literal, no `check_token`) and on the
first-round fix (p(tok) was NaN then; the review moved it to 0).

| test | origin/main | fix |
|---|---|---|
| host argmax of an all-NaN row | FAIL: 0, not `TOKEN_NONE` | pass |
| device argmax vs host argmax (ties, NaN, -FLT_MAX, +inf rows) | pass (kernel unchanged) | pass |
| embed gather, single id | FAIL: id 1000 of a 1000-row table read past the table, no fault, not a NaN row | pass |
| embed gather, T rows | FAIL: `CUDA_ERROR_ILLEGAL_ADDRESS` | pass |
| p(tok) for an id past the row | FAIL: `CUDA_ERROR_ILLEGAL_ADDRESS` | pass |
| trim map | FAIL: `CUDA_ERROR_ILLEGAL_ADDRESS` | pass |
| bf16 row gather | FAIL: `CUDA_ERROR_ILLEGAL_ADDRESS` | pass |

`compute-sanitizer --tool memcheck` over the fix: 0 invalid accesses. Its 6 reported errors are
`CUDA_ERROR_NOT_FOUND` from `Engine::func` trying the embed kernels in the modules before
`qmatvec` (API results, not memory errors).

`gpu-tests-round2/` (same card, one gpulock lease, the final code after the review): the 9 tests
pass, including the two the review added (the sampled-spec gathers give p = 0 for an id past the
row; the residual fallback gives `TOKEN_NONE`), and `compute-sanitizer --tool memcheck
--report-api-errors no` reports 0 errors. `cpu-old-sampler/`: the new memra-sampling test on
origin/main 7a43ac1e56 fails (an all-NaN row samples token 0).

`gpu-tests-old-round2/`: the two round-2 tests on origin/main fail too (the sampled-spec gather of
`TOKEN_NONE` faults with `CUDA_ERROR_ILLEGAL_ADDRESS`; the residual fallback returns token 0).

The sm_120a and sm_121a push-gate records are in the evidence commit that follows.

CPU: `cargo clippy` (engine, server, sampling: lib, bins, tests) clean; memra-engine lib 743 passed;
memra-server, memra-kv, memra-tier 1643 passed; memra-sampling 35 passed. The sampler's frozen
comparison oracle (`nucleus_order_tests.rs`) pinned the old answer for an all -inf row (the last
candidate, token 4095); it now follows the contract.

## Not covered

A row with some NaN values and some finite ones still picks the largest finite value (argmax) or
draws from a NaN softmax (sampled). That is a different failure (a partly broken forward) and is
not changed here.
