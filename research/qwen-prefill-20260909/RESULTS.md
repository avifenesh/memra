# Exact Qwen prefill attention staging

Three-plane KV staging reduces 131070-token cold TTFT from 68.295 to 63.450 seconds, a 7.095% reduction, on one RTX 5090 desktop. The prime chunk remains 1024. MEMRA_PRIME_KV_T3 stays default OFF; this is an exact kernel improvement, not a new dedicated-serving verdict.

## Cold HTTP results

Three interleaved fresh boots per arm at each size, one cold request per boot, 128 maximum output tokens, vendor-default sampling with no sampling fields. Input counts and zero cached tokens are checked. Measurements are medians, with individual rows in measurements.json and requests.csv.

| Prompt tokens | OFF TTFT s | ON TTFT s | Reduction | OFF / ON input tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 2.524213 | 2.507496 | 0.662% | 3245.37 / 3267.00 |
| 32768 | 11.341811 | 11.018565 | 2.850% | 2889.13 / 2973.89 |
| 131070 | 68.295111 | 63.449565 | 7.095% | 1919.17 / 2065.74 |

The full fresh-boot battery has 24 distinct boot nonces and 96 bounded requests: 18 cold probes, 30 unseeded decode probes and 48 cache-chain turns. All return valid final usage; no OOM, request error, retry or detected repetition loop occurs in this battery. Output caps are measurement bounds, not claims that every answer finished naturally.

## Mechanism

The original kernel stages two K and two V tiles plus probabilities in shared memory. The new kernel keeps two K tiles and one V tile, and forms the probability MMA operands directly from the existing C-fragment values using the same scalar RN BF16 conversions. QK, online softmax, PV and normalization keep their original per-row operations.

Shared memory falls from 69888 to 49152 bytes. Driver resource queries report 255 registers/thread and 8 local bytes in both arms; the static maximum active blocks/SM increases from one to two. The server trace confirms the new kernel is selected. There is no new global allocation, chunk-size change, precision change or decode dispatch.

The complete baseline 131k trace attributes 30.848 s to attention and 25.798 s to trunk GEMMs. GDN already uses chunkwise MMA; NVFP4 trunk GEMMs already use W4A8 INT8 tensor cores. Allocation tracing also hashes tap data, so its host-idle time is not a normal-serving launch-cost claim.

## Correctness and decode

- Twelve typed GPU cells are byte-identical to the existing quantized-attention reference, covering causal/noncausal cases, uneven tails, multiple KV heads and 131070 depth.
- Real 32768/131070-token primes match pinned main in taps, draft features, positions, final logits, boundary logits and complete boundary state.
- K=1..8 matches plain on the 8192-token chat prompt. The canonical margin gate has flips=0, bad=0 with a zero-flip budget.
- Kernel-check: 92 green cells, 22 skipped cells retained in the log. Memcheck and synccheck: zero errors.
- Four greedy HTTP restore turns match exactly; warm cached counts are 8160, 8192 and 8224 in both arms.

Unseeded c=1 decode, five prompts across three boots/arm: pooled 161.69 to 167.53 tok/s, individual-request median 169.68 to 164.55. Those aggregates move in opposite directions because sampled outputs differ; no decode speedup is claimed. An additional fixed-seed sampled control matches all 15 output pairs, with median paired throughput ratio 0.999911. Its absolute medians are 153.627/153.696 tok/s. Decode is unchanged within that control's measured noise.

The eight-turn cache twin starts at 8192 tokens, with three boots/arm. All 21 warm turns per arm restore at least 8160 tokens. Median warm TTFT is 0.186667/0.186187 s. This is not a 131k sustained-cache or four-session qualification.

## Failed controls and default decision

The removed eight-warp experiment was exact but flat at 1024 rows. Its 4096-row kernel improved, but a sequence of 8k, 32k and 131k requests in one boot failed during decode after long-prefix publication. The same sequence also failed in one chunk-1024 T3 boot. Both failures remain in the private companion receipts and are excluded from successful timing rows. No cause is assigned to the kernel from these unpaired sampled failures.

Default: OFF, decide-by 2026-09-23. Proposal: enable T3 for the next batched Qwen/5090 configuration after the intended retained-state workload clears the documented failure. The fresh cold gains justify that proposal; they do not justify a blanket serving or concurrency claim. No release, fleet change or deployment is performed by this lane.

Binary SHA256: 1422ac35e20c1225e3ad923ca96e0879f329a0f0b8320b5a9a8ac380a79152f0. Runtime source at measurement: da9fe9b1e; later base updates are confined to DSV4 runtime/tests and registry/CI documentation; the Qwen runtime files are unchanged. Gate logs are in gates/. Operational profiles, raw Nsight reports, complete requests and the SGLang comparison are in the private companion report. Hosted CI and the adversarial self-review are attached to PR #403. Revuto could not run because its AGY provider hit quota; that is not represented as a passed review.
