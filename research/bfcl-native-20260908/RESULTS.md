# Native validation results

- Source-FP32 checkpoint parity: max_abs 0.00018501282, max_rel 0.0000083913355,
  argmax 4 = 4. The oracle explicitly dequantizes the original FP8 block scales before
  FP32 execution; it is a source-weight oracle, not an FP8 production-runtime claim.
- memra-gguf: 241 passed, 1 ignored. Includes exact nonuniform-code/scale QKV slices,
  rejection of malformed/unaligned slices and explicit activation-format parsing.
- GPU kernel-check: 93 cells passed, 22 skipped. Windowed hd128 and hd256 cover the
  CPU and f32 reference bands, masking, single/double-buffer bit identity and zero-window
  identity. Dynamic FP8 dispatch at m=1/2/5/9 matches its native primitive bitwise;
  the legacy q8_1 activation path differs and is the negative control.
- The real long-prompt path previously panicked at the hd128-only wrapper with hd256.
  The new path completes. A prefill-only diagnostic at 49,666 tokens produced
  3.3862 / 3.1968 / 3.1991 seconds. These are diagnostic measurements, not release
  performance claims or full end-to-end sampled throughput.

## Open numerical qualification

Strict 0.05 optimized-logit comparison still fails. Against the repaired offline FP8
runtime oracle, max_abs is approximately 1.97 (eager/batch) and 2.33 (prime), while the
18-token oracle argmax matches. No tolerance was relaxed to manufacture a pass.
The current GPU program retains the engine's q8_0 K / q5_1 V cache and f32 intermediate
storage; this is not byte identity with a vendor BF16-KV runtime.

The offline FP8 runtime control needed two explicit fixes to its generic Transformers
loading path: initialize the tied output head from the embedding, and retain the config's
BF16 activation container before FP8Linear quantization rather than casting activations
to the physical FP8 weight dtype. Those repairs are not an unchanged-vendor-runtime claim.

Sampled tool calls can be valid, but one sampled control emitted a complete call inside
an unterminated thinking block; the chat API correctly did not expose it as a tool call.
A greedy offline FP8 control closed the thinking block and emitted the intended call.
This difference is retained for model-quality analysis; the parser is not altered to
force a passing result. NativeReference is not production admission.

## Resident prime slab admission candidate

The admission estimate charged all prefill workspace on warm requests even when the
model retained reusable prime slab planes. Those allocations already reduced effective
free VRAM. The candidate subtracts only the requested prefix of physically allocated,
unborrowed slab planes from the workspace charge. Growth, speculative execution,
draft-backed execution, hyper and multi-device paths receive no credit. KV allocation,
transient workspace beyond the slabs and the reserve remain charged.

Two pure accounting tests pass remotely, including growth, malformed geometry and
overflow refusal. Remote format and server compile checks pass.

The pressure test passed three interleaved baseline/candidate windows on the non-serving
RTX 5090. Reserving 8,144 MiB emulated the laptop's 24,463 MiB capacity. Each window used
a cold pair followed by a warm pair of real 53,856-token requests, capped at 32 output
tokens with no sampling parameters (metadata defaults 0.6 / 0.95 / top_k20).
Baseline completed 3/12 requests and rejected 9/12 with HTTP429. Candidate completed
12/12 with HTTP200 and no CUDA OOM. Its reusable-slab credit was 2,281 MB: estimated
request cost fell from 7,868 to 5,587 MB while the 1,611 MB reserve stayed unchanged.
Both arms charged the same 74,240 bytes per context token. Cold candidate admission
initially charged the full workspace before any reusable allocation existed.

The candidate still serialized the pairs at this capacity. Its median cold/warm pair
walls were 39.03 / 39.55 seconds. These availability windows do not establish a general
throughput win; baseline's shorter failed-pair wall is not comparable useful work.
See `receipts/prime-credit-pressure.json` for every response and
`receipts/prime-credit-admission.log` for the accounting and refusal readback.

Candidate source: `1ea01297aa98d87a7195fba6425f78b16d302201`; server SHA256
`2ca6e5623bade2dfb63d304cd1907d6f854f0cc026b7923906a09bc214f6cc16`.
Baseline source: `8cfb182babe9fd64a708a10fce1e2c3f29200528`; server SHA256
`9c31aee15ad79f1447982cb5178fceab89b4e9d9cac70e71dc8a98ef2ca2bb4b`.
The isolated candidate target avoids stale dependencies observed when switching the
shared Cargo target between worktrees. Main-runner/spec and broader regression gates
remain pending. The active BFCL server resumed its original binary without this change.
