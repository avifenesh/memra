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
