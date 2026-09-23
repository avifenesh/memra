# Canonical Gemma Eager declaration

Source `24668af517c83f5bbee64de0a0f0a16c1f4dec74` passed the selected 27-case
caller campaign, baseline and corrected transfer gates, and the v3 generic battery.
Its separate real Gemma run stopped at `EAGER_INELIGIBLE`, before numerical capture:
the generic Eager table excludes SlidingWindowAttention, SlidingKvState,
GeluTanhActivation and GemmaResidual. That prerequisite remains FAILED and HPOST
math remains UNRUN; no receipt or model qualification was issued.

The selected artifact is the official text GGUF
`google/gemma-4-12B-it-qat-q4_0-gguf@29d097773436b69ff9feafd636ab4cf873786537`,
`gemma-4-12b-it-qat-q4_0.gguf`, SHA256
`93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b`.
CPU inspection gives a non-PLE `gemma4_dense` plan with 667 tensors, hidden width
3840 and vocabulary 262144. The model format is unchanged.

The declaration adds a separate `gemma_eager` operation-registry column and
`native-gemma-eager` implementation for the existing public `decode_step_h` Gemma
walk. Selection uses the same canonical residual-program classifier as that entry.
The generic Eager and fresh-KV table stays unchanged. PLE, shared-KV source packs
and parallel-MoE programs remain outside this increment. Prime, graph, pipeline and
speculative eligibility/admission are unchanged.

CPU controls pin the exact twelve-operation set, retain every frozen legacy table,
reject generic SWA through this declaration, reject PLE/MoE variants, and verify
that a matching Eager receipt grants no other surface. Missing and rehashed
wrong-implementation receipts refuse. The generated registry document is checked
against the authoritative renderer.

This is a source/CPU candidate until independent review and fresh native proof.
The real artifact must first pass the existing independent Eager parity capture,
then HPOST off/on runs must retain raw T1 and independently normalized expected
rows, actual prime exports and the actual pooling consumer. Existing tolerances,
the original Qwen T16 oracle and all historical failures remain unchanged.
