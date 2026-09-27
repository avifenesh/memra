# Standalone external draft binding

This candidate follows frozen root integration `3a0f1ed71c804410498c46a2420617fbb380ede6`.
`MtpHead::load_draft` now prepares a separately typed `PreparedExternalDraftSource` before
allocating draft tensors. The compiler accepts the current standalone GGUF NextN surface;
it does not reuse the complete-model contract for a sparse draft or flatten a composite source.

The contract validates every declared NextN block and every present trunk copy. Only depth
zero is selected, preserving the external loader's existing single-head behavior. Private
block heads take precedence over file heads; duplicate private aliases refuse. Unselected
heads and embeddings remain bound to the opened artifact identity and are unavailable to
execution. A draft runtime carries private draft-only authority and cannot enter a text model
root. Existing ordinary/composite root identities and placement code remain unchanged.

Student geometry is distinct from the target interface: concat input and norms remain at the
target width, the inner full-attention/dense block uses the declared narrower width, and the
output-up projection restores the carrier width. The compiler checks the independent matrix
shapes and head geometry, including attached output-up scales. Unsupported student programs,
residuals and mixers refuse. Ordinary attention checks the gate, head geometry, RoPE and norm
parameters consumed from the target configuration; Step keeps its own per-layer geometry and
the existing engine program/scratch checks. No numeric kernel or fallback branch changes.

Trimmed heads bind their selected physical rows to an ordered I32/I64 `d2t` map. The map must
have exactly one dimension, match the selected head, contain unique in-range target token IDs,
and reject negative or oversized integers before narrowing. Integer GGUF rows now have an
explicit integer census classification, so I8/I16/I32/I64 weights cannot pass as quantized
weights. No shared JSON decoder, dependency or config implementation changes.

The opened draft identity is retained by `MtpHead::external_source_identity()`. It covers all
opened bytes, the draft binding/selection, the map values and target config/program context.
It is not a composed target-weight/draft rewrite identity. The existing composite-identity
refusal is unchanged; this candidate does not enable rewrite admission or native support.

## Evidence

Twelve focused CPU tests pass, using real micro GGUF files with independently specified tensor
shapes. They cover private/file-head ownership, three declared draft blocks, student inner and
outer widths/scales, a scalar projection-chain control, original encoded Q8_0 head bytes and
macro scale, Step per-layer SWA/gate geometry, I32/I64 map order, full opened identity after
pathname replacement, identity sensitivity to unselected bytes and map order, target mismatch,
malformed census refusal before payload reads, bad map shape/type/range/duplicates, and integer
weight refusal. No ignored tests or checkpoint-dependent early returns are in this focused run.
These are micro-artifact CPU controls, not a checkpoint or complete draft-inference qualification.

Strict GGUF Clippy and Linux DOCS_RS engine lib/test Clippy pass, as do formatting and whitespace
checks. The engine check typechecks the actual production loader, but does not execute CUDA.
The frozen root suite was not rerun. Existing CPU targets were reused with incremental caching
disabled. `evidence.json` records source hashes, lossless raw/development logs and the preserved
CPU executable. Fifteen protected files and four standalone identity implementations are
byte-identical to the root candidate.

## Remaining acceptance

Independent source review is pending. Composite external draft input, target-weight/draft
rewrite identity composition, the separate target-head self-trim/ranks surface, and native
numeric/serving/H2D/I/O qualification remain separate acceptance steps. #542 source and worker/LRU
policy are untouched. No remote job, main integration, rental or topup occurred.
