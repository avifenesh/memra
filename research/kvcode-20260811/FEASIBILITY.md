# Speculative KV Coding vs memra's GGUF-derived KV cache

Date: 2026-08-11

Lane: `lane/cx-kvcode`

Scope: read-only feasibility research; no implementation or measurement.

## Verdict

**DOOR-OPEN — narrowly, as an exact prefix/offload codec experiment, default OFF.** The
canonical proposal defines losslessness as reconstructing the target cache exactly, not as
matching within an error tolerance [E1]. It therefore does **not** cross the lossless boundary
that rejects verifier expert masking or reduced-set verification
(`research/spec-landscape-20260810/SURVEY.md:498-512`).

That verdict does **not** validate the advertised ratio on memra and does **not** open the live
attention path. The public artifact calls itself an early research note [E4]; its evaluated
predictor is an FP8 form of the target [E2/E3], while memra's daily cache is packed Q8_0 K plus
Q5_1 V and is consumed directly as flat byte planes
(`crates/memra-kv/src/lib.rs:13-41`, `crates/memra-engine/cu/flash_attn.cu:168-205`). No
public result in [E1-E6] covers those packed symbols, block headers, direct attention reads,
random access, append, coder throughput, or a memra-compatible predictor.

The first eligible arm is therefore **stored-prefix bytes only**: encode the exact packed bytes
that `PrefixCache` currently deep-copies, decode them into the existing flat `KvLayer` planes,
and require the new `KVCODE-BYTE-ID` gate below before running any output-level gate
(`crates/memra-server/src/worker.rs:2217-2271`,
`crates/memra-server/src/worker.rs:2274-2322`). A bounded-lossy residual, a dequantize/requantize
round trip, or a codec that only preserves tokens/logits is **DOOR-CLOSED** for scored work
because none proves reconstruction of the authoritative cache bytes
(`research/spec-landscape-20260810/SURVEY.md:503-512`).

## External evidence quoted

The source of record is Fergus Finn's canonical research note, retrieved 2026-08-11 and marked
modified that day [E4]. These are selected verbatim fragments; later references use the labels
below.

- **[E1] Exactness and endpoint pipeline:** “recovers KV_full exactly”.
  [Canonical note, pipeline](https://fergusfinn.com/blog/kv-entropy-coder/#what-predicts-a-kv-cache)
- **[E2] Evaluated predictor and reported FP8 range:** “predictor is the FP8 version of the
  target”; “3.08× ... 3.90×”.
  [Canonical note, early results](https://fergusfinn.com/blog/kv-entropy-coder/#early-results)
- **[E3] Evaluated predictor topology:** “same architecture”.
  [Canonical note, optimized predictor](https://fergusfinn.com/blog/kv-entropy-coder/#an-optimised-version-of-the-same-model)
- **[E4] Evidence maturity:** “early research note”.
  [Canonical note, maturity](https://fergusfinn.com/blog/kv-entropy-coder/#whats-next);
  [open engineering work](https://fergusfinn.com/blog/kv-entropy-coder/#engineering-throughput-and-bit-identical-predictors)
- **[E5] Named storage use case:** “Bigger prefix caches”.
  [Canonical note, use cases](https://fergusfinn.com/blog/kv-entropy-coder/#whats-it-good-for)
- **[E6] Named transfer use case:** “Cross-datacenter disaggregated prefill”.
  [Canonical note, use cases](https://fergusfinn.com/blog/kv-entropy-coder/#whats-it-good-for)

The note supplies no arXiv identifier, predictor parameter count, host/GPU placement, coder
throughput, end-to-end latency, or public memra-format result [E1-E6]. It describes both endpoints
re-running the predictor and requires identical predictor distributions, but leaves physical
placement and practical throughput as engineering work [E1/E4].

## 1. Existing posture this extends

The house baseline is the own-trimmed MTP regime: byte-verbatim NextN extraction from the serving
GGUF, with every surveyed mechanism treated as an extension rather than a replacement
(`research/spec-landscape-20260810/SURVEY.md:14-24`). Its byte-verbatim provenance rule is explicit
(`docs/DRAFT-REGIME.md:27-31`), and speculative decoding remains exact because target verification
must reproduce the plain target for K=1..8 (`crates/memra-engine/src/bin/run_spec.rs:1-8`).

The existing survey recorded Speculative KV Coding as a default-OFF signpost and already required
`run-gen` plus `run-spec` before adoption
(`research/spec-landscape-20260810/SURVEY.md:640-660`). This report narrows that posture in three
ways supported below:

1. the evaluated predictor is target-shaped FP8, not a demonstrated tiny predictor [E2/E3];
2. memra has flat, quantization-blocked KV planes, not a general paged/block-table cache
   (`crates/memra-kv/src/lib.rs:213-245`, `crates/memra-kv/src/lib.rs:464-547`); and
3. raw KV-byte identity must precede `run-gen`/`run-spec`, because those binaries gate outputs,
   not encoded cache payloads (`crates/memra-engine/src/bin/run_spec.rs:1-8`,
   `crates/memra-engine/src/bin/kernel_check.rs:4285-4336`).

The prefix-cache cross-check in the same survey is otherwise directionally correct: the shipped
surface is an exact-token longest-common-prefix pool, not non-prefix reuse
(`research/spec-landscape-20260810/SURVEY.md:662-678`,
`crates/memra-server/src/worker.rs:1933-1994`).

## 2. What memra actually stores

### GGUF supplies geometry; it does not contain a runtime KV cache

The GGUF reader parses a v3 header, metadata table, tensor table, and tensor-data blob; there is
no runtime-cache section in that on-disk layout (`crates/memra-gguf/src/lib.rs:1-10`). Model
geometry comes from GGUF metadata: key/value head dimensions are read from
`attention.key_length` / `attention.value_length`, and the KV-head count comes from
`attention.head_count_kv` with architecture-specific handling
(`crates/memra-gguf/src/config.rs:502-513`, `crates/memra-gguf/src/config.rs:719-767`).

The runtime encoding is selected independently by `MEMRA_KV_K` / `MEMRA_KV_V`: default Q8_0 K
and Q5_1 V, with explicit FP8 K and Q4_0/FP8 V alternatives
(`crates/memra-kv/src/lib.rs:11-41`). The engine loads a cache-format-specific attention fatbin,
and treats every non-default pair as a separate numeric configuration
(`crates/memra-engine/src/lib.rs:286-313`). Therefore “GGUF KV format” means **GGUF-derived
geometry plus GGML-compatible runtime quant blocks**, not a KV payload serialized in the model
file (`crates/memra-gguf/src/lib.rs:43-88`,
`crates/memra-engine/cu/flash_attn.cu:168-180`).

**Format verdict: N/A for GGUF serialization.** A volatile Speculative-KV bitstream does not fit
an existing GGUF runtime-cache field because no such field exists
(`crates/memra-gguf/src/lib.rs:1-10`). The model GGUF may remain unchanged; any persistent or
transported cache would need its own manifest/protocol keyed to the model, prompt tokens, selected
KV formats, geometry, predictor, and entropy model. Those keys are required to reproduce the
inputs that current allocation derives from GGUF and environment policy
(`crates/memra-gguf/src/config.rs:502-513`, `crates/memra-kv/src/lib.rs:13-41`).

### Per-layer, per-head layout and dtype

Each full-attention layer owns a `KvLayer` with separate GPU `u8` K and V planes. Within one token,
elements are ordered `[kv_head, dim]`; every 32-element quant block remains inside a head, and
K/V have independent token-byte strides (`crates/memra-kv/src/lib.rs:213-232`). The allocator
computes the row width from KV-head count times K/V head dimension, requires each width to be
divisible by 32, and resolves per-layer FP8 doors before sizing
(`crates/memra-kv/src/lib.rs:285-340`).

The daily block layouts are byte-structured, not merely scalar arrays: Q8_0 carries one f16 scale
plus 32 int8 codes in 34 bytes; Q5_1 carries f16 scale, f16 minimum, a 32-bit high-bit mask, and 16
nibble bytes in 24 bytes (`crates/memra-engine/cu/flash_attn.cu:168-205`). The same layouts are
registered as GGML Q8_0/Q5_1 types by the GGUF reader
(`crates/memra-gguf/src/lib.rs:43-50`, `crates/memra-gguf/src/lib.rs:71-88`).

Prefill appends post-RoPE K/V through the same packed-row quantizer used by decode, then advances
the layer length (`crates/memra-engine/src/hybrid_forward.rs:2399-2403`,
`crates/memra-engine/src/hybrid_forward.rs:2502-2513`). Decode appends one packed row and gives
attention byte views covering exactly `len * token_stride`
(`crates/memra-engine/src/decode.rs:2758-2801`). Consequently, a lossless codec must reproduce
the **packed plane bytes**, including scale/min/header bits; reconstructing dequantized floats and
re-quantizing them is not the authoritative identity operation
(`crates/memra-engine/cu/flash_attn.cu:168-205`).

Hybrid linear-attention layers do not own growing K/V planes; they own fixed recurrent state, while
full-attention layers receive `KvLayer` allocations
(`crates/memra-gguf/src/config.rs:96-101`, `crates/memra-kv/src/lib.rs:489-545`). Speculative KV
Coding is therefore scoped to `Cache.kv` planes; recurrent state is outside the claimed method and
must remain unchanged (`crates/memra-kv/src/lib.rs:265-283`; [E1]).

### Paging and allocation reality

The general cache is not PagedAttention: a `KvLayer` owns one contiguous K `CudaSlice<u8>` and one
contiguous V `CudaSlice<u8>`, and the allocator reserves the layer's physical rows directly
(`crates/memra-kv/src/lib.rs:217-232`, `crates/memra-kv/src/lib.rs:464-534`). Attention consumes a
contiguous prefix/range view, not a block table
(`crates/memra-engine/src/hybrid_forward.rs:2554-2582`,
`crates/memra-engine/src/decode.rs:2776-2801`). The 32-element Q8_0/Q5_1 units are **quantization
blocks**, not cache pages (`crates/memra-engine/cu/flash_attn.cu:168-180`).

The one alternate physical layout is a default-OFF Step35 sliding-window ring. It rebases retained
rows so the reader still receives one contiguous view; it does not expose page ids or a block
table (`crates/memra-kv/src/lib.rs:77-92`, `crates/memra-kv/src/lib.rs:95-193`). Prefix snapshot and
restore explicitly refuse ring sessions
(`crates/memra-server/src/worker.rs:2217-2228`,
`crates/memra-server/src/worker.rs:2274-2284`). **Paged-KV interaction is therefore N/A on current
main; SWA-ring interaction is also N/A for the proposed prefix-cache first arm.**

Under PP-N/PP-2 placement, each layer's cache is allocated on the device of its owning stage, and
only the hidden-state boundary crosses stages
(`crates/memra-engine/src/pp.rs:43-49`, `crates/memra-engine/src/pp.rs:1253-1284`). Speculative KV
Coding therefore has no current PP-boundary KV transfer to shrink. It could only change per-stage
capacity or the local attention representation, neither of which [E1-E6] evaluates for memra.

### Prefix-cache representation

`PrefixCache` is keyed by `(model, cache namespace)` and scans exact token prefixes within that
pool (`crates/memra-server/src/worker.rs:786-794`,
`crates/memra-server/src/worker.rs:1825-1830`,
`crates/memra-server/src/worker.rs:1973-1994`). A `PrefixEntry` stores one K/V plane pair per layer,
the boundary position, recurrent state, logits, and byte charge
(`crates/memra-server/src/worker.rs:1793-1817`).

Snapshot copies exactly `len * k_tok_bytes` and `len * v_tok_bytes` from every layer into compact
GPU buffers (`crates/memra-server/src/worker.rs:2217-2245`). Restore first allocates a fresh live
cache, copies those bytes back plane-by-plane, restores the lengths, and then serves the session
(`crates/memra-server/src/worker.rs:2274-2322`,
`crates/memra-server/src/worker.rs:5171-5206`). This is a clean lossless storage boundary for
[E5], but compressing an entry would not reduce the fresh live cache allocated on a hit
(`crates/memra-server/src/worker.rs:5183-5206`).

Current prefix reuse is also not automatically composed with the house speculative path: the
prefix probe runs only when the request is not `spec_eligible`
(`crates/memra-server/src/worker.rs:5158-5182`). The coding method is mathematically distinct from
token speculation, as the survey notes (`research/spec-landscape-20260810/SURVEY.md:649-653`), but
the present server would need a separate integration decision before an MTP session could consume
a coded `PrefixEntry`.

## 3. What the public proposal proves—and does not

| Question | Pinned answer | Evidence |
| --- | --- | --- |
| Exact or tolerance-bounded? | **Exact by construction.** The decoder is supposed to recover the target cache, not an approximation. This is the decisive reason the door is not closed. | [E1] |
| Exact relative to what? | Relative to the selected target cache. The reported FP8 arm codes an already-quantized FP8 target, so its losslessness is inside that numeric configuration; it does not undo BF16→FP8 loss. | [E1/E2] |
| Demonstrated memra byte identity? | **No.** The source reports Qwen3 BF16/FP8-symbol rates, while memra's default target bytes are Q8_0/Q5_1 blocks with per-block headers. | [E2]; `crates/memra-engine/cu/flash_attn.cu:168-205` |
| Evaluated predictor size? | Target-equivalent parameter count is the only defensible reading: the concrete predictor is the same architecture with narrower weights. This is an inference from the construction; the source gives no independent parameter count. | [E2/E3] |
| “Tiny predictor” demonstrated? | **No.** A smaller transformer plus learned shape maps is proposed as future work, not used for the reported table. | [E2/E4] |
| Predictor work per token/prefix? | Both endpoints run a full predictor forward over the same token sequence. No FLOP count, latency, memory footprint, or amortized per-token cost is reported. | [E1/E2/E4] |
| Host or device? | **Unspecified.** “Encode side” and “decode side” are logical endpoints, not hardware-placement claims. | [E1/E4] |
| Practical coder? | Arithmetic coding supplies the exactness argument, but coder throughput and bit-identical predictor distributions remain listed engineering work; no end-to-end memra-format round trip is reported. | [E1/E4] |
| Reported ratio scope? | The note reports 2.37–2.70× for its BF16 table and 3.08–3.90× for its FP8-target table across Qwen3 sizes after C4 calibration; neither range transfers to Q8_0/Q5_1 without a new experiment. | [E2]; `crates/memra-kv/src/lib.rs:13-41` |
| Source-supported serving shapes? | Candidate uses are transferred/disaggregated prefill and larger stored prefix caches; the note also discusses offloaded-cache transfer. It does not report a long-context hot-attention, high-concurrency scheduler, or PP-2 result. | [E5/E6] |

## 4. Memra compatibility matrix

| Surface | Status | Reason and required boundary |
| --- | --- | --- |
| GGUF model file | **N/A** | GGUF provides tensors and geometry, not runtime KV payloads (`crates/memra-gguf/src/lib.rs:1-10`, `crates/memra-gguf/src/config.rs:502-513`). Keep GGUF unchanged. |
| Default Q8_0-K/Q5_1-V planes | **DOOR-OPEN for exact byte coding** | Encode packed bytes directly, including block headers; do not code dequantized values and re-quantize (`crates/memra-engine/cu/flash_attn.cu:168-205`). Published ratios are N/A [E2]. |
| Live resident attention cache | **HOLD / no first arm** | Attention reads flat byte ranges every step; [E1-E6] supplies no direct compressed-read kernel, random-access format, append protocol, or coder-throughput evidence (`crates/memra-engine/src/decode.rs:2758-2801`). |
| Cross-request `PrefixCache` storage | **Best first arm** | Snapshot/restore already forms an exact compact-byte boundary; decode into today's planes before restore (`crates/memra-server/src/worker.rs:2217-2322`). This matches the source's stored-prefix use case [E5]. |
| Active session allocation | **Unchanged by first arm** | A prefix hit allocates a fresh flat live cache before restoration, so entry compression alone saves only stored-entry bytes (`crates/memra-server/src/worker.rs:5183-5206`). |
| Step35 SWA ring | **N/A now** | Ring sessions are explicitly refused by flat prefix snapshot/restore (`crates/memra-server/src/worker.rs:2217-2228`, `crates/memra-server/src/worker.rs:2274-2284`). |
| General paged KV | **N/A—absent** | Current cache owns contiguous per-layer planes and range views, not block tables (`crates/memra-kv/src/lib.rs:217-245`, `crates/memra-engine/src/hybrid_forward.rs:2554-2582`). |
| PP-2 | **No transfer win identified** | KV is stage-owned and only hidden state crosses the boundary (`crates/memra-engine/src/pp.rs:43-49`, `crates/memra-engine/src/pp.rs:1253-1284`). Capacity is a possible later question, not a source-backed result. |
| House MTP/spec sessions | **Orthogonal in theory; not composed today** | The method codes verified cache, but current prefix admission excludes spec-eligible sessions (`crates/memra-server/src/worker.rs:5158-5182`). |
| Bounded-lossy residual variant | **DOOR-CLOSED** | It changes the authoritative cache and fails the same lossless doctrine as reduced-set verification (`research/spec-landscape-20260810/SURVEY.md:503-512`). |

## 5. First gate: `KVCODE-BYTE-ID`

This gate must be added **before** any future implementation can run a scored arm. It is stricter
than the current KV quantization round-trip cell, which compares dequantized floats with a
tolerance rather than proving cache-byte equality
(`crates/memra-engine/src/bin/kernel_check.rs:4285-4336`).

### Frozen inputs

Record the runtime commit, source GGUF hash, exact prompt token ids, selected K/V formats, model
geometry, prefix length, predictor artifact hash, calibration/entropy-table hash, and codec
version. The source GGUF and environment jointly determine the current layout
(`crates/memra-gguf/src/config.rs:502-513`, `crates/memra-kv/src/lib.rs:13-41`).

### Reference and candidate

1. Prime the ordinary default-OFF reference and capture, for every full-attention layer, exactly
   `k[0..len*k_tok_bytes]`, `v[0..len*v_tok_bytes]`, `len`, `len_d`, and `cache.pos`; these are the
   same authoritative ranges that attention and prefix snapshot consume
   (`crates/memra-engine/src/decode.rs:2773-2801`,
   `crates/memra-server/src/worker.rs:2234-2245`).
2. Encode those packed bytes, independently reconstruct the predictor distribution at the decode
   endpoint, and decode into fresh buffers. Do not dequantize/requantize
   (`crates/memra-engine/cu/flash_attn.cu:168-205`; [E1]).
3. Compare every decoded K byte, V byte, length, and position with the reference. **Pass requires
   zero differing bytes and identical metadata at every layer.** Any predictor-distribution
   disagreement, including a one-ULP endpoint difference called out by the source, is a failure
   [E4].
4. Repeat the codec invocation twice and require an identical compressed bitstream and identical
   decoded bytes. This separates deterministic codec proof from downstream token coincidence [E1/E4].

The first fixture matrix must include the daily Q8_0/Q5_1 pair, synthetic block contents that
exercise every scale/min/header/code field, and prefix lengths around split/chunk boundaries;
those are the current layout and serving seams (`crates/memra-kv/src/lib.rs:13-41`,
`crates/memra-engine/src/hybrid_forward.rs:2502-2513`,
`crates/memra-server/src/worker.rs:5171-5245`). Explicit alternative KV formats are separate
numeric configurations and require their own within-format byte gate
(`crates/memra-engine/src/lib.rs:286-313`).

### Downstream gates only after byte identity

After `KVCODE-BYTE-ID` is green, the default-OFF arm must still pass `run-gen` argmax MATCH and
the naked `run-spec` K=1..8 sweep, matching the standing local battery
(`tools/local-ci.sh:109-168`). For a prefix-cache arm, restored cache bytes and continuation token
bytes must also match the uncoded snapshot/restore path per request; the dual-PP design provides
the relevant default-OFF, every-request hash precedent
(`research/dualpp-20260811/DESIGN.md:100-112`).

`run-gen`/`run-spec` success cannot waive a raw-byte mismatch. A mismatch means the codec changed
the cache even if the tested prompt did not cross an argmax boundary; that is a lossless-bar
failure, not a tolerance to tune (`research/spec-landscape-20260810/SURVEY.md:503-512`).

## 6. Stop conditions and evidence still missing

Close the scored door immediately if reconstruction is approximate, if endpoint predictor state
is not bit-identical, or if any packed cache byte differs [E1/E4]. Keep the live attention door
closed until a separate design supplies append/random-access semantics and proves exact packed-byte
consumption against the current direct byte views
(`crates/memra-engine/src/decode.rs:2758-2801`).

The following are deliberately **unknown**, not performance claims:

- compression ratio for memra Q8_0/Q5_1 block bytes and their scale/min headers [E2];
- predictor parameter/storage cost for a genuinely smaller predictor [E4];
- predictor and coder host/device placement [E1/E4];
- encode/decode throughput, memory high-water, and whether the predictor repays itself [E4];
- block-random access, incremental append, rollback, and direct compressed-attention semantics
  relative to memra's flat planes (`crates/memra-kv/src/lib.rs:403-413`,
  `crates/memra-engine/src/decode.rs:2758-2801`);
- any benefit to PP-2, where KV does not cross the stage boundary
  (`crates/memra-engine/src/pp.rs:43-49`); and
- any interaction with Step35 ring prefix reuse, which current code refuses
  (`crates/memra-server/src/worker.rs:2217-2228`).

No throughput or quality claim for memra follows from this report. The only positive conclusion is
semantic: the canonical proposal aims at exact reconstruction, so a byte-identity-first storage
experiment is admissible; everything beyond that remains unproven [E1/E4].
