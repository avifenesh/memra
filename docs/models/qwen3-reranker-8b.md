# Qwen3-Reranker-8B

Qwen's 8B reranker, a capture model: served through `POST /v1/rerank` from a Q8_0 GGUF since
v0.116.0. It runs prefill only, so it has no drafter, no speculative path and no board row.

| | |
|---|---|
| **Serving route** | Q8_0 GGUF through `/v1/rerank`, beside a chat model in one process. Gated before v0.116.0 against a vendor `transformers` fp16 reference; no tracked receipt holds the gate figures. Standing gate: none |
| **Pack** | no pack record names this checkpoint |
| **Artifact** | the community `mradermacher/Qwen3-Reranker-8B-GGUF` Q8_0 file, minted f16 then `llama-quantize` (no revision pin is tracked) |
| **Output** | P("yes") over the {"yes", "no"} logit pair at the final position, under the Qwen3-Reranker prompt |

## Run it

No cookbook block. Add an `alias=path` entry to `MEMRA_MODELS`; a capture model loads plain beside
a speculative chat model. The request and response schema is in
[API-SURFACES.md](../API-SURFACES.md), and the serving contract in
[serving](../SERVING.md#embeddings-and-rerank--the-capture-surfaces-laneembed-serve-2026-08-26).

## Measured

Reported, no tracked receipt: the rerank gate is strict full-order parity plus |Δscore| ≤ 0.12
against the reference logit-score path. The fp16 directory scored 32/32 positions exact (max |Δ|
0.0996); the shipped Q8_0 file scored 30/32 exact, with one bottom-two tail swap at reference
margin 0.052, under the Q8 tail rule (strict top-half parity, tail flips under 0.06).

## Limits

- Do not mint this model with `convert_hf_to_gguf --outtype q8_0` directly from the safetensors: the
  rerank head comes out broken (order scrambled on every gate query, reproduced from two
  independent llama.cpp checkouts). The f16 then `llama-quantize` pipeline of the same weights
  passes.
- An f16 GGUF refuses at load (`embed_gather: unsupported dtype F16`), so f16 cannot serve as the
  discriminator.
- Serve capture models on batch-class keys so they shed under interactive load
  ([serving](../SERVING.md)); `MEMRA_Q8RP=0` saves the Q8 decode mirrors a capture model never uses.

## Deeper sources

- Code: `crates/memra-server/src/embed_api.rs`.
- The original capture-model section: [history](../history/MODELS.md#capture-models--embeddings-and-rerank-v01160).
- The embedder: [Qwen3-Embedding-8B](qwen3-embedding-8b.md).
