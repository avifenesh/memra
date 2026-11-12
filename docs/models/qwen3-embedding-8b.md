# Qwen3-Embedding-8B

Qwen's 8B embedding model, a capture model: served through `POST /v1/embeddings` from a Q8_0 GGUF
since v0.116.0. It runs prefill only, so it has no drafter, no speculative path and no board row.

| | |
|---|---|
| **Serving route** | Q8_0 GGUF through `/v1/embeddings`, beside a chat model in one process. Gated before v0.116.0 against a vendor `transformers` fp16 reference on pinned real-corpus probes; no tracked receipt holds the gate figures. Standing gate: none |
| **Pack** | no pack record names this checkpoint |
| **Artifact** | the official `Qwen/Qwen3-Embedding-8B-GGUF` Q8_0 file (no revision pin is tracked) |
| **Output** | the last-token post-final-norm hidden state, L2-normalized, 4,096 dimensions; `dimensions` truncates (MRL) and re-normalizes |

## Run it

No cookbook block. Add an `alias=path` entry to `MEMRA_MODELS`; a capture model loads plain beside
a speculative chat model. The request and response schema is in
[API-SURFACES.md](../API-SURFACES.md), and the serving contract in
[serving](../SERVING.md#embeddings-and-rerank--the-capture-surfaces-laneembed-serve-2026-08-26).

## Measured

Reported, no tracked receipt: the embedding gate is a cosine floor of 0.995 against the fp16
reference, with both prefill paths exercised (prime at or above the prime floor, `decode_step_h`
below it). A three-model process (a speculative chat model plus both capture models) was run with
`/health` listing all three.

## Limits

- An f16 GGUF refuses at load (`embed_gather: unsupported dtype F16`).
- On a process with a Q8 capture model, set `MEMRA_Q8RP=0`: the capacity-keyed Q8 decode mirrors
  arm on the embedder by default and cost 8.5 GB for a decode path a capture model never takes.
- Serve capture models on batch-class keys so they ride the harvest lane and shed under interactive
  load with a retryable 429 ([serving](../SERVING.md)).

## Deeper sources

- Code: `crates/memra-server/src/embed_api.rs`.
- The original capture-model section: [history](../history/MODELS.md#capture-models--embeddings-and-rerank-v01160).
- The reranker: [Qwen3-Reranker-8B](qwen3-reranker-8b.md).
