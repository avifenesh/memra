# GB10 prime chunk: 8192 rows for the serial GDN hybrid trunk (2026-11-06)

**Status:** decided on one device class (unified-memory 120a family, measured on one DGX Spark),
scoped to a trunk with Gated DeltaNet layers, only GDN/full-attention mixers and no mHC topology.
Every other device and family keeps 4,096 rows.

## Question

The eager prime walks 4,096-token chunks (`PRIME_CHUNK_MAX_TOKENS`). On the serial trunk every
chunk attends through the quantized KV cache and GDN boundaries snap to the GDN grid, so the
chunk width cannot move a value (`MEMRA_PRIME_CHUNK` FLAGS row). Is a wider default chunk faster
on GB10, and what does it cost?

## Decision

Armed at model load (`hybrid.rs`, beside the step37 family defaults): 8,192 rows when the trunk is
the serial GDN hybrid program on a unified-memory 120a device. `MEMRA_PRIME_CHUNK` still decides
when set; ring-on walks keep their clamp.

## Evidence

Served A/B on one GB10, Qwen3.8-27B NVFP4+Q5_K GGUF with the masked MTP draft, the round-1 stack
`ef37ca487b`, cells S C8 C16 C16 C8 S, one fresh boot per cell
([`research/gb10-prefill-20261106/round2-chunk-ab/`](../../research/gb10-prefill-20261106/round2-chunk-ab/)):

| chunk | 32,784 tokens | 131,051 tokens | admission workspace (131k) | MemAvailable floor |
|---|---|---|---|---|
| 4,096 | 38.118, 38.143 s | 189.959, 190.031 s | 4,563 MB | 81.4 GiB |
| 8,192 | 38.002, 37.994 s (-0.35%) | 188.544, 189.032 s (-0.64%) | 6,442 MB | 81.3 GiB |
| 16,384 | 38.066, 37.996 s (-0.26%) | 187.766, 187.639 s (-1.21%) | 10,200 MB | 74.5 GiB |

The answer text is identical in all six boots. At 259,951 tokens with 8,192-row chunks the answer
equals base and MemAvailable stayed above 71.8 GiB (`round2-mem-256k-chunk8192/`).

## Rejected

- 16,384 rows: a further -0.57% at 131k, but it books 3.8 GB more admission workspace per
  concurrent 131k prime and cost 7 GiB of MemAvailable on that prime. This profile serves many
  sessions per Spark, so the memory goes to concurrency instead.
- A global default: the mHC trunk (glm5_next) changes bits when its chunk moves (a documented
  near-tie class), so the default is armed per family.
