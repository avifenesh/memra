# GB10 prime chunk: 8192 rows for the serial GDN hybrid trunk (2026-11-06)

**Status:** rejected on 2026-11-06, before landing. The single-request A/B below showed a small
win, but the independent review asked for the peer cost, and the fairness cell (section
"Fairness") showed that a wider chunk doubles the walker's peer quantum. Every device and family
keeps 4,096 rows; the decision as first written is kept below the fairness section.

## Question

The eager prime walks 4,096-token chunks (`PRIME_CHUNK_MAX_TOKENS`). On the serial trunk every
chunk attends through the quantized KV cache and GDN boundaries snap to the GDN grid, so the
chunk width cannot move a value (`MEMRA_PRIME_CHUNK` FLAGS row). Is a wider default chunk faster
on GB10, and what does it cost?

## Fairness (2026-11-06, the reason for the rejection)

The prime walker (`MEMRA_PRIME_YIELD`) advances one chunk per worker tick and serves its peers
between chunks, so the chunk is the peer quantum. One GB10, Qwen3.8-27B GGUF with the masked MTP
draft at 262,144 context, the spec gate pinned (`MEMRA_SPEC_GATE_LOW=64 HIGH=65`), one fresh boot
per cell, cells C4 C8 C8 C4 (`MEMRA_PRIME_CHUNK=4096` against the 8,192 default). Each cell: a cold
131,042-token prompt at t=0, then three cold 2,003-token peers at +10, +15 and +20 s, 32 greedy
tokens each
([`research/gb10-prefill-20261106/fairness-chunk/`](../../research/gb10-prefill-20261106/fairness-chunk/)):

| chunk | long prime TTFT | peer TTFT (+10 / +15 / +20 s) | `tick_max_ms` |
|---|---|---|---|
| 4,096 | 196.42, 197.39 s | 5.09 / 6.70 / 8.66 s; 5.17 / 6.77 / 8.72 s | 7,362; 7,350 |
| 8,192 | 195.21, 195.18 s | 9.62 / 6.85 / 13.23 s; 9.59 / 6.83 / 13.22 s | 14,255; 14,243 |

The long prime gains 0.6%. The worst peer first token goes from 8.7 to 13.2 s (+52%), the first
peer's from 5.1 to 9.6 s, and the longest worker tick doubles. Every request finished with the same
text on both arms. A Spark serves many sessions, so the peer cost outweighs the gain: rejected.
Not tried: arming the wider chunk only while no peer is admitted. It would need its own fairness
cell, because a peer that arrives mid-prime still waits one wide chunk.

## Decision as first written (superseded by the fairness section)

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
