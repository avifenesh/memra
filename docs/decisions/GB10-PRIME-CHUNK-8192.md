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
The variant that arms the wider chunk only while no peer is admitted was measured on 2026-11-07
(next section) and is rejected too.

## Wide calls only while the prime is alone (2026-11-07, rejected)

F8F4 GEMM v2 (memra#1077) builds each weight image once per prime call, so a wider call saves
image passes as well as GEMM tails, and the single-request gain grew. The variant: the walker
merges two adjacent 4,096-row ranges of one trunk segment into one 8,192-row call while the prime
is the worker's only active request, nothing is queued and no admission is pending. A peer
admitted or waiting turns merging off at the next call. Admission books the 8,192-row workspace
(6,442 MB per 131k prime instead of 4,563 MB). The code, armed on a unified-memory 120a device
for the serial GDN hybrid dense trunk with `MEMRA_PRIME_WIDE=0` as the off arm, is kept on the ref
`research/gb10-prime-wide-measured` (`cc9592a7e4`); it did not land.

Fairness cell, the same harness as above on that one binary, unit 2, cells W0 W1 W1 W0
(`MEMRA_PRIME_WIDE=0` against the armed default)
([`research/gb10-prefill-20261106/prime-wide-fairness-unit2/`](../../research/gb10-prefill-20261106/prime-wide-fairness-unit2/)):

| arm | long prime TTFT | peer TTFT (+10 / +15 / +20 s) | `tick_max_ms` |
|---|---|---|---|
| 4,096 only | 181.89, 181.65 s | 3.76 / 4.91 / 6.40 s; 3.77 / 4.92 / 6.41 s | 6,910; 6,899 |
| wide while alone | 180.94, 181.11 s | 7.07 / 8.30 / 5.54 s; 7.10 / 8.34 / 5.57 s | 8,653; 8,645 |

Every request returned the same text on both arms. The prime is alone for its first 10 s, so it
runs two wide calls there (7.10 and 7.33 s), and the first peer, at +10 s, arrives inside the
second: its first token goes from 3.8 to 7.1 s (+88%), the worst peer's from 6.4 to 8.3 s (+30%).
The long prime itself gains 0.4%, because peers turn merging off for the rest of it.

Single request, the same binary, unit 2, cells P V V P (P = `MEMRA_PRIME_WIDE=0`), one cold
prompt per length per boot
([`research/gb10-prefill-20261106/prime-wide-served-unit2/`](../../research/gb10-prefill-20261106/prime-wide-served-unit2/)):

| arm | 32,784 tokens | 131,051 tokens | 259,951 tokens | MemAvailable floor |
|---|---|---|---|---|
| 4,096 only | 32.760, 32.560 s | 168.257, 167.913 s | 434.489, 431.313 s | 66.0, 66.4 GiB |
| wide while alone | 31.291, 31.254 s (-4.25%) | 161.539, 161.557 s (-3.89%) | 419.192, 417.965 s (-3.31%) | 64.7, 64.7 GiB |

The answer text is identical in all four boots and to the landed v2 boots. Alone, the wide calls
are worth about 4%.

The gate for this variant was that the worst peer beside a 131k prime must not regress against
4,096 rows. Measured against 4,096 rows on the same binary it regresses by 30%. It stays under
the 8.7 s that 4,096 rows gave in the 2026-11-06 cell, but that cell ran the pre-v2 binary, whose
4,096-row calls were slower; measured against that number, the variant would hand back peer
latency that v2 earned. Any call wider than the quantum keeps a peer that arrives during it
waiting longer, and a time cap that kept wide calls under the 4,096-row tick at depth would leave
almost nothing to merge. 4,096 rows stay, and the code did not land.

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
