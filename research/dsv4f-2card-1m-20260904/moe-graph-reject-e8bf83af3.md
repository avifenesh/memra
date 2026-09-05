# Rejected per-layer MoE CUDA graph

Date: 2026-09-05 UTC

Implementation: `64f0316d3`  
Revert: `e8bf83af3`

The arm captured the position-independent routed plus shared MoE body after
eager routing had written stable per-session workspace buffers. Graphs were
keyed by layer and transaction width. The captured arithmetic and pointers were
identical to eager; only fourteen-ish kernel submissions per layer were
collapsed.

On the exact DSV4 0731 Safetensors artifact and two RTX PRO 6000 cards, a
1,025-token, chunk-32 gate passed final-logit and every-live-cache-class bit
identity. Three interleaved runs measured:

| arm | median wall |
| --- | ---: |
| eager | 7.296527 s |
| per-layer MoE graph | 7.330332 s |

Speedup was 0.995x after amortizing the first 43 captures over 32 chunks. This
segment is too small: graph launch plus per-session capture cost consumes the
saved submissions. A viable graph boundary must cover most or all of a layer or
round and deal explicitly with position/cache scalar updates. The arm was fully
reverted.

Durable extracted gate lines are in `remote-gate-lines-20260905.log`. The provider
setting was 500 W/card; no power, clock, bandwidth or compute-limit attribution
is made.
