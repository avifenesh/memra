# Decision records

What was chosen, what was rejected, and the measurement that settled it. Kept permanently:
these are the rationale half of this repo's record, and a rejection is often the more useful
half — "we tried X and it lost by Y% on Z" cannot be reconstructed from the code that shipped.

| Record | Decides |
|---|---|
| [MTP-LEARNED-DEPTH-20260920.md](MTP-LEARNED-DEPTH-20260920.md) | full-head MTP depth research: fixed, native adaptive and learned policies on Qwen and Gemma |
| [SAFETENSORS-DECISION.md](SAFETENSORS-DECISION.md) | safetensors as the semantic source, and what that does not mean for the compute format |
| [FORMAT-DECISION.md](FORMAT-DECISION.md) | which artifact formats the engine imports and serves |
| [QUANT-GEMM-DECISION.md](QUANT-GEMM-DECISION.md) | the quantized GEMM path |
| [RIG-NATIVE-DECODE.md](RIG-NATIVE-DECODE.md) | decoding in rig-native layouts rather than the source layout |
| [PHASE1-HYBRID.md](PHASE1-HYBRID.md) | the hybrid phase-1 shape |
| [BEST-OF-ALL-WORLDS.md](BEST-OF-ALL-WORLDS.md) | per-device arm selection instead of one compromise default |
| [VISION-LANE.md](VISION-LANE.md) | in-engine vision tower, and the parity oracle that gates it |
| [PUBLIC-BOUNDARY-DETECTION.md](PUBLIC-BOUNDARY-DETECTION.md) | what the public-boundary gate matches, which candidate rules were rejected as too noisy, and why published refs need their own scan |
| [ORNITH-PAIR-OWNER.md](ORNITH-PAIR-OWNER.md) | why source-verbatim pair-owner MoE ordering stays out of the runtime while its receipts and exact candidate remain banked |
| [PRO6000-MULTICARD.md](PRO6000-MULTICARD.md) | why PP is the 2–4 card PCIe baseline, and when dense TP or MoE TP/EP/ETP may replace it |
| [KV-PHYSICAL-RECLAIM.md](KV-PHYSICAL-RECLAIM.md) | VMM-backed KV planes for physical demote/reclaim (gate-only door); pool trim rejected by measurement |
| [PREFIX-CACHE-POLICY.md](PREFIX-CACHE-POLICY.md) | plain LRU as the only prefix-cache eviction policy; the segmented (SLRU) arm and its two doors deleted after losing every pair of the incident-shape A/B |
| [PINNED-DESTINATIONS.md](PINNED-DESTINATIONS.md) | the contract path's pinned host destinations cached on the RTX PRO 6000 Blackwell class and write-combined on the RTX 5090 class and elsewhere, a per-device default from two pre-registered A/B cells; an env door and a global flip rejected |
| [DSV4-TPEP-DEFAULT.md](DSV4-TPEP-DEFAULT.md) | DSv4-Flash on two cards loads TP/EP (expert-ID EP plus exact attention TP2, plain steps on full-token replay) by default, PP-2 the rollback; its plain-concurrency and context costs and the lanes that close them |
| [COMPLETE-RESULT-PATH-V1.md](COMPLETE-RESULT-PATH-V1.md) | the async/background delivery contract for non-streaming requests that outlast the 90 s synchronous deadline: a per-request opt-in, not a conversation-store restore; design frozen, implementation and box verification still owed |
| [RELEASE-PROFILE.md](RELEASE-PROFILE.md) | the workspace release profile (opt-level 3, fat LTO, one codegen unit): what it can and cannot reach, tokenization 6% faster, other serving flat, results byte-identical, binary 17% smaller, relinks 1.5-6.7x slower |
| [SM121-SPARK-TARGET.md](SM121-SPARK-TARGET.md) | GB10 (DGX Spark) builds its own `121a` SASS in the `120a` policy family; the one-binary `120f` family build rejected |
| [GB10-PRIME-ATTENTION-H2.md](GB10-PRIME-ATTENTION-H2.md) | GB10 primes quantized-KV attention with the two-head (h2) twin by default (exact, -13.9% served 131k TTFT); t3 and the FA2 numerical class rejected there. Superseded on GB10 by the h2p ring (GB10-PRIME-ATTENTION-H2P.md), which keeps h2's numbers |
| [GB10-PRIME-ATTENTION-H2P.md](GB10-PRIME-ATTENTION-H2P.md) | GB10 runs h2's row program on a three-stage mbarrier K/V ring with the second head one QK behind the first (h2p), bit-identical to h2; probe 68 -> 85 TFLOPS at depth 131,072; one-barrier tiles, Q in shared memory, FA2 staging, mask-free interior tiles rejected |
| [GB10-PRIME-GDN-GLUE.md](GB10-PRIME-GDN-GLUE.md) | GB10's prime: GDN compact q/k at any dividing head ratio (16 key / 48 value broadcast three times), heads-fastest K2/K3/K5, fused K4+K5 (Y and chunk-start states stay on chip), fused add + RMSNorm writing the next GEMM's F8F4 v2 / q8_1 activation, gate and up in one launch writing the down GEMM's silu activation, a tiled dp4a matvec; all bit-identical, prime bench 32.19 -> 25.70 s at 33,890 tokens; K4-only rewrites rejected |
| [GB10-PRIME-GLUE-2.md](GB10-PRIME-GLUE-2.md) | GB10's prime, second pass: the residual add in the F8F4 v2 GEMM epilogue (wo, ssm_out, down), K2 at C = 32 in half the shared memory, the GDN conv with the q/k L2 norms, the attention q split + q-norm + RoPE in one kernel; all bit-identical, prime bench -3.7% at 33,890 tokens over the first pass, served -3.3% at 131k; the beta/alpha dp4a reshapes rejected (probe) |
| [GB10-PRIME-CHUNK-8192.md](GB10-PRIME-CHUNK-8192.md) | REJECTED 2026-11-06: an 8,192-row GB10 prime chunk default (exact, -0.64% on a lone 131k prime) doubles the prime walker's peer quantum (worst peer first token 8.7 -> 13.2 s, tick_max 7.4 -> 14.2 s); 4,096 stays. 16,384 was rejected earlier for its admission and memory cost |
| [GB10-F8F4-GEMM-V2.md](GB10-F8F4-GEMM-V2.md) | GB10 runs F8F4 prefill GEMMs of at least 2,048 rows on the v2 tile (per-call weight image, pipelined 16-warp 256 x 128 tile, grouped raster; exact, -8.0% served 131k TTFT at 4,096-row chunks on both units, -10.7% at 8,192); two 8-warp CTAs per SM and resident images rejected |

Adding one: a decision that changes a default, a format, a target or an arm belongs here, with
the measurement that settled it. Superseded records get a banner naming what replaced them —
they are not deleted. See `AGENTS.md` § "Measurements and decisions are a corpus".
