# Decision records

Each record states what was chosen, what was rejected, and the measurement that settled it. Read
the record before changing a default, format, target or arm it covers. Records are kept
permanently, and a rejection is often the more useful half: "we tried X and it lost by Y% on Z"
cannot be reconstructed from the code that shipped.

Verdicts: **Adopted** (in force), **Rejected** (measured and not taken), **Amended** (in force with
a later change the record names), **Superseded** (replaced; the row names the replacement),
**Plan** (an early design or build plan; the code has moved past it). The record wins over its
line here.

## Formats, sources and build targets

| Record | Date | Verdict | Decides |
|---|---|---|---|
| [FORMAT-DECISION.md](FORMAT-DECISION.md) | 2026-08-15 | Adopted | Official safetensors semantics in, rig-native layouts inside; neither GGUF blocks nor HF layouts are the optimization target. |
| [SAFETENSORS-DECISION.md](SAFETENSORS-DECISION.md) | undated | Adopted | HF safetensors loading through the `safetensors` crate (mmap per shard, routing by `index.json`) and the HF-to-ggml name map. |
| [QUANT-GEMM-DECISION.md](QUANT-GEMM-DECISION.md) | 2026-06-26 | Adopted | Resident quantized GEMMs (llama.cpp-style MMVQ decode and MMQ prefill, per-tensor dtype dispatch) instead of f32 dequantization at load. |
| [RIG-NATIVE-DECODE.md](RIG-NATIVE-DECODE.md) | 2026-08-15 | Adopted | Rig-native decode on sm_120a means fused multi-tensor walks (fewer launches, larger transfers), not a new layout; each increment gated. |
| [SM121-SPARK-TARGET.md](SM121-SPARK-TARGET.md) | 2026-11-01 | Adopted | GB10 (DGX Spark) builds its own `121a` SASS in the `120a` policy family; one `120f` family binary rejected. |
| [RELEASE-PROFILE.md](RELEASE-PROFILE.md) | 2026-11-02 | Adopted | Release profile opt-level 3, fat LTO, one codegen unit: results byte-identical, tokenization 6% faster, binary 17% smaller, relinks 1.5-6.7x slower. |
| [GRID-Y-ROW-BLOCKS.md](GRID-Y-ROW-BLOCKS.md) | 2026-11-11 | Adopted | Launches with a row count in CUDA grid y split into blocks of at most 65,535 rows; launches without a split refuse by name (memra#1200). |
| [PUBLIC-BOUNDARY-DETECTION.md](PUBLIC-BOUNDARY-DETECTION.md) | 2026-08-18 | Adopted | What the public-boundary gate matches, which candidate rules were rejected as too noisy, and why published refs need their own scan. |

## Correctness gates

| Record | Date | Verdict | Decides |
|---|---|---|---|
| [ARGMAX-GATE-GAP-SHIFT-FLOOR.md](ARGMAX-GATE-GAP-SHIFT-FLOOR.md) | 2026-11-10 | Adopted | The argmax gate judges each row's gap shift against a floor calibrated per model, prompt, decode program and device; a cell with no floor is UNCALIBRATED (memra#1199). |
| [ARGMAX-GATE-STORED-ROWS.md](ARGMAX-GATE-STORED-ROWS.md) | 2026-11-10 | Adopted | The argmax gate also compares a cell's window byte for byte with main's stored rows for the same cell; tolerances and a base build at gate time rejected (memra#1204). |

## Serving, admission and delivery

| Record | Date | Verdict | Decides |
|---|---|---|---|
| [ADMIT-BY-MEMORY-DEFAULT.md](ADMIT-BY-MEMORY-DEFAULT.md) | 2026-11-04 | Adopted | Memory-shaped admission (32,768-token open-output charge) is the default on the full-power RTX PRO 6000 Workstation Edition; `MEMRA_ADMIT_BY_MEMORY` deleted. |
| [ADMIT-W-RELEASE-DEFAULT.md](ADMIT-W-RELEASE-DEFAULT.md) | 2026-11-04 | Adopted | The prefill workspace leaves both admission books at prime completion on the full-power RTX PRO 6000 Workstation Edition; the door deleted. |
| [BATCH-OOM-RECOVER-DEFAULT.md](BATCH-OOM-RECOVER-DEFAULT.md) | 2026-11-04 | Amended | A batched decode chunk's untouched CUDA OOM retries once on the RTX PRO 6000 Workstation Edition, and since 2026-11-08 on the RTX 5090 Laptop GPU. |
| [C8-SERVING-24GB.md](C8-SERVING-24GB.md) | 2026-11-07 | Adopted | c8 on a 24 GB card: a memory-deferred request holds the queue, deadline-first prefill only at risk, seed rows join the concat prime, the cache yields the transient reserve. |
| [PREFILL-WORKSPACE-HELD.md](PREFILL-WORKSPACE-HELD.md) | 2026-11-08 | Adopted | On the RTX 5090 Laptop GPU admission charges the prefill workspace a prime actually holds and bounds concat batches by free memory. |
| [PRIME-FAIRNESS-DEFAULT.md](PRIME-FAIRNESS-DEFAULT.md) | 2026-09-22 | Adopted | The owned prime walker yields to its peers by default; `MEMRA_PRIME_YIELD=0` is the rollback seam (memra#521). |
| [BACKGROUND-DELIVERY-DEFAULT.md](BACKGROUND-DELIVERY-DEFAULT.md) | 2026-11-09 | Adopted | `background: true` is always accepted on Responses, chat and text completions; `MEMRA_BACKGROUND_RESPONSES` deleted on two PASS receipts. |
| [COMPLETE-RESULT-PATH-V1.md](COMPLETE-RESULT-PATH-V1.md) | 2026-09-28 | Amended | The delivery contract for non-streaming requests past the 90 s deadline: a per-request opt-in, not a conversation store. The record tracks what is still owed. |
| [GEMMA-UNOPENED-THOUGHT.md](GEMMA-UNOPENED-THOUGHT.md) | 2026-11-04 | Adopted | gemma4 parsers route text before an unopened `<channel\|>` at the start of a turn to reasoning; a stream holds at most 256 bytes. |
| [POOLED-MEMBER-CLOSE.md](POOLED-MEMBER-CLOSE.md) | 2026-11-11 | Adopted | A pooled MTP or DSpark cohort member closes at the round its stream ended, not when the whole cohort call returns (memra#1209). |
| [HANDOFF-DIRECT-EXPORT.md](HANDOFF-DIRECT-EXPORT.md) | 2026-11-04 | Adopted | The host-tier handoff export writes with `O_DIRECT` on the full-power RTX PRO 6000 Workstation Edition, buffered elsewhere; `MEMRA_KV_HOST_HANDOFF_IO` deleted. |
| [VISION-LANE.md](VISION-LANE.md) | 2026-08-15 | Adopted | The in-engine Qwen vision tower for image input, gated by tower parity against the HF f32 oracle; video was left for a later lane. |

## KV cache, prefix cache and spill

| Record | Date | Verdict | Decides |
|---|---|---|---|
| [KV-PHYSICAL-RECLAIM.md](KV-PHYSICAL-RECLAIM.md) | 2026-09-20 | Superseded | VMM-backed KV planes for physical demote and reclaim as a gate-only door; pool trim rejected. Superseded for serving by KV-VMM-SERVING-DEFAULT. |
| [KV-VMM-SERVING-DEFAULT.md](KV-VMM-SERVING-DEFAULT.md) | 2026-11-04 | Adopted | On-demand VMM K/V planes are the serving default on the full-power RTX PRO 6000 Workstation Edition, pooled planes elsewhere; `MEMRA_KV_ALLOCATOR` deleted. |
| [KV-PARK-COMPACT-DEFAULT.md](KV-PARK-COMPACT-DEFAULT.md) | 2026-11-04 | Adopted | Continuation-pool park compaction is the default on the RTX 5090 class and the RTX PRO 6000 Workstation Edition, keyed by device name. |
| [PINNED-DESTINATIONS.md](PINNED-DESTINATIONS.md) | 2026-09-21 | Amended | Pinned host destinations are cached on the RTX PRO 6000 class and write-combined elsewhere; since 2026-11-04 this sets only the tier gates' transfer engine. |
| [PREFIX-CACHE-POLICY.md](PREFIX-CACHE-POLICY.md) | 2026-09-21 | Adopted | Plain LRU is the only prefix-cache eviction policy; the segmented (SLRU) arm and its two doors deleted. Re-tested 2026-11-07, unchanged. |
| [PREFIX-CACHE-DISCRETE.md](PREFIX-CACHE-DISCRETE.md) | 2026-11-07 | Adopted | A discrete card's derived prefix budget leaves the session peak; value-per-byte eviction, second-hit admission and four other levers rejected. |
| [GENERIC-SPILL-INTERFACE-V1.md](GENERIC-SPILL-INTERFACE-V1.md) | 2026-09-19 | Adopted | `memra-tier::contracts` is the only shared schema for spill control-plane contracts; no runtime default or hardware qualification. |

## Speculative decoding and sampling

| Record | Date | Verdict | Decides |
|---|---|---|---|
| [MTP-LEARNED-DEPTH-20260920.md](MTP-LEARNED-DEPTH-20260920.md) | 2026-09-20 | Rejected | An online learned MTP depth policy is not promoted: Qwen lost 2.45% to fixed K=3, Gemma gained 0.43% over native adaptation; existing policies kept. |
| [SPEC-K-COLD-LONG-K2.md](SPEC-K-COLD-LONG-K2.md) | 2026-11-09 | Adopted | Cold prompts of 1,024+ tokens speculate at K=2 on the RTX 5090 Laptop GPU and the RTX PRO 6000 Max-Q; GB10 and every other card keep K=3. |
| [SPEC-K-COLD-SHORT-5090.md](SPEC-K-COLD-SHORT-5090.md) | 2026-11-10 | Rejected | K=2 for cold short prompts on the RTX 5090 Laptop GPU; the rung stays at K=3 because the faster depth follows acceptance. |
| [SPEC-ENTRY-YIELD-BOUND.md](SPEC-ENTRY-YIELD-BOUND.md) | 2026-11-10 | Adopted | A speculating session yields at a burst's entry with nothing emitted at most once in a row, then runs one round (memra#1201). |
| [SPEC-STALL-MARK.md](SPEC-STALL-MARK.md) | 2026-11-11 | Adopted | When a speculating session's last four burst entries each met a waiting request, the cost model marks that concurrency plain for 64 s (memra#1211). |
| [SAMPLED-VERIFY-POSITION-KEYED.md](SAMPLED-VERIFY-POSITION-KEYED.md) | 2026-11-10 | Adopted | The MTP route verifies a sampled round by position (one Philox event per generated position); the Leviathan/Chen rejection walk rejected (memra#1154). |
| [SAMPLED-PENDING-CARRY.md](SAMPLED-PENDING-CARRY.md) | 2026-11-10 | Adopted | A sampled MTP burst carries its pending token into the next burst like a greedy one; GB10 sampled decode +8.4%, greedy flat (memra#1186). |
| [DSV4-SAMPLER-RADIX-DEFAULT.md](DSV4-SAMPLER-RADIX-DEFAULT.md) | 2026-09-07 | Adopted | The DSv4 sampler orders candidates by radix when `MEMRA_DSV4_SAMPLE_SORT` is unset; `comparison` stays the explicit oracle arm. |
| [GB10-DEVICE-PENALTY-DEFAULT.md](GB10-DEVICE-PENALTY-DEFAULT.md) | 2026-11-10 | Adopted | The device penalty arm is on for the dense and MoE Qwen3.5 classes on GB10, keyed by device name; every other family stays off there. |
| [FA-VERIFY-ROWS.md](FA-VERIFY-ROWS.md) | 2026-11-10 | Adopted | A speculative verify reads the KV once through the rows walk from a measured depth floor per GQA shape, on every device, with no flag. |

## Prefill, attention and GEMM kernels

| Record | Date | Verdict | Decides |
|---|---|---|---|
| [GB10-F8F4-GEMM-V2.md](GB10-F8F4-GEMM-V2.md) | 2026-11-06 | Amended | GB10 runs F8F4 prefill GEMMs of 2,048+ rows on the v2 tile; GB10-F8F4-GEMM-V3 replaced its mainloop on 2026-11-08. |
| [GB10-F8F4-GEMM-V3.md](GB10-F8F4-GEMM-V3.md) | 2026-11-08 | Adopted | The v2 tile's mainloop is fed by TMA under mbarriers on GB10; `MEMRA_MMQ_F8F4_V3=0` is the rollback; discrete cards keep v1. |
| [GB10-PRIME-ATTENTION-H2.md](GB10-PRIME-ATTENTION-H2.md) | 2026-11-06 | Superseded | GB10 primes quantized-KV attention with the two-head (h2) twin. Superseded on GB10 by GB10-PRIME-ATTENTION-H2P, which keeps h2's numbers. |
| [GB10-PRIME-ATTENTION-H2P.md](GB10-PRIME-ATTENTION-H2P.md) | 2026-11-07 | Adopted | GB10 runs h2's row program on a three-stage mbarrier K/V ring, bit-identical to h2; probe 68 to 85 TFLOPS at depth 131,072. |
| [GB10-PRIME-GDN-GLUE.md](GB10-PRIME-GDN-GLUE.md) | 2026-11-07 | Adopted | GB10 prime: GDN compact layout, fused K4+K5, fused norms with activation quantize, gate and up in one launch; bit-identical, prime bench 32.19 to 25.70 s. |
| [GB10-PRIME-GLUE-2.md](GB10-PRIME-GLUE-2.md) | 2026-11-07 | Adopted | GB10 prime, second pass: residual add in the GEMM epilogue, K2 at C = 32, fused conv + L2, fused attention q path; all bit-identical. |
| [GB10-PRIME-CHUNK-8192.md](GB10-PRIME-CHUNK-8192.md) | 2026-11-06 | Rejected | An 8,192-row GB10 prime chunk: exact and 0.64% faster alone, but it doubles the prime walker's peer quantum; 4,096 stays. |
| [PRIME-TAIL-FOLD.md](PRIME-TAIL-FOLD.md) | 2026-11-09 | Adopted | A cold prime with one stop for a cache capture or turn checkpoint runs as one walk, the capture tapped inside the call; bit-identical to the split. |
| [FA-SPLIT-DEEP-RUNG.md](FA-SPLIT-DEEP-RUNG.md) | 2026-11-10 | Adopted | GB10's FA-decode split ladder takes 512 keys a split past 16,384 keys with 4 KV heads and past 32,768 with 2; other parts keep today's ladder. |
| [QWEN-PRIME-FA2-DEFAULT.md](QWEN-PRIME-FA2-DEFAULT.md) | 2026-11-04 | Adopted | The Qwen 24 Q / 4 KV / d256 prime takes the FA2 attention class on the 170-SM RTX 5090 at prime chunk 1024; `MEMRA_PRIME_ATTN_FA2` deleted. |
| [GEMMA-FA-ONE-PROGRAM.md](GEMMA-FA-ONE-PROGRAM.md) | 2026-11-04 | Adopted | Gemma 4 SWA prefill runs one windowed attention program at every prompt length; `MEMRA_GEMMA_FA_ONE_PROGRAM` deleted. |
| [MOE-PREFILL-PLACEMENT-INDEPENDENT.md](MOE-PREFILL-PLACEMENT-INDEPENDENT.md) | 2026-11-11 | Adopted | A softmax-routed MoE prefills with the resident pairs program on both expert placements; SLRU reads the same bytes from cache slots (memra#1217). |
| [GEMMA-MOE-PLACEMENT-INDEPENDENT.md](GEMMA-MOE-PLACEMENT-INDEPENDENT.md) | 2026-11-11 | Adopted | A gemma4_moe model runs the resident MoE program on both expert placements, decided once per boot (memra#1225). |
| [GEMMA-KQUANT-EXPERT-DISPATCH.md](GEMMA-KQUANT-EXPERT-DISPATCH.md) | 2026-11-11 | Adopted | Every gemma4_moe device arm picks its expert kernel per projection qtype, so a k-quant bank (Q3_K, Q4_K, Q6_K) runs a kernel that decodes it (memra#1234). |
| [MOE-INTOKEN-PREFETCH-DEFAULT.md](MOE-INTOKEN-PREFETCH-DEFAULT.md) | 2026-11-04 | Adopted | The legacy MoE slot cache prefetches the next routed expert in-token on the RTX 5090 Laptop GPU and the RTX PRO 6000 Workstation Edition. |
| [DSV4-HC24-SPLIT-DOTS.md](DSV4-HC24-SPLIT-DOTS.md) | 2026-09-09 | Adopted | DSv4's HC-24 pre-attention and pre-FFN dots run as split dots with 16 slices by default; sequential dots, S8 and S32 rejected as the default and kept as explicit classes. |
| [DSV4-MATRIX-MOE-PROGRAM.md](DSV4-MATRIX-MOE-PROGRAM.md) | 2026-09-11 | Adopted | DeepSeek-V4 loads the matrix expert program for prefill, decode and verify (memra #461); the scalar reference program stays only as the class oracle. |
| [ORNITH-PAIR-OWNER.md](ORNITH-PAIR-OWNER.md) | 2026-08-25 | Rejected | Source-verbatim pair-owner ordering for Ornith's NVFP4 MoE gate/up path (4.5% to 6.8% slower at B=16); receipts and the exact candidate banked. |

## Multi-card placement

| Record | Date | Verdict | Decides |
|---|---|---|---|
| [PRO6000-MULTICARD.md](PRO6000-MULTICARD.md) | undated | Adopted | `MEMRA_PARALLEL=auto` (opt-in) plans one model over 2-4 RTX PRO 6000 cards: contiguous PP for dense, whole-expert EP for a routed MoE when every rank keeps its reserve, else PP or a refusal. |
| [DSV4-TPEP-DEFAULT.md](DSV4-TPEP-DEFAULT.md) | 2026-09-25 | Amended | DSv4-Flash on two cards loads TP/EP by default; since 2026-11-04 PP-2 is the explicit `pp` placement, not a rollback seam. |
| [TP-PEER-PULL-DEFAULT.md](TP-PEER-PULL-DEFAULT.md) | 2026-11-04 | Amended | An armed TP walk with no transport set uses peer pull on sm_100a builds and host-canonical elsewhere; the glm5 symmetric posture takes device-push. |
| [GLM5-TP-DEVICE-SAMPLER-DEFAULT.md](GLM5-TP-DEVICE-SAMPLER-DEFAULT.md) | 2026-11-04 | Adopted | The GLM TP-2 plain route samples on the device on sm_100a builds; the host sampler stays everywhere else; `MEMRA_GLM5_TP_DEVICE_SAMPLE` deleted. |
| [GLM5-B200-DOORS.md](GLM5-B200-DOORS.md) | 2026-11-04 | Adopted | The GLM-5.3-Flash sm_100a doors decided from their receipts: which became defaults, which were deleted, which calls stay open. B200 is best effort. |

## Early engine plans

| Record | Date | Verdict | Decides |
|---|---|---|---|
| [BEST-OF-ALL-WORLDS.md](BEST-OF-ALL-WORLDS.md) | 2026-06-26 | Plan | Which engine each component's best implementation is taken from, and memra's own edges. Its single RTX 5090 scope is outgrown. |
| [PHASE1-HYBRID.md](PHASE1-HYBRID.md) | undated | Plan | The phase-1 build plan for a logit-matched Qwen3.5-9B hybrid forward in Rust and cudarc on sm_120: module layout, per-layer graphs, staged kernels. |

## Adding a record

A decision that changes a default, a format, a target or an arm gets a record here, with the
measurement that settled it, and a row in this index. A superseded record keeps its numbers and
gets a banner naming what replaced it; it is never deleted. See `AGENTS.md`, "Measurements and
decisions are a corpus".
