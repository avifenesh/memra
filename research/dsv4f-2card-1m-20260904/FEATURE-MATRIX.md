# DeepSeek-V4-Flash-0731 on 2x RTX PRO 6000

Research refreshed 2026-09-04. Tracking: memra #4. This is an implementation
decision record, not a production-support claim.

## Artifact decision

The target is an exact-source sharded **Safetensors** artifact, never GGUF:

1. source checkpoint revision `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`;
2. streamed expert-only NVFP4 mint, preserving dense FP8 tensors and every
   non-expert byte;
3. immutable house artifact revision plus per-file sha256 manifest;
4. Memra's native Safetensors loader and serialized DSV4 ModelPlan.

The house artifact contract is intentionally mixed. Trunk routed experts are
NVFP4 with `weight_scale` and `weight_scale_2`; the bundled `mtp.*` DSpark
experts remain source MXFP4 with their single E8M0 `scale`. There is no invented
`input_scale` tensor. The ModelPlan must describe these bytes exactly and refuse
any missing or extra quantization auxiliary.

## Model feature census

Primary configuration:
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json

| feature | exact 0731 shape | serving consequence |
| --- | --- | --- |
| scale | 284B total, about 13B active/token | weight bandwidth and MoE dispatch dominate decode |
| context | 1,048,576 trained positions, YaRN from 65,536 | 1M is a real model limit, not an extrapolated rope setting |
| trunk | 43 layers, hidden 4096, mHC multiplier 4 | four residual streams; boundary transport carries the whole mHC state |
| local attention | 128-token uncompressed sliding window | small fixed ring per layer preserves exact local detail |
| CSA | overlapping 4:1 compressor plus learned 64-head, 128-dim indexer; top-k 512 | compact long history plus bounded sparse attention; indexer scan is the long-context hot path |
| HCA | non-overlapping 128:1 compressor, full attention over compressed entries | at 1M only about 8,192 long-range entries per HCA layer |
| shared K/V | one 512-dim K=V head (MQA) | substantially less state than conventional per-head K/V |
| position/output | partial RoPE, learned sinks, grouped low-rank output projection | these are semantic operations, not optional inference approximations |
| MoE | 256 routed experts, top 6 plus one shared expert; first 3 hash layers | routing exactness and native FP4 kernels are load-bearing |
| activation/numerics | mHC Sinkhorn, sqrt-softplus routing, SwiGLU limit 10 | f32 islands and accumulation order are checkpoint semantics |
| DSpark | bundled 3-block drafter, block size 5, target layers 40/41/42, Markov rank 256 | source-native first speculative candidate; no separate model path |

Hugging Face's current implementation documentation is the clearest independent
semantic cross-check for CSA/HCA, shared K=V, partial RoPE, sinks and grouped
output projection:
https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/deepseek_v4.md

## Why 1M fits this pair

The exact two-card capacity gate at `MEMRA_CTX=1048576` measured:

| stage | post-load allocation | compact cache | chunk-32 verify workspace | post-allocation total | remaining headroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| device 0 | 83.487 GiB | 7.044 GiB | 0.233 GiB | 90.769 / 94.970 GiB | 4.201 GiB |
| device 1 | 83.581 GiB | 6.418 GiB | 0.232 GiB | 90.237 / 94.970 GiB | 4.733 GiB |

That is the complete active semantic state in the current f32 implementation,
including CSA/HCA stores, indexer stores, SWA rings and pending compressor state.
It is not a conventional token-wise full-attention KV allocation. The configured
1M model limit therefore binds before steady-state cache capacity on one active
session. The DSpark-aware split reserves 10.83 GiB on the tail stage and moves
the cut from layer 22 to 23. The full compact state, DSpark and chunk-32 transient
workspace fit simultaneously. Exact hierarchical top-512 selection also passes
against a host oracle at 250,003 candidates, the 4:1 compressed-index scale for
one million tokens. See `capacity-1m-streamtopk-5a3820ec9.md`. This proves
allocation and exact-selection reach, not completed 1M prefill throughput.

Charging every request for a 1M-capacity cache would still destroy concurrency.
This lane therefore adds per-session capacity planning: device cache capacity is
`prompt + admitted output`, bounded by the model-wide 1M rope/kernel plan. A 1M
request can reserve the full compact state while ordinary requests reserve only
their actual envelope.

## Cache architecture selected for Memra

```text
active session
  compact SWA + CSA + HCA + indexer state on its owning GPU
        |
        | pause/retire once, D2H live rows only
        v
pinned host LRU, keyed by (model thread, PC-ISO namespace, exact token prefix)
        |
        | exact strict-prefix hit, H2D once, feed non-empty suffix
        v
fresh capacity-planned device state + fresh scratch
```

The host tier does **not** stream selected KV rows over PCIe for every decode
token. An active 1M state already fits and per-token host service would spend
PCIe bandwidth in the latency-critical loop. Host RAM is cheap here because the
compact state moves only at a conversation pause/resume boundary and remains
compressed; dead capacity tails, verify transients and step workspaces never
cross PCIe.

DSpark state is not reconstructed approximately. A parked speculative session
also carries its three persistent 128-token rings and newest trunk tap. Transient
draft rows are scratch. Restored sampling stays keyed to the same absolute token
positions as cold sampling.

An L3 file/direct-I/O tier is a follow-up only after pinned L2 copy wall and pause
frequency are measured. It may serialize the opaque host image with an artifact,
plan and layout hash. It must never become an unversioned raw dump.

## Current engine comparison

| engine | RTX PRO 6000 / 2-card | 1M/cache posture | speculative posture | current consequence |
| --- | --- | --- | --- | --- |
| Memra | native PP2, native SM120 NVFP4 expert kernels, exact mixed ST artifact | compact model-native state; capacity-planned sessions; pinned parked tier in this lane | bundled DSpark works on device; DFlash2 remains a separately measured candidate | only engine in this comparison whose two-card host tier is designed around the exact DSV4 semantic planes |
| SGLang | official cookbook lists Flash TP2 on RTX PRO 6000 | cookbook says HiCache and MegaMoE are unsupported on RTX PRO 6000; SM120 HiSparse/HiCache issues remain | bundled DSpark is the official 0731 route, but an open SM120 top-k=192 issue reports startup failure without a padding fix | strongest external serving control, but its documented RTX PRO path lacks the cache tier requested here |
| vLLM | current official recipe targets larger 96 GB GPU counts; community SM120 reports exist | hybrid cache manager and FP8 cache, but no source-backed two-card 1M recipe located | EAGLE/DSpark support is evolving | use as an offline behavioral/perf control, never a Memra serving fallback |
| TensorRT-LLM | DSV4 heterogeneous cache manager is architecture-aligned | GPU/host cache machinery exists, hardware recipe not specific to this pair | speculative support varies by release | useful design control; not an external dependency |
| llama.cpp | experimental local path | no serving-grade model-native 1M cache receipt found | open DSV4/DSpark long-generation/leak reports | not a correctness or performance baseline for this target |

Current SGLang cookbook:
https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx

Relevant current SM120 evidence:

- sparse prefill gap: https://github.com/sgl-project/sglang/issues/31578
- DSpark top-k=192 failure and measured patched control:
  https://github.com/sgl-project/sglang/issues/33985
- HiSparse multi-long-request failure:
  https://github.com/sgl-project/sglang/issues/26427
- HiCache/breakable graph integration issues:
  https://github.com/sgl-project/sglang/issues/25526

## Public beat targets

The strongest reproducible two-card ledger found in the 2026-09-04 sweep is:
https://github.com/Infatoshi/dsv4-flash-2x-rtxpro6000s/blob/main/docs/BENCHMARKS.md
with raw rows in:
https://github.com/Infatoshi/dsv4-flash-2x-rtxpro6000s/blob/main/bench/scoreboard.jsonl
and
https://github.com/Infatoshi/dsv4-flash-2x-rtxpro6000s/blob/main/bench/sweep_results.jsonl

Its contract is localhost, two RTX PRO 6000 Workstation cards, fixed prompt,
greedy/ignore-EOS, 256 generated tokens, usage-block token counts, one warm-up and
five measured reps. Memra must first reproduce that shape, then run the vendor-default
sampled twin separately. A vendor-sampled result is never compared numerically with a
greedy public row.

| matched cell | public result to beat | Memra requirement |
| --- | ---: | --- |
| 486-token prompt, c1, plain | 109.48 tok/s | median above 109.48, x5 interleaved with the selected spec arm |
| 486-token prompt, c1, DSpark K3 | 217.08 tok/s | median above 217.08 and exact spec==plain greedy bytes |
| 486-token prompt, c16, plain | 577.52 aggregate tok/s | aggregate above 577.52, no FIFO accounting trick, every stream complete |
| 486-token prompt, c16, DSpark K3 | 700.14 aggregate tok/s | aggregate above 700.14 with per-request acceptance and fairness receipts |
| 2,044-token prompt, c1, DSpark K3 | 168.09 tok/s | above 168.09 |
| 8,188-token prompt, c1, plain / DSpark | 69.68 / 103.03 tok/s | beat both on matched arms |
| 36k warm shared prefix, DSpark | 247 tok/s c1; about 825 aggregate c8 | beat after exact host/prefix restore, reporting cache transfer separately |
| 504,381-token cold prompt | 116 s TTFT (4,339 input tok/s), 66 decode tok/s at 500k | TTFT below 116 s and depth-matched decode above 66 |

Independent corroboration gives two additional targets:

- A 380 W two-card vLLM run reports 6.8k prefill at 4,167-4,209 tok/s,
  TTFT 1.63 s and 149-170 tok/s cold decode; warm-context decode reaches 331
  tok/s. Source:
  https://github.com/lastloop-ai/vllm-blackwell-guide/blob/main/docs/DEEPSEEK-V4-FLASH.md
- A current two-card community ledger reports 1M-token prefill at 2,464 tok/s,
  218.6 decode tok/s and a 2.8M-token KV pool. It lacks enough row-level protocol
  detail to become the primary claim baseline, but Memra should exceed all three
  before calling the result state of the art. Source:
  https://github.com/local-inference-lab/rtx6kpro/blob/master/daily-summaries/2026-07/2026-07-16.md

A TP4 public row reaches 1,047,552-token TTFT 6.46 s and 10.4 decode tok/s at
that depth with a split-indexer kernel. It is not a two-card comparison, but it
proves the algorithmic TTFT ceiling is far below the two-card community's
hundreds-of-seconds path and makes indexer materialization the main Memra prefill
target. Source:
https://github.com/ambientlight/rtx-pro-6000-bench/blob/main/docs/DEPLOY-MXFP4-W4A4-DEEPSEEK-V4-FLASH-SM120.md

Every final table must print TTFT, prefill tok/s, decode-only tok/s, total-wall
tok/s, prompt/output tokens, context depth, sampling, cache state, power cap,
GPU clocks, artifact revision, Memra SHA and engagement receipts. "Beats public"
is allowed only where the card count and row contract actually match.

The first Memra plain baseline on the target pair is recorded in
`public-baseline-6408a3f3a.jsonl`: 44.8966 tok/s mean and 44.8977 median over
five cold-cache repetitions of the public fixed 24-token prompt and 256-token
greedy/ignore-EOS output at the provider-enforced 500 W cap. This is below the
public fixed-prompt 109.3-111.45 tok/s rows. A real-source long-prefill probe also
hit the 120-second server deadline before returning; neither row is publishable
performance, and both make decode fusion plus chunked prefill active blockers.
The power cap is receipt metadata only: no power-limited diagnosis or 600 W
projection is permitted unless phase-level profiling shows that the measured
critical path is limited by power or clocks.

## Drafter decision

1. **DSpark first.** It is bundled in the exact 0731 checkpoint, trained against
   this trunk and already semantics-gated in Memra.
2. Sweep full-depth, fixed K=1..5 and confidence-slot thresholds on real prompts.
   Acceptance is diagnostic; the decision metric is served wall and decode tok/s.
3. **DFlash2 is not assumed better.** Admit it only from a revision-pinned,
   architecture-compatible trained checkpoint with a strict tensor census and
   tokenizer/template identity. Compare at matched verify width so draft-source
   cost is separated from verify-loop cost.
4. If no external DFlash2 checkpoint satisfies that contract, the most recommended
   version is bundled DSpark with the measured confidence-window knee, not an
   invented or cross-family drafter.

## Remaining implementation gates

- [x] exact mixed NVFP4/MXFP4 Safetensors ModelPlan
- [x] device-path DSpark and position-keyed sampled driver
- [x] capacity-planned device cache allocation
- [x] exact 1M allocation reach with DSpark and chunk-32 workspace on the target
  pair (`capacity-1m-streamtopk-5a3820ec9.md`; 4.201/4.733 GiB remains free)
- [x] bounded exact hierarchical top-512 selection through 250,003 candidates,
  including deliberate ties against the host ordering oracle
- [x] live-state pinned-host snapshot/restore implementation
- [x] strict PC-ISO token-prefix server wiring, default OFF
- [x] engine gate: cold vs restore cache classes bit-for-bit, plain and DSpark
  (`host-cache-gate-6408a3f3a.log`: exact target pair, prompt 32 + 17 warm,
  capacity 61 -> 158, future continuation 11; plain host 24,123,392 bytes,
  trunk+DSpark 24,958,976 bytes; all proposal ids/confidence bits, rings,
  cache classes and future logits identical)
- [x] served cold-vs-restored greedy identity gate, plain and DSpark
  (`host-cache-serve-gate-19e74601b.json`; 160 cached + 11 suffix; response
  identity; DSpark 49/69 accepted; no server fault)
- [x] vendor-default eight-turn sampled serving/cache/spec engagement
  (`vendor-default-8turn-19e74601b.jsonl`; no sampling parameters; turns 2-8
  restore 418..693 tokens and DSpark engages on all turns)
- [x] fixed-seed sampled cache transparency: the original monolithic cold path
  diverged at turn 2 (`sampled-seeded-8turn-19e74601b.jsonl`); chunked prefill
  now gives identical outputs and DSpark telemetry over all eight warm/cold turns
  (`adaptive-prime-cache-22c618b1b.md`; 418..691 restored tokens on turns 2..8)
- [x] bounded device chunk prefill through width 64, with width 1 vs 64 exact
  cache/logit/DSpark equality and monolithic teacher forcing 15/16 agree plus
  one 0.290964-margin in-band near-tie, zero out-of-band
- [x] fixed-seed sampled cache transparency on chunked prefill: eight warm/cold
  turns produce identical output hashes and identical DSpark telemetry
- [x] adaptive short-prime isolation: prompts through the configured chunk width
  retain the canonical monolithic path and are not parked; longer cacheable
  prompts retain the transparency-gated chunked path
- [x] grouped ModelOpt f16 prefill arm priced and rejected
  (`grouped-modelopt-prefill-reject-850436a9c.md`): zero-copy split-plane mapping
  was exact and frozen-source TTFT fell to 46.49 s, but teacher forcing was only
  7/16 agreement and fixed-seed cache transparency failed because cold prefill
  and warm exact-decode history inhabit different numerical classes; product
  path fully reverted
- [x] DSpark-only fused selected-expert dispatch is component-bit-exact and
  proposal-identical (`dspark-fused-moe-gate.md`); 1.171x proposal speedup but
  only 1.010x whole-loop, so it remains explicit/default-off rather than a
  selected serving claim
- [x] exact expert-major prefill slot ordering priced and rejected
  (`expert-slot-sort-reject-989feb991.md`): cache/logit bit identity passed,
  but 160-token median was 1.768474 s reference versus 1.769360 s sorted;
  product path fully reverted
- [x] batched prefill indexer selection: scalar T<=8 preserved; wide transactions
  collapse per-position score/top-k/index launches and improve the 9,952-token
  TTFT 83.36 -> 75.65 seconds (9.25%)
- [ ] long-prefill performance remains a blocker: the selected frozen row is only
  131.48 prompt tok/s, far below the public 4,339 prompt tok/s control
  (`chunk-prefill-sweep-20260905.jsonl`, `chunk32-long-nsys-6c604f9bf.md`)
- [ ] two-card execution topology remains a blocker: the long trace measured
  33.633/35.301 s kernel-busy per card but only 0.451 microseconds of overlap;
  the current PP2 request is effectively serial and needs a same-layer TP/EP or
  genuine cross-request pipeline schedule
- [x] same-layer topology feasibility priced
  (`two-card-expert-topology-probe-b194bcf57.md`): expert-ID EP preserves all
  contribution bits and gives 1.28-1.37x integrated selected-expert speed;
  balanced intermediate TP drifts only 1.2e-7 absolute but is flat at 1.007x,
  so do not build the current-kernel loader around that split
- [x] exact per-layer MoE CUDA graph priced and rejected
  (`moe-graph-reject-e8bf83af3.md`): logits/cache bits passed, but a
  1,025-token median regressed 7.296527 -> 7.330332 s; full layer/round graph
  coverage is required rather than fourteen-kernel segment graphs
- [ ] chunked prefill at 256K, 512K and 1M without transient OOM
- [ ] resumable multi-session scheduler and cross-session batch decode
- [ ] c1/c2/c4/c8/c16 throughput, fairness and admission cells
- [ ] plain vs DSpark K/VT sweep; DFlash2 only if its artifact contract passes
- [ ] full 1M prompt plus output, then concurrent shorter sessions while 1M is parked
- [ ] exact artifact/revision/plan/binary receipt bundle and `NativeQualified`
- [ ] optimized rewrite receipts on the winning posture and `NativeTuned`

No serving or speed claim is earned until every applicable gate above is backed by
the exact two-card artifact/plan/binary tuple.
