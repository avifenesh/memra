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

Memra's existing exact cache census at `MEMRA_CTX=1048576` is approximately:

| stage | compact cache allocation |
| --- | ---: |
| device 0 | 6.92 GiB |
| device 1 | 7.55 GiB |
| pair | 14.47 GiB |

That is the complete active semantic state in the current f32 implementation,
including CSA/HCA stores, indexer stores, SWA rings and pending compressor state.
It is not a conventional token-wise full-attention KV allocation. The configured
1M model limit therefore binds before steady-state cache capacity on one active
session. The remaining reachability risks are transient prefill workspace,
indexer materialization and the second card's weight/drafter headroom.

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
- [x] live-state pinned-host snapshot/restore implementation
- [x] strict PC-ISO token-prefix server wiring, default OFF
- [ ] engine gate: cold vs restore cache classes bit-for-bit, plain and DSpark
- [ ] served 8-turn sampled cache twin with real `n_cached` receipts
- [ ] chunked prefill at 256K, 512K and 1M without transient OOM
- [ ] resumable multi-session scheduler and cross-session batch decode
- [ ] c1/c2/c4/c8/c16 throughput, fairness and admission cells
- [ ] plain vs DSpark K/VT sweep; DFlash2 only if its artifact contract passes
- [ ] full 1M prompt plus output, then concurrent shorter sessions while 1M is parked
- [ ] exact artifact/revision/plan/binary receipt bundle and `NativeQualified`
- [ ] optimized rewrite receipts on the winning posture and `NativeTuned`

No serving or speed claim is earned until every applicable gate above is backed by
the exact two-card artifact/plan/binary tuple.
