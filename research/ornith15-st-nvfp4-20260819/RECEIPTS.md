# Ornith-1.5-35B-A3B NVFP4-MTP GGUF — mint + gates + publish receipts (2026-08-19)

Published: `Avifenesh/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF`, 2026-08-19T20:38:29Z — first NVFP4
quantization of this model in any format at publication time (checked ~20:00Z: official repos
BF16/Q4–Q8 GGUF + modelopt-NVFP4 safetensors only; community = imatrix K-quant GGUF, MLX wave,
two empty MXFP8 placeholders; zero NVFP4 GGUF). Model released 2026-08-18T06:24Z → published
~34.5h after drop. Raw logs in `gates/` (this dir).

## Artifact

| | |
|---|---|
| main | `Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf`, 20,188,038,400 B, sha256 `ff60a0c6443c24b5087f68170b12863a7b407a790d47db39498d009f79b91772` |
| source | official BF16 GGUF (`qwen35moe`, 41 blocks, `nextn_predict_layers=1`), lfs sha256 `a3ee48dd…3307591` |
| recipe | fork llama-quantize (branch `nvfp4-imatrix-scale-search`, tip `c730697`), ftype NVFP4, `--output-tensor-type q5_k --token-embedding-type q5_k`; result: blocks+MTP NVFP4, embd/output Q5_K, norms F32 — matches the Qwen3.6-27B house mint (checked per owner: q27 artifact = Q5_K embd/output). Deviation vs 9B precedent: `nextn.eh_proj` NVFP4 (9B had Q8_0) — acceptance measured with it; remint test STRUCK (owner 2026-08-20: head quant already measured better-or-equal on this pipeline, hqmtp receipt) |
| masked draft | `mtp-…-frspec-owngen32768.gguf` 945,170,176 B — `make-trimmed-draft.sh` hqmtp order (mask → NVFP4 head + Q4_K_M block), ranks from `frspec-owngen` built-in mixed pack, own generations only |
| ranks | `ornith15-ranks-owngen-32768.txt` (ST load-time trim via `MEMRA_FRSPEC_TRIM`) + `.gguf` |

## Gates (memra v0.94.0 = origin/main 43bd1afb84, one RTX PRO 6000, CUDA_VISIBLE_DEVICES=0)

| gate | result | log |
|---|---|---|
| kernel-check | GREEN | `gates/kernel-check.log` |
| run-spec K=1..8 self-consistency | PASS every K (spec ≡ plain, token-identical) | `gates/run-spec.log` |
| batched-prime ≡ tokenwise argmax | MATCH (all probes) | `gates/run-gen.log` |
| cross-oracle 48-tok greedy vs BF16 ST (transformers 5.8.1, CPU) | 2/3 probes token-identical; probe 2 diverges at tok 14 where official Q8_0 makes the IDENTICAL choice (near-tie, quant-typical) | `gates/oracle-cpu.json`, `gates/run-gen.log` |

## Spec economics measured (contended shared box — ratios valid, absolutes NOT bankable)

K=2 greedy 256-tok code gen, 3 dead-flat reps (`gates/ab-draft.log`): embedded head 53.7%
acceptance, masked 48.8%; both spec arms 0.48x vs plain decode (~100 vs ~208 tok/s in-window).
Short 32-tok probes: 9–28% acceptance (think-phase openings drown the head) — probe length
matters. Q8_0 control (`gates/run-spec-q8control.log`): 2–4% on raw-token probes → quant
exonerated; distribution + head strength are the drivers.

**Serving posture: spec-OFF default for this model** until acceptance improves. Known lever:
agentic-session ranks corpus (Q38's own-gen agentic corpus lost only ~2pt to the mask vs ~5pt
here on the generic pack). The masked head + hqmtp artifacts shipped with honest numbers per
research-results doctrine — a method-vs-method loss is a result.

## Serving-shape gates (2026-08-19/20, memra-server v0.94.0, GPU0, contended box — ratios only)

| gate | result | log |
|---|---|---|
| server bring-up (`MEMRA_MODELS`, 262k ctx advertised, prefix-cache on, admission calibrated) | PASS | `gates/serve.log` |
| chat completion, greedy — coherent, `<think>` split into `reasoning`/`reasoning_details`, clean content | PASS | `gates/serve-chat-long.json` |
| determinism ×2 (same request, byte-identical message) | IDENTICAL | `gates/serve-chat1.json` |
| tools round-trip (`<function=` XML dialect → OpenAI `tool_calls`, args JSON exact, finish_reason=tool_calls, zero content leak) | PASS | `gates/serve-tools1.json` |
| spec-on (naked default) vs spec-off serve A/B, 3×256-tok greedy | **spec-on 3.2x SLOWER** (11.26s vs 3.55s; server acceptance 24.3%, K=3, dead-flat reps) | `gates/serve-bench-spec*.log` |
| width cells c1/c4/c8, spec-off, 128-tok | 211 / 474 / **577 tok/s aggregate** (c8 ×2 flat) — direction-only, contended | `gates/widths.log` |

**Serving default for this model: plain decode (spec-off).** `serve_spec_enabled()` is global —
per-model default needs either the launch-recipe env or (cleaner) a runtime acceptance-based
demotion lane. Documented, not yet an engine change.

## Ranks-corpus hypothesis: REFUTED (2026-08-20)

v2 masked draft from 44 REAL agentic session prompts (sxc + hermes pools) vs v1 (generic
built-in pack) vs embedded, K=2, 256-tok, agentic+code probes, ×2 dead-flat
(`gates/ab-draft-v2.log`): embedded 50.4/53.7, v1 49.6/48.8, v2 48.1/49.6. Corpus flavor does
not move the mask cost at this scale; base acceptance ~50% + spec round cost is the binding
constraint. v2 NOT published (adds nothing over v1). Spec on this model is structurally
net-loss at current acceptance — method-vs-method result, recorded per doctrine.

## Real agentic-CLI round-trip + long-ctx (2026-08-20)

- **Codex gate PASS** (`gates/codex-roundtrip.log`): real `codex exec` 0.147.0 over memra's
  `/v1/responses` (wire_api=responses, custom provider, tunnel to the serve box) — codex issued
  the shell tool call, `zsh -lc 'echo ORNITH-TOOLS-GATE'` executed, marker returned through the
  tool result, model narrated the output. Genuine round-trip per the gemma-gate criteria
  (marker present on non-echo lines; no OOM/disconnect).
- **Long-ctx probe**: 21,619-token prompt → 96-token completion in 9.6 s end-to-end through
  the tunnel (contended card); usage accounting exact, coherent output. Full-depth 262k cell
  belongs to the sealed battery.

## Head diagnosis (owner, 2026-08-20)

Base acceptance (~50% long-gen, ~25% serve-level K=3) is the ceiling, not the mask: the mask
pays when head-read/compute is the bound, and here the **vendor MTP head looks undertrained
for the RL'd trunk** (Ornith-1.5's self-improvement loop moved the trunk; the 1-layer head
did not follow). Corpus-flavor refutation above is consistent. Follow-up lane candidate:
**continued-training of the MTP head** on trunk own-generation next-token pairs (1-layer head,
own-gen data, sft-pipeline machinery exists) — would move both spec arms; not started.

## Not claimed

Engine support row in MODELS.md, serving battery, tools round-trip gate, template byte-parity
fixture, vision path, sealed perf bank — all pending the quiet-window battery (§2 of PLAN.md).
This publication is an artifact + receipts, not a memra support claim.

## v2 head update published (2026-08-20T06:14Z)

The MTP-head training lane (`../ornith15-mtp-train-20260820/`) shipped: repo files replaced
in place (`Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf` sha `72ff9600…518fd3`, masked draft sha
`46f0dd4c…73fe899`) — vendor head continued-trained with depth-3 chain-rollout on own
generations. Serve-level K=3 same-window A/B: acceptance 0.352 -> 0.431 embedded, 0.393
masked; run-spec K=1..8 PASS on the reminted file; spec remains net-loss vs plain on this
model (84 vs 196 tok/s single-stream, contended card) so the spec-off posture stands.
HF xet dedup uploaded only ~475MB for the 20.2GB main file — byte-level confirmation that
the remint changed blk.40 only.
