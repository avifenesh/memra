# Step-3.7-Flash 196B-A11B

StepFun's 196B-A11B mixture-of-experts on two or more cards. The cookbook route is the IQ4_XS GGUF
with its Q8_0 MTP head as a two-card pipeline (PP-2); image input, structured output and graphed
draft chains were gated on the NVFP4 safetensors checkpoint.

| | |
|---|---|
| **Serving route** | IQ4_XS GGUF plus the Q8_0 MTP head, 2x RTX PRO 6000 Blackwell, PP-2. Last gated 2026-09-20 at `fd7ce385b7` on a Server Edition pair: argmax MATCH for prefill/decode and batched-prime/tokenwise, `run-spec` K=1..8 PASS, and `kernel-check` with the `tools/kernel-check-step35.cells` cells against this artifact ALL GREEN (98 cells, 21 skipped). Standing gate: none. The pre-push hook takes an extra hash-bound `step-pro` gate (`MEMRA_HARDWARE_GATE=step-pro` with `MEMRA_STEP_PRO_RECEIPT`) that checks a receipt for the official FP8 checkpoint on RTX PRO 6000; it adds to the push qualification, it does not replace it |
| **Pack** | `step35`, support `None`: no pack state; the routes run the hand-written Step path |
| **Artifacts** | GGUF `stepfun-ai/Step-3.7-Flash-GGUF@0b69336d2fd2adfdef9c66e425f7778196c31482` (IQ4_XS, 3 shards, plus `Step3.7-flash-mtp-Q8_0.gguf`; 108,700,839,040 bytes). NVFP4 `stepfun-ai/Step-3.7-Flash-NVFP4@4275532ffd9a9496ff36b7a2dc4a9db1048da438`. FP8 `stepfun-ai/Step-3.7-Flash-FP8@b3d7916fccac844cca050d7520f2aaa513f9a84f` |
| **Drafters** | the published external Q8_0 MTP head (`Step3.7-flash-mtp-Q8_0.gguf`; the GGUF trunk itself reads `nextn=0`). On PP-2 it does not draft |
| **Hardware measured** | 2x and 3x RTX PRO 6000 Blackwell |
| **Input** | text; image on the NVFP4 artifact (`MEMRA_STEP_VISION_DIR`, default off); no video |

## Run it

Commands are in the [cookbook](../COOKBOOK.md#step-37-flash-196b-a11b). Boot proof for the split:
`[pp] cross-device transport: stage0=dev0 stage1=dev1`. PP topology and admission are in
[serving](../SERVING.md#pipeline-parallel-pp-2-serving).

- On PP-2 the spec gate defaults to LOW=0 / HIGH=1 on a sharded cross-device placement, so the
  request-conditioned K policy picks K=0: the pair runs plain batched decode in one numeric class
  at every batch width, and served bytes do not depend on load history.
- The PP-2 split adds zero deviation: every f32 logit of every step bit-compared, 0 differing bits
  across seven gate configurations (`research/pp2-batch-20260806/`,
  `research/pp2-spec-20260806/`, `research/pp2-hardening-20260806/`).

## Surfaces gated on the NVFP4 artifact

Each was gated on a 2x RTX PRO 6000 Blackwell pair on `stepfun-ai/Step-3.7-Flash-NVFP4@4275532ffd9a…`
in August 2026. None was gated on the IQ4_XS PP-2 route.

| Surface | What passed | Receipt |
|---|---|---|
| Image input | the checkpoint's perception_encoder tower (BF16 inside the NVFP4 artifact) runs in-engine behind `MEMRA_STEP_VISION_DIR`; per-token parity min-cos 1.000000 against a NumPy reference and the checkpoint's own torch code; four end-to-end probes; vision requests admit at K=0, text keeps spec and stays byte-identical to the seam-off boot | [`research/step37-vision-20260830/`](../../research/step37-vision-20260830/CENSUS.md) |
| Structured output | `response_format` serves post-think: the think phase runs unconstrained, the grammar clamps from the atomic `</think>` token (id 128799); generation that ends inside the reasoning channel returns a named 400, never an empty 200; `MEMRA_POSTTHINK_CEILING` (default off) forces the close | [`research/step37-postthink-grammar-20260830/`](../../research/step37-postthink-grammar-20260830/RESULTS.md) |
| Graphed draft chains | the 3-head step-modulo draft chain is CUDA-graph captured on the serving shape, greedy and filtered-sampling (temp 0.5, top_p 0.9) byte-identical to eager | [`research/step37-draft-graph-serving-20260830/`](../../research/step37-draft-graph-serving-20260830/RESULTS.md) |

## Measured

The 2026-09-20 record above is a pass/fail gate with no throughput claim. The official FP8
checkpoint also passed as PP-3 on three cards that day: verify-class argmax MATCH (max logit
difference 3.338e-5) and `run-spec` K=1..8 PASS
([record](../../research/modelplan-onboarding-537-20260919/native-20260920/RESULTS.md)).

## Limits

- PP-2 is pipeline parallelism, not tensor parallelism. `MEMRA_STEP_TP` (TP2/TP4 over the official
  FP8 trunk) and `MEMRA_STEP_EP` are correctness seams, not serving routes; see
  [FLAGS.md](../FLAGS.md).
- With reasoning on, natural think closes on real agentic prompts ran p50 2,119 and p90 3,554
  think tokens, so a small `max_tokens` on a constrained request reaches the named 400.
- RTX PRO 6000 Blackwell is the blocking target for Step multi-card evidence; RTX 5090
  evidence is a later compatibility follow-up.

## Deeper sources

- Lane index: the `step37-*` and `step-*` rows of [`research/INDEX.md`](../../research/INDEX.md);
  the last e4m3 lane is `research/step-e4m3-x8-20260814/`, later lanes target the NVFP4 artifact.
- Pack: `crates/memra-gguf/src/model_packs/step35/`.
- The earlier card and the notes under the board: [history](../history/MODELS.md#step-37-flash-196b-a11b-card-before-2026-11-12).
