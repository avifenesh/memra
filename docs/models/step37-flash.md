# Step 3.7 Flash 196B-A11B

| | Recommended use |
|---|---|
| **Status** | Supported on the qualified GGUF pipeline path — PP-2 serving, image input, structured output, and graphed draft chains; FP8 tuning is separate work |
| **Best starting path** | IQ4_XS trunk with the Q8_0 MTP head on two-card PP-2 |
| **Hardware** | 2× RTX PRO 6000 Blackwell |
| **Use this when** | The model does not fit one card and you want the qualified pipeline-parallel path |

Use the [Step cookbook](../COOKBOOK.md#step-37-flash-196b-a11b). This is PP-2, not tensor
parallelism. Topology and admission behavior are in [serving](../SERVING.md#pipeline-parallel-pp-2-serving).

Since the qualified pipeline landed, three more surfaces serve on the same endpoint:

- **Image input** (lane/step37-vision-20260830): the checkpoint's own perception_encoder
  tower runs in-engine behind `MEMRA_STEP_VISION_DIR` (default OFF; [flags](../FLAGS.md)),
  with the vendor's exact tiling law. Per-token parity-gated against two independent
  references before any serving path; text requests stay byte-identical to the seam-off
  boot and keep MTP spec engaged. Receipts: `research/step37-vision-20260830/`.
- **Structured output** (lane/step37-postthink-grammar): `response_format`
  `json_object`/`json_schema` served post-think — the think phase runs unconstrained as
  trained, the grammar engages from the tokenizer's `
</think>

` close token. Fail-closed
  named 400s if generation ends inside the reasoning channel. Receipts:
  `research/step37-postthink-grammar-20260830/`.
- **Draft-graph serving** (lane/step37-draft-graph-serving-20260830): the 3-head
  step-modulo draft chain is CUDA-graph captured on the serving shape, greedy and
  filtered-sampling arms byte-identical to eager. Receipts:
  `research/step37-draft-graph-serving-20260830/`.
