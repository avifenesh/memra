# Qwen3.8 27B

| | Recommended use |
|---|---|
| **Status** | Supported and tuned on both safetensors and GGUF paths |
| **5090 default** | DFlash2 on the NVFP4+Q5_K GGUF trunk with q4 weights and the agentic 32K FR-Spec rank map; the masked MTP head remains the rollback path |
| **Best starting path** | The qualified 5090 DFlash2 q4 + agentic-rank route; safetensors NVFP4 MTP remains the native-format fallback |
| **Portable path** | NVFP4+Q5_K GGUF with the embedded or attached MTP head |
| **Fastest spec route** | DFlash2 drafter on the GGUF trunk (`MEMRA_DSPARK_SPEC`, q4 drafter default + FR-Spec trim; sampled sessions stack in the LOW band since v0.113.0) — beats the MTP head on every rung of the vendor-default sampled shape on RTX PRO 6000 (c1 127/117, c2 128/120, c4 87/85 agg tok/s) and serves both production origins; +3.87% local 5090 E2E over full-head DFlash2 |
| **Hardware** | RTX 5090 for single-card use; RTX PRO 6000 Blackwell for larger context and serving headroom |
| **Use this when** | You want the main general-purpose and agentic path exercised by Memra |

Start with the [Qwen3.8 cookbook](../COOKBOOK.md#qwen38-27b). Choose the agentic, prose, or mixed
ranks file only after reading the [model detail](../MODELS.md#qwen38-27b-in-detail). Conditions and
receipts are in [performance](../PERFORMANCE.md).
