# Step 3.7 Flash 196B-A11B

| | Recommended use |
|---|---|
| **Status** | Supported on the qualified GGUF pipeline path; FP8 tuning is separate work |
| **Best starting path** | IQ4_XS trunk with the Q8_0 MTP head on two-card PP-2 |
| **Hardware** | 2× RTX PRO 6000 Blackwell |
| **Use this when** | The model does not fit one card and you want the qualified pipeline-parallel path |

Use the [Step cookbook](../COOKBOOK.md#step-37-flash-196b-a11b). This is PP-2, not tensor
parallelism. Topology and admission behavior are in [serving](../SERVING.md#pipeline-parallel-pp-2-serving).
