# Ornith 1.5 35B-A3B

| | Recommended use |
|---|---|
| **Status** | Supported and tuned |
| **Best starting path** | NVFP4+Q5_K GGUF with the continued-trained MTP head and FR-Spec ranks trim |
| **Hardware** | RTX PRO 6000 Blackwell for the qualified serving configuration |
| **Input** | Text and image through the checkpoint's vision tower |
| **Use this when** | You want the current Ornith agentic and cached-long path |

Use the [Ornith 1.5 cookbook](../COOKBOOK.md#ornith-15-35b-a3b). Vision, trim, and cache behavior
remain separate gates; consult [models](../MODELS.md#ornith-15-35b-a3b-in-detail) and
[serving](../SERVING.md) before changing them.
