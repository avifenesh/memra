# Gemma 4 31B

| | Recommended use |
|---|---|
| **Status** | GGUF path supported and tuned; safetensors tuning remains separate |
| **Best starting path** | QAT Q4_0 GGUF with the Gemma assistant drafter |
| **Hardware** | RTX PRO 6000 Blackwell for the documented full serving configuration |
| **Input** | Text; optional image input through the qualified Gemma vision seam |
| **Use this when** | You want the dense Gemma 4 path, including the documented vision configuration |

Use the [Gemma 4 31B cookbook](../COOKBOOK.md#gemma-4-31b). Do not substitute the Qwen vision
flags; the cookbook names the Gemma-specific path.
