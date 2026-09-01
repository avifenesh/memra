# Multimodal

| | Recommended use |
|---|---|
| **Qwen3.8 27B** | Text, image, and video on its qualified vision path |
| **Ornith 1.5 35B-A3B** | Text and image through the checkpoint's vision tower |
| **Gemma 4 31B** | Text with the documented Gemma-specific image configuration |
| **Read next** | The model card, [Cookbook](../COOKBOOK.md), and [Serving](../SERVING.md) |

Vision support is model-specific. Use the tower, projector, template, and modality gates named by
the model's own configuration; never substitute another family's vision flags because the model
still produces text.
