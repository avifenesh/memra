# Multimodal

For readers sending images or video alongside text. Vision is per model: each family has its own
tower, flags and parity gate.

## Request shape

OpenAI `image_url` (and, for Qwen3.8-27B, `video_url`) content parts. Data URIs are always accepted;
`http(s)` URLs are fetched only with `MEMRA_FETCH_URLS=1`. At most 8 images per request. Vision
tokens bill as ordinary prompt tokens.

## Recommended model and card

| Model | Input | Seam | Run it |
|---|---|---|---|
| [Qwen3.8-27B](../models/qwen38-27b.md) | image, video | `MEMRA_VISION_DIR` | [cookbook](../COOKBOOK.md#qwen38-27b) |
| [Ornith-1.5-35B-A3B](../models/ornith15-35b-a3b.md) | image | `MEMRA_VISION_DIR` | [cookbook](../COOKBOOK.md#ornith-15-35b-a3b) |
| [Gemma-4 31B](../models/gemma4-31b.md) | image | `MEMRA_GEMMA_VISION=1` + `MEMRA_GEMMA_MMPROJ` | [cookbook](../COOKBOOK.md#gemma-4-31b) |
| [Step-3.7-Flash 196B-A11B](../models/step37-flash.md) | image, on the NVFP4 artifact | `MEMRA_STEP_VISION_DIR` (default off) | [model card](../models/step37-flash.md) |
| [GLM-5.3-Flash](../models/glm53-flash.md) | image | on when the artifact carries `model.visual.*` | [model card](../models/glm53-flash.md) |

## Settings that matter

- One vision family per process: never substitute another family's vision flags.
- Gemma-4 31B image input runs with `NVIDIA_TF32_OVERRIDE=0`: its tower matched the reference only
  with TF32 off, and no code turns TF32 off for it ([cookbook](../COOKBOOK.md#gemma-4-31b)).
- `MEMRA_FETCH_URLS=1` with `MEMRA_FETCH_URLS_ALLOWED_HOSTS` for remote images; resolved addresses
  pass SSRF checks.
- A configured Qwen or Gemma tower that cannot load is a boot panic, not a silent text-only
  server.

Flag detail: [FLAGS.md](../FLAGS.md).

## How to validate

Each tower was gated by per-token cosine parity against its reference before serving; the model
card names the receipt. Send a real image through the endpoint and check the prompt-token count.

## Read next

The model card, [the cookbook](../COOKBOOK.md), [serving](../SERVING.md).
