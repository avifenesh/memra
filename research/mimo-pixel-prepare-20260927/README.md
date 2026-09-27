# MiMo RGB8 pixel preparation

This lane adds decoded RGB8 `[T,H,W,3]` input to the bounded native
`patchify_prepared_frames` path. It uses the pinned
`preprocessor_config.json` (SHA-256
`b269e51bdc1c53ef7c82522984378513f462869cf83c4c75ec3f98bd92aa7972`)
and the Transformers 5.3 `Qwen2VLImageProcessor` source
`image_processing_qwen2_vl.py` (SHA-256
`218ff9374af6f9c3c3b0e0ca89d2c21308bd1043bf65c8386d658902caf88457`).

`oracle.json` records eight synthetic image and video cases. They cover no
resize, enlargement, reduction on one and both axes, the aspect-ratio edge,
the exact 256-patch cap, two frames, and an odd video tail. The
comparison reconstructs normalized CHW frames from the Python processor's
actual patch rows. With Transformers 5.3.0, Pillow 12.1.1, and NumPy 2.3.5,
all 463,872 F32 values match exactly. Maximum absolute error is 0 and maximum
relative error is 0, using `abs(native-source)/max(abs(source),1e-6)`.

Reproduce from the repository root after installing those Python packages:

```sh
CARGO_BUILD_JOBS=2 cargo build -p memra-reference --example mimo_pixel_prepare
python crates/memra-reference/tools/check_mimo_pixel_prepare.py \
  --source /tmp/mimo-transformers-5.3/transformers/models/qwen2_vl/image_processing_qwen2_vl.py \
  --config /tmp/mimo-current-source-20260927/preprocessor_config.json \
  --binary target/debug/examples/mimo_pixel_prepare
```

The component accepts uniform decoded RGB8 frame arrays. It does not handle
file decoding, channel conversion, video frame selection, timestamps, request
assembly, or model serving. It rejects geometry beyond the current 256-patch
projector cap and 32 temporal groups.
