"""Compare Memra RGB8 preparation with the pinned Qwen2VLImageProcessor source.

Usage:
  python check_mimo_pixel_prepare.py \
    --source /tmp/mimo-transformers-5.3/transformers/models/qwen2_vl/image_processing_qwen2_vl.py \
    --config /tmp/mimo-current-source-20260927/preprocessor_config.json \
    --binary target/debug/examples/mimo_pixel_prepare
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys


CONFIG_SHA256 = "b269e51bdc1c53ef7c82522984378513f462869cf83c4c75ec3f98bd92aa7972"
SOURCE_SHA256 = "218ff9374af6f9c3c3b0e0ca89d2c21308bd1043bf65c8386d658902caf88457"


def synthetic_frames(count, height, width, numpy):
    t, y, x = numpy.indices((count, height, width), dtype=numpy.int32)
    channels = (
        (17 * x + 29 * y + 11 * t + (x * y) % 37) % 256,
        (97 * x + 7 * y + 53 * t + (x ^ y)) % 256,
        (3 * x + 131 * y + 19 * t + (x * y) % 251) % 256,
    )
    return numpy.stack(channels, axis=-1).astype(numpy.uint8)


def unpack_patches(rows, grid, count, numpy):
    grid_t, grid_h, grid_w = (int(value) for value in grid)
    merged = rows.reshape(grid_t, grid_h // 2, grid_w // 2, 2, 2, 3, 2, 16, 16)
    frames = merged.transpose(0, 6, 5, 1, 3, 7, 2, 4, 8)
    return frames.reshape(grid_t * 2, 3, grid_h * 16, grid_w * 16)[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    args = parser.parse_args()

    if hashlib.sha256(args.config.read_bytes()).hexdigest() != CONFIG_SHA256:
        raise SystemExit("pinned preprocessor config SHA-256 differs")
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise SystemExit("pinned Transformers source SHA-256 differs")
    sys.path.insert(0, str(args.source.resolve().parents[3]))

    import numpy as np
    import PIL
    from PIL import Image
    import transformers
    from transformers.image_utils import ChannelDimension
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    if Path(inspect.getsourcefile(Qwen2VLImageProcessor)).resolve() != args.source.resolve():
        raise SystemExit("Qwen2VLImageProcessor did not load from the pinned source")
    if transformers.__version__ != "5.3.0":
        raise SystemExit("Transformers version differs from 5.3.0")
    config = json.loads(args.config.read_text())
    if (config["patch_size"], config["temporal_patch_size"], config["merge_size"]) != (16, 2, 2):
        raise SystemExit("pinned patch geometry differs")

    processor = Qwen2VLImageProcessor.from_pretrained(str(args.config.parent))
    if (
        not processor.do_resize
        or processor.resample != Image.Resampling.BICUBIC
        or not processor.do_rescale
        or processor.rescale_factor != 1 / 255
        or not processor.do_normalize
        or not processor.do_convert_rgb
    ):
        raise SystemExit("source processor defaults differ from the RGB8 preparation contract")
    cases = (
        ("image_unchanged", 1, 64, 96),
        ("image_upscale", 1, 33, 65),
        ("image_downscale", 1, 80, 128),
        ("image_downscale_both_at_cap", 1, 270, 270),
        ("image_aspect_edge", 1, 1, 200),
        ("video_even_resize", 2, 65, 97),
        ("video_odd_unchanged", 3, 64, 64),
        ("video_odd_resize", 3, 33, 65),
    )
    reports = []
    for name, count, height, width in cases:
        rgb8 = synthetic_frames(count, height, width, np)
        images = list(rgb8)
        rows, grid = processor._preprocess(
            images,
            do_resize=processor.do_resize,
            size=processor.size,
            resample=processor.resample,
            do_rescale=processor.do_rescale,
            rescale_factor=processor.rescale_factor,
            do_normalize=processor.do_normalize,
            image_mean=processor.image_mean,
            image_std=processor.image_std,
            patch_size=processor.patch_size,
            temporal_patch_size=processor.temporal_patch_size,
            merge_size=processor.merge_size,
            do_convert_rgb=processor.do_convert_rgb,
            data_format=ChannelDimension.FIRST,
            input_data_format=ChannelDimension.LAST,
        )
        reference = unpack_patches(rows, grid, count, np)
        result = subprocess.run(
            [str(args.binary.resolve()), str(count), str(height), str(width)],
            input=rgb8.tobytes(),
            capture_output=True,
            check=True,
        )
        shape = np.frombuffer(result.stdout[:16], dtype="<u4")
        native = np.frombuffer(result.stdout[16:], dtype="<f4")
        expected_shape = (count, 3, int(grid[1]) * 16, int(grid[2]) * 16)
        if tuple(shape) != expected_shape or native.size != reference.size:
            raise SystemExit(f"{name}: native shape or extent differs from source")
        native = native.reshape(expected_shape)
        difference = np.abs(native.astype(np.float64) - reference.astype(np.float64))
        relative = difference / np.maximum(np.abs(reference.astype(np.float64)), 1e-6)
        reports.append(
            {
                "case": name,
                "input_thwc": [count, height, width, 3],
                "output_tchw": list(expected_shape),
                "max_abs": float(difference.max()),
                "max_relative_floor_1e-6": float(relative.max()),
                "differing_elements": int(np.count_nonzero(difference)),
                "elements": int(difference.size),
            }
        )

    print(
        json.dumps(
            {
                "source_sha256": SOURCE_SHA256,
                "config_sha256": CONFIG_SHA256,
                "transformers": transformers.__version__,
                "pillow": PIL.__version__,
                "numpy": np.__version__,
                "cases": reports,
            },
            indent=2,
        )
    )
    if any(case["max_abs"] > 1e-6 for case in reports):
        raise SystemExit("native preparation differs from the pinned source beyond 1e-6")


if __name__ == "__main__":
    main()
