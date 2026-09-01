#!/usr/bin/env python3
# GLM-5.3-Flash vision tower upstream fixture (lane/glm5-vision, 2026-08-30).
#
# External implementation (transformers 5.16.1, torch CPU) used OFF-SERVING to create
# pinned oracle evidence only, per the engine-consumption law. Pins:
#   model: zai-org/GLM-5.3-Flash @ 04c4e9e95c5da8862dced7e5056455116f83a7e0
#   tower shard: model-00062-of-00062.safetensors
#     sha256 d3087816db95f962a3a74c057b8398e1492458ecd99608f706a400f112a825c6
#   transformers: 5.16.1 (first release carrying glm5_next; vision classes byte-identical
#     to transformers main @ fetch time, diffed in-lane), torch 2.13.0+cpu
#
# Two images, both sized so smart_resize is IDENTITY (no resampling): the pixel pipeline
# (rescale 1/255 -> CLIP mean/std normalize -> patchify) is then exact arithmetic, so the
# Rust preprocessor can be gated bit-tight on them and the tower is gated independently
# of resize kernels (gemma-vision lane law).
#   deterministic-112: 112x112 procedural RGB (seeded), grid 8x8, 16 merged tokens
#   text-448: 448x448 white canvas with unguessable seeded text, grid 32x32, 256 tokens
#     (also the can't-hallucinate probe image: bank the ground-truth string)
#
# Banked per image (f32 LE binaries + meta.json):
#   pixels.bin           processor pixel_values [n_patches, 1176] (c,t,ph,pw flat order)
#   pos_ids.bin          u32 LE [n_patches, 2] (h,w) block-major merge order
#   stage_patch.bin      after patch_embed                [n_patches, 1024]
#   stage_blk0.bin       after block 0                    [n_patches, 1024]
#   stage_post.bin       after all blocks + post_layernorm [n_patches, 1024]
#   stage_down.bin       after downsample (pre-merger)    [n_tokens, 4096]
#   out_f32.bin          merger pooler_output, f32 weights [n_tokens, 4096]
#   out_bf16.bin         merger pooler_output, bf16 weights (artifact-dtype class row)
import json, hashlib, pathlib, sys
import numpy as np
import torch

torch.manual_seed(0)
torch.set_grad_enabled(False)
torch.set_num_threads(8)

HERE = pathlib.Path(__file__).resolve().parent
SHARD = pathlib.Path.home() / "models/glm53-vision/model-00062-of-00062.safetensors"
OUT = HERE / "fixtures"
OUT.mkdir(exist_ok=True)

CFG = json.load(open(HERE.parent / "glm53-flash-bringup-20260827/glm-config.json"))
VCFG = CFG["vision_config"]

from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel
from transformers.models.glm5_next.image_processing_glm5_next import Glm5NextImageProcessor
from safetensors import safe_open

sha = hashlib.sha256(SHARD.read_bytes()).hexdigest()
assert sha == "d3087816db95f962a3a74c057b8398e1492458ecd99608f706a400f112a825c6", sha

vcfg = Glm5NextVisionConfig(**{k: v for k, v in VCFG.items() if k != "model_type"})
model = Glm5NextVisionModel._from_config(vcfg)
model.eval()

state = {}
with safe_open(SHARD, framework="pt") as f:
    for name in f.keys():
        if name.startswith("model.visual."):
            state[name[len("model.visual."):]] = f.get_tensor(name)
missing, unexpected = model.load_state_dict(state, strict=False)
assert not missing, missing
assert not unexpected, unexpected


def det_image_112():
    """Deterministic procedural 112x112 RGB: seeded noise + gradients + a hard edge."""
    rng = np.random.default_rng(20260830)
    h = w = 112
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = (x / w * 255).astype(np.uint8)
    g = (y / h * 255).astype(np.uint8)
    b = rng.integers(0, 256, (h, w), dtype=np.uint8)
    img = np.stack([r, g, b], axis=-1)
    img[40:72, 40:72] = [255, 0, 0]  # hard red square: structure the tower must see
    return img


def text_image_448():
    """448x448 white canvas with an unguessable seeded string (can't-hallucinate probe)."""
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(96518274)
    words = ["ZK%04d" % rng.integers(10000), "QV%04d" % rng.integers(10000),
             "XR%04d" % rng.integers(10000)]
    truth = " ".join(words)
    im = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(im)
    # default bitmap font, scaled up by nearest-neighbor for legibility + determinism
    small = Image.new("RGB", (112, 112), "white")
    ds = ImageDraw.Draw(small)
    for i, wd in enumerate(words):
        ds.text((6, 20 + i * 24), wd, fill="black")
    im = small.resize((448, 448), Image.NEAREST)
    return np.asarray(im), truth


def run(tag, img_np, extra_meta=None):
    proc = Glm5NextImageProcessor()
    batch = proc(images=[torch.from_numpy(img_np).permute(2, 0, 1)], return_tensors="pt")
    pixels = batch["pixel_values"].to(torch.float32)  # [n_patches, 1176]
    grid = batch["image_grid_thw"][0].tolist()  # [t, h, w]
    t, gh, gw = grid
    assert t == 1
    n = gh * gw
    assert pixels.shape == (n, 3 * 2 * 14 * 14), pixels.shape

    from transformers.vision_utils import get_vision_position_ids
    pos = get_vision_position_ids(batch["image_grid_thw"], vcfg.spatial_merge_size)

    stages = {}
    def bank_hook(name):
        def hook(_m, _i, out):
            o = out[0] if isinstance(out, tuple) else out
            stages[name] = o.detach().to(torch.float32).clone()
        return hook

    outs = {}
    for dtype, key in [(torch.float32, "f32"), (torch.bfloat16, "bf16")]:
        m = model.to(dtype)
        hooks = []
        if key == "f32":
            hooks.append(m.patch_embed.register_forward_hook(bank_hook("patch")))
            hooks.append(m.blocks[0].register_forward_hook(bank_hook("blk0")))
            hooks.append(m.post_layernorm.register_forward_hook(bank_hook("post")))
            hooks.append(m.downsample.register_forward_hook(bank_hook("down")))
        r = m(pixels.to(dtype), grid_thw=batch["image_grid_thw"])
        outs[key] = r.pooler_output.detach().to(torch.float32)
        for h in hooks:
            h.remove()
    model.to(torch.float32)

    d = OUT / tag
    d.mkdir(exist_ok=True)
    def bank(name, arr):
        raw = np.ascontiguousarray(arr.numpy() if isinstance(arr, torch.Tensor) else arr)
        (d / name).write_bytes(raw.tobytes())
        return {"shape": list(raw.shape), "sha256": hashlib.sha256(raw.tobytes()).hexdigest()}

    meta = {
        "model": "zai-org/GLM-5.3-Flash",
        "revision": "04c4e9e95c5da8862dced7e5056455116f83a7e0",
        "shard_sha256": sha,
        "transformers": "5.16.1", "torch": torch.__version__,
        "grid_thw": grid, "n_patches": n, "n_tokens": n // 4,
        "pixels": bank("pixels.bin", pixels),
        "pos_ids": bank("pos_ids.bin", pos.to(torch.uint32).numpy()),
        "stage_patch": bank("stage_patch.bin", stages["patch"]),
        "stage_blk0": bank("stage_blk0.bin", stages["blk0"]),
        "stage_post": bank("stage_post.bin", stages["post"]),
        "stage_down": bank("stage_down.bin", stages["down"].reshape(-1, 4096)),
        "out_f32": bank("out_f32.bin", outs["f32"]),
        "out_bf16": bank("out_bf16.bin", outs["bf16"]),
        "f32_vs_bf16_max_abs": float((outs["f32"] - outs["bf16"]).abs().max()),
        "out_f32_max_abs": float(outs["f32"].abs().max()),
    }
    if extra_meta:
        meta.update(extra_meta)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(tag, "grid", grid, "tokens", n // 4,
          "| f32 out max_abs %.4f" % meta["out_f32_max_abs"],
          "| f32-vs-bf16 max_abs %.5f" % meta["f32_vs_bf16_max_abs"])


def det_image_448x224():
    """Well-conditioned large NON-SQUARE fixture (grid 32x16): every patch distinct, so
    large rope positions (h to 31, w to 15) and the h!=w axis order are pinned tightly.
    (text448 cannot pin numerics: a mostly-white canvas makes ~1k near-identical tokens,
    and softmax ties amplify f32 reduction-order noise — upstream fresh-vs-banked differs
    from ITSELF by ~1.0 max-abs at post_blocks on it; measured 2026-08-30.)"""
    rng = np.random.default_rng(31415926)
    h, w = 448, 224
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = ((np.sin(x / 9.0) * 0.5 + 0.5) * 255).astype(np.uint8)
    g = ((np.cos(y / 13.0) * 0.5 + 0.5) * 255).astype(np.uint8)
    b = rng.integers(0, 256, (h, w), dtype=np.uint8)
    return np.stack([r, g, b], axis=-1)


img112 = det_image_112()
from PIL import Image
Image.fromarray(img112).save(OUT / "det112.png")
run("det112", img112)

img_big = det_image_448x224()
Image.fromarray(img_big).save(OUT / "det448x224.png")
run("det448x224", img_big)

img_text, truth = text_image_448()
Image.fromarray(img_text).save(OUT / "text448.png")
run("text448", img_text, {"ground_truth_text": truth,
                          "role": "can't-hallucinate probe image; numerically "
                                  "ill-conditioned for stage parity (see det448x224 note)"})
print("ground truth:", truth, file=sys.stderr)
