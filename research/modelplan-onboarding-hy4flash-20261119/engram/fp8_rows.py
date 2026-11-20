"""Engram FP8 rows: the artifact's stored rows against the pinned transformers round trip.

Range-reads blocks of consecutive rows of the four Engram tables from the BF16 source
(tencent/Hy4-flash@d88c9dfd) and from the artifact (Avifenesh/Hy4-flash-NVFP4@f9eac06c: F8_E4M3
`weight` plus one F32 `weight_scale` per row), and compares, per row:

- producer: the artifact's codes and scales against `scale = f32(amax) / 448`, codes rounded to
  nearest even and saturated at 448, computed in F32 from the BF16 source row;
- f32: the pinned `_fp8_round_trip` of the source row in float32 against the stored `q * scale`;
- bf16: the same round trip on BF16 activations (what transformers runs with a BF16 model), against
  the stored row rounded to BF16;
- idempotence: the round trip applied to the stored row in float32, and to the exact float64
  `q * scale`;
- the largest code of each row.

    venv/bin/python -I fp8_rows.py OUT.json
"""

import hashlib
import json
import random
import struct
import sys
import urllib.request

import torch

import transformers.models.hy_v4_flash.modeling_hy_v4_flash as model_mod


def blob(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


assert blob(model_mod.__file__) == "2d796211a4f867bd4c46d4a952cfee0375929f72", "modeling pin"
round_trip = model_mod._fp8_round_trip

SOURCE = ("tencent/Hy4-flash", "d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959")
ARTIFACT = ("Avifenesh/Hy4-flash-NVFP4", "f9eac06cf022cfd3dc2e8ed423922466b2adcd87")
LAYERS = (1, 11, 21, 31)
BLOCKS, ROWS, DIM = 8, 256, 96


def get(repo, path, a=None, b=None):
    url = f"https://huggingface.co/{repo[0]}/resolve/{repo[1]}/{path}"
    headers = {"User-Agent": "engram-rows"}
    if a is not None:
        headers["Range"] = f"bytes={a}-{b}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180).read()


def locate(repo, index, name):
    shard = index["weight_map"][name]
    n = struct.unpack("<Q", get(repo, shard, 0, 7))[0]
    hdr = json.loads(get(repo, shard, 8, 8 + n - 1))
    info = hdr[name]
    return shard, 8 + n + info["data_offsets"][0], info["dtype"], info["shape"]


def read(repo, where, first, count, row_bytes):
    shard, base, _, _ = where
    a = base + first * row_bytes
    return get(repo, shard, a, a + count * row_bytes - 1)


src_index = json.loads(get(SOURCE, "model.safetensors.index.json"))
art_index = json.loads(get(ARTIFACT, "model.safetensors.index.json"))
report = {"source": SOURCE, "artifact": ARTIFACT, "blocks_per_layer": BLOCKS, "rows_per_block": ROWS, "layers": {}}
for layer in LAYERS:
    name = f"model.layers.{layer}.engram_layer.multi_head_embedding.embedding.weight"
    src = locate(SOURCE, src_index, name)
    art = locate(ARTIFACT, art_index, name)
    scl = locate(ARTIFACT, art_index, name[: -len("weight")] + "weight_scale")
    assert src[2] == "BF16" and art[2] == "F8_E4M3" and scl[2] == "F32", (src[2], art[2], scl[2])
    rows = src[3][0]
    assert src[3] == [rows, DIM] and art[3] == [rows, DIM] and scl[3] == [rows]
    rng = random.Random(20261120 + layer)
    starts = sorted(rng.randrange(rows - ROWS) for _ in range(BLOCKS))
    x, q, s = [], [], []
    for first in starts:
        x.append(torch.frombuffer(bytearray(read(SOURCE, src, first, ROWS, 2 * DIM)), dtype=torch.bfloat16).view(ROWS, DIM))
        q.append(torch.frombuffer(bytearray(read(ARTIFACT, art, first, ROWS, DIM)), dtype=torch.float8_e4m3fn).view(ROWS, DIM))
        s.append(torch.frombuffer(bytearray(read(ARTIFACT, scl, first, ROWS, 4)), dtype=torch.float32))
    x, q, s = torch.cat(x), torch.cat(q), torch.cat(s)
    n = x.shape[0]

    # The producer rule, in F32 from the BF16 row.
    xf = x.float()
    amax = xf.abs().amax(-1)
    s_rule = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    q_rule = (xf / s_rule[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    stored = q.float() * s[:, None]
    exact = q.double() * s.double()[:, None]

    rt32 = round_trip(xf)
    rt16 = round_trip(x)
    stored16 = stored.to(torch.bfloat16)
    eq16 = rt16.view(torch.int16) == stored16.view(torch.int16)
    rel16 = ((rt16.float() - stored).abs() / stored.abs().clamp_min(1e-30)).masked_fill(stored == 0, 0.0)
    scale16 = (xf.abs().amax(-1).to(torch.bfloat16) / 448).float()
    max_code = q.float().abs().amax(-1)
    report["layers"][str(layer)] = {
        "rows_total": rows,
        "block_starts": starts,
        "rows_sampled": n,
        "zero_rows": int((amax == 0).sum()),
        "producer_codes_equal": int((q_rule.view(torch.uint8) == q.view(torch.uint8)).all(-1).sum()),
        # fp8q.c writes +0 for a value that rounds to zero; torch's cast keeps the sign (-0).
        "producer_code_values_equal": int((q_rule.float() == q.float()).all(-1).sum()),
        "producer_code_diffs_are_signed_zeros": bool(
            ((q_rule.view(torch.uint8) == q.view(torch.uint8)) | ((q_rule.float() == 0) & (q.float() == 0))).all()
        ),
        "producer_scales_equal": int((s_rule.view(torch.int32) == s.view(torch.int32)).sum()),
        "f32_round_trip_bit_equal_rows": int((rt32.view(torch.int32) == stored.view(torch.int32)).all(-1).sum()),
        "bf16_round_trip_bit_equal_rows": int(eq16.all(-1).sum()),
        "bf16_round_trip_bit_equal_elements": int(eq16.sum()),
        "elements": n * DIM,
        "bf16_round_trip_max_rel_diff": float(rel16.max()),
        "bf16_scale_vs_stored_max_rel_diff": float(((scale16 - s).abs() / s).max()),
        "idempotent_f32_rows": int((round_trip(stored).view(torch.int32) == stored.view(torch.int32)).all(-1).sum()),
        # On the exact dequantization (f64 `q * scale`); the F32 product `stored` is rounded.
        "idempotent_f64_rows": int((round_trip(exact) == exact).all(-1).sum()),
        "rows_max_code_448": int((max_code == 448).sum()),
        "amax_min": float(amax.min()),
        "amax_max": float(amax.max()),
        "row_rms_min": float(xf.square().mean(-1).sqrt().min()),
        "row_rms_median": float(xf.square().mean(-1).sqrt().median()),
    }
    print(layer, json.dumps(report["layers"][str(layer)]), flush=True)
json.dump(report, open(sys.argv[1], "w"), indent=1)
