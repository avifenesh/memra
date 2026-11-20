"""The DFly2 drafter's MLP widths in a pinned HF revision, from its safetensors headers (HTTP range
requests, no weights): every `model.dfly_layers.*.mlp.*` and `hidden_correction.*` tensor's dtype
and shape, and the width each layer's gate/up/down imply (NVFP4 `U8` weights pack two inputs a
byte).

usage: python3 -I source_widths.py REPO REVISION > widths.json
"""

import json
import struct
import sys
import urllib.request

repo, rev = sys.argv[1], sys.argv[2]
base = f"https://huggingface.co/{repo}/resolve/{rev}/"


def get(path, rng=None):
    req = urllib.request.Request(base + path, headers={"User-Agent": "header-census"})
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


index = json.loads(get("model.safetensors.index.json"))["weight_map"]
names = sorted(n for n in index if n.startswith("model.dfly_layers.") and (".mlp." in n or "hidden_correction" in n))
files = sorted({index[n] for n in names})
rows, widths = [], {}
for f in files:
    n = struct.unpack("<Q", get(f, (0, 7)))[0]
    hdr = json.loads(get(f, (8, 8 + n - 1)))
    for name in names:
        if index[name] != f or name not in hdr:
            continue
        dtype, shape = hdr[name]["dtype"], hdr[name]["shape"]
        rows.append({"name": name, "dtype": dtype, "shape": shape, "shard": f})
        if name.endswith(".weight") and ".mlp." in name:
            layer = name.split(".")[2]
            cols = shape[1] * (2 if dtype == "U8" else 1)
            width = shape[1 if name.endswith("down_proj.weight") else 0]
            width = cols if name.endswith("down_proj.weight") else width
            widths.setdefault(layer, set()).add(width)
print(json.dumps({"repo": repo, "revision": rev, "tensors": rows,
                  "mlp_width_per_layer": {k: sorted(v) for k, v in sorted(widths.items())}}, indent=1))
