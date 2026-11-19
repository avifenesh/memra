"""Range-read the four engram lookup_table tensors (I32) of the pinned checkpoint and summarize them."""
import array, hashlib, json, struct, sys, urllib.request

repo, rev = sys.argv[1], sys.argv[2]
base = f"https://huggingface.co/{repo}/resolve/{rev}/"
index = json.loads(urllib.request.urlopen(base + "model.safetensors.index.json", timeout=60).read())


def get(path, a, b):
    req = urllib.request.Request(base + path, headers={"Range": f"bytes={a}-{b}", "User-Agent": "header-census"})
    return urllib.request.urlopen(req, timeout=120).read()


out = {}
for layer in (1, 11, 21, 31):
    name = f"model.layers.{layer}.engram_layer.lookup_table"
    shard = index["weight_map"][name]
    n = struct.unpack("<Q", get(shard, 0, 7))[0]
    hdr = json.loads(get(shard, 8, 8 + n - 1))
    s, e = hdr[name]["data_offsets"]
    raw = get(shard, 8 + n + s, 8 + n + e - 1)
    vals = array.array("i")
    vals.frombytes(raw)
    distinct = len(set(vals))
    out[name] = {
        "shard": shard,
        "dtype": hdr[name]["dtype"],
        "shape": hdr[name]["shape"],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "min": min(vals),
        "max": max(vals),
        "distinct": distinct,
        "identity_prefix": next((i for i, v in enumerate(vals) if v != i), len(vals)),
        "pad_120002": vals[120002] if len(vals) > 120002 else None,
    }
json.dump(out, sys.stdout, indent=1)
print()
