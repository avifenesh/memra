"""Read safetensors headers of a pinned HF revision by HTTP range requests (no weights)."""
import hashlib, json, struct, sys, urllib.request

repo, rev = sys.argv[1], sys.argv[2]
base = f"https://huggingface.co/{repo}/resolve/{rev}/"


def get(path, rng=None):
    req = urllib.request.Request(base + path, headers={"User-Agent": "header-census"})
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


idx = get("model.safetensors.index.json")
index = json.loads(idx)
files = sorted(set(index["weight_map"].values()))
out = {"repo": repo, "revision": rev, "index_sha256": hashlib.sha256(idx).hexdigest(), "files": {}, "tensors": []}
for name in files:
    n = struct.unpack("<Q", get(name, (0, 7)))[0]
    raw = get(name, (8, 8 + n - 1))
    hdr = json.loads(raw)
    meta = hdr.pop("__metadata__", None)
    out["files"][name] = {"header_len": n, "header_sha256": hashlib.sha256(raw).hexdigest(), "metadata": meta}
    for t, v in hdr.items():
        a, b = v["data_offsets"]
        out["tensors"].append([t, v["dtype"], v["shape"], name, b - a])
    print(name, len(hdr), file=sys.stderr)
for name in ["config.json", "hf_quant_config.json", "quant_config.json", "tokenizer.json",
             "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]:
    try:
        b = get(name)
    except Exception as e:  # noqa: BLE001
        out["files"][name] = {"error": str(e)}
        continue
    out["files"][name] = {"size": len(b), "sha256": hashlib.sha256(b).hexdigest()}
    if name in ("config.json", "hf_quant_config.json", "quant_config.json"):
        open(sys.argv[3] + "/" + name, "wb").write(b)
json.dump(out, sys.stdout)
