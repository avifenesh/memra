import json, os, struct, collections, sys
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
files = sorted({f for k, f in idx.items() if k.startswith("mtp.")})
by = collections.Counter(); n = collections.Counter()
for f in files:
    with open(os.path.join(d, f), "rb") as fh:
        hl = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hl))
    for k, v in hdr.items():
        if not k.startswith("mtp.") or k == "__metadata__":
            continue
        size = v["data_offsets"][1] - v["data_offsets"][0]
        parts = k.split(".")
        if ".experts." in k:
            cat = "routed experts"
        elif ".shared_experts." in k:
            cat = "shared expert"
        elif ".attn." in k or "attn" in parts[3:4]:
            cat = "attention"
        else:
            cat = ".".join(parts[2:4]) if len(parts) > 3 else parts[-1]
        by[cat] += size; n[cat] += 1
tot = sum(by.values())
print(f"mtp tensors {sum(n.values())}, {tot/1e9:.2f} GB")
for c, b in by.most_common(20):
    print(f"{b/1e9:8.3f} GB  {n[c]:6d}  {c}")
blocks = sorted({k.split('.')[1] for k in idx if k.startswith("mtp.")})
print("mtp block ids", blocks[:12])
