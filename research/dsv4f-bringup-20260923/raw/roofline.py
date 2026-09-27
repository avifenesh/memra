#!/usr/bin/env python3
"""Per-token weight bytes for DSv4-Flash decode from the safetensors headers.

Routed experts count top-k of n_routed per layer; every other tensor counts once
(shared expert, attention, norms, HC, embedding row excluded, head included).
The MTP/DSpark tensors are reported separately."""
import json, struct, glob, re, sys, os
d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
topk = cfg.get("num_experts_per_tok"); nexp = cfg.get("n_routed_experts")
tot = {"routed": 0, "other": 0, "embed": 0, "mtp": 0}
for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        b = meta["data_offsets"][1] - meta["data_offsets"][0]
        if name.startswith("mtp") or ".mtp." in name:
            tot["mtp"] += b
        elif re.search(r"\.experts\.\d+\.", name):
            tot["routed"] += b
        elif "embed" in name:
            tot["embed"] += b
        else:
            tot["other"] += b
per_tok = tot["other"] + tot["routed"] * topk / nexp
print(json.dumps({"topk": topk, "n_routed": nexp, "bytes": tot,
                  "per_token_bytes": per_tok, "per_token_GB": per_tok / 1e9}, indent=1))
