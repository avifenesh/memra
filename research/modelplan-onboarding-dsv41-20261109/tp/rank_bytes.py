"""Per-rank device bytes of the DeepSeek-V4.1 trunk under the dsv41_gpu::tp partition, from the safetensors headers.

usage: rank_bytes.py CKPT_DIR [N ...]   (prints JSON lines: one per N)
Classes: split by N (routed experts, wq_b, wo_a, wo_b, attn_sink, head), replicated (every other trunk tensor),
host (embed, Engram tables), not loaded by the trunk (mtp.* DSpark blocks, vision). wo_a is loaded as bf16 (twice
its fp8 bytes) and its fp8 scales are not kept.
"""
import json
import os
import re
import struct
import sys

ckpt = sys.argv[1]
ns = [int(x) for x in sys.argv[2:]] or [1, 2, 4]
DT = {"F8_E4M3": 1, "F8_E8M0": 1, "I8": 1, "BF16": 2, "F32": 4, "F16": 2, "I32": 4, "I64": 8, "U8": 1, "BOOL": 1}
tensors = {}
for f in sorted(os.listdir(ckpt)):
    if not f.endswith(".safetensors"):
        continue
    with open(os.path.join(ckpt, f), "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    for k, v in h.items():
        if k == "__metadata__":
            continue
        sz = DT[v["dtype"]]
        for d in v["shape"]:
            sz *= d
        tensors[k] = (v["dtype"], sz)

cls = {"split": 0, "replicated": 0, "host": 0, "other": 0}
detail = {}
for k, (dt, sz) in tensors.items():
    if k.startswith("mtp.") or k.startswith("aligner.") or "vision" in k or k.startswith("visual") or "mm_projector" in k:
        c = "other"
    elif k == "embed.weight" or ".engram.embed." in k:
        c = "host"
    elif re.match(r"layers\.\d+\.ffn\.experts\.", k) or k == "head.weight" or re.search(
        r"\.attn\.(wq_b|wo_a|wo_b)\.(weight|scale)$|\.attn\.attn_sink$", k
    ):
        c = "split"
        if ".attn.wo_a." in k:
            sz = sz * 2 if k.endswith(".weight") else 0
    else:
        c = "replicated"
    cls[c] += sz
    key = re.sub(r"\d+", "#", k)
    detail.setdefault((c, key), 0)
    detail[(c, key)] += sz

for n in ns:
    per = cls["split"] / n + cls["replicated"]
    print(json.dumps({"ranks": n, "device_bytes_per_rank": int(per), "split_total": cls["split"],
                      "replicated_per_rank": cls["replicated"], "host": cls["host"], "not_in_trunk": cls["other"],
                      "per_rank_gib": round(per / 2**30, 2)}))
top = sorted(((v, c, k) for (c, k), v in detail.items() if c == "replicated"), reverse=True)[:8]
print(json.dumps({"largest_replicated": [[k, round(v / 2**30, 3)] for v, c, k in top]}))
