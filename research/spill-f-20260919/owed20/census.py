#!/usr/bin/env python3
"""OWED 20 tensor census of the pinned Step-3.7-Flash IQ4_XS GGUF, from its shard headers.

Usage: census.py  -> census.json and a printed summary (reads the first 16 MiB of each file at
stepfun-ai/Step-3.7-Flash-GGUF@0b69336d2fd2adfdef9c66e425f7778196c31482 from MEMRA_F_HEADS, each
checked against raw/HEADS.sha256; see raw/README.md).

For every tensor: name, ggml type, shape, and its byte extent. The extent is the offset delta to the
next tensor in the same shard (the last tensor runs to the pinned file size), and it must equal
the size the type and shape predict; any mismatch refuses the census. The expert bank is every
`blk.N.ffn_{gate,up,down}_exps.weight`; the per-expert slice is the bank tensor divided by
`expert_count`.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
# The 16 MiB heads stay out of git (raw/README.md); their hashes are tracked in raw/HEADS.sha256.
HEADS = Path(os.environ.get("MEMRA_F_HEADS", str(Path.home() / ".local/share/memra-lane-f-private/owed20/raw")))
PIN = "stepfun-ai/Step-3.7-Flash-GGUF@0b69336d2fd2adfdef9c66e425f7778196c31482"
FILES = [("IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf", 46483327296,
          "b940497a9cec2f801f07e3a9783f2115fd8bf79cbd453225b4f73d86bcd11259"),
         ("IQ4_XS/Step-3.7-flash-IQ4_XS-00002-of-00003.gguf", 46999941600,
          "e7e0caaaf0057fabc8bf9b71cbe41322f9945a44df7240bb58e6b7c375e7ffec"),
         ("IQ4_XS/Step-3.7-flash-IQ4_XS-00003-of-00003.gguf", 11510293728,
          "ccbd3df81b4f4cb8e73d899734944bcbdefcf436faec9203353419c6750c0590"),
         ("Step3.7-flash-mtp-Q8_0.gguf", 3707276416,
          "469a81667a6cd6d87a85d501d57155fd90cee5af7010fd289c5169881763fd57")]
# ggml type id: (name, elements per block, bytes per block)
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
         7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
         12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
         16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
         20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
         24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
         29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2)}


class Reader:
    def __init__(self, data):
        self.b, self.p = data, 0

    def take(self, n):
        if self.p + n > len(self.b):
            raise SystemExit("REFUSED: header longer than the fetched head")
        out = self.b[self.p:self.p + n]
        self.p += n
        return out

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode("utf-8", "replace")

    def value(self, t):
        fixed = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        if t in fixed:
            return struct.unpack(fixed[t], self.take(struct.calcsize(fixed[t])))[0]
        if t == 8:
            return self.string()
        if t == 9:
            inner, n = self.u32(), self.u64()
            return [self.value(inner) for _ in range(n)]
        raise SystemExit(f"REFUSED: unknown kv type {t}")


def parse(data):
    r = Reader(data)
    if r.take(4) != b"GGUF":
        raise SystemExit("REFUSED: not a GGUF head")
    version, n_tensors, n_kv = r.u32(), r.u64(), r.u64()
    kv = {}
    for _ in range(n_kv):
        key = r.string()
        kv[key] = r.value(r.u32())
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        dims = [r.u64() for _ in range(r.u32())]
        ttype, offset = r.u32(), r.u64()
        tensors.append({"name": name, "dims": dims, "type_id": ttype, "offset": offset})
    align = kv.get("general.alignment", 32)
    data_start = (r.p + align - 1) // align * align
    return version, kv, tensors, data_start


def predicted(t):
    name, per, size = TYPES[t["type_id"]]
    n = 1
    for d in t["dims"]:
        n *= d
    if n % per:
        raise SystemExit(f"REFUSED: {t['name']} has {n} elements, not a multiple of {name}'s block {per}")
    return name, n, n // per * size


def main():
    out = {"pin": PIN, "files": [], "tensors": []}
    for path, size, sha in FILES:
        name = Path(path).stem + ".head.bin"
        head = (HEADS / name).read_bytes()
        want = dict(l.split()[::-1] for l in (HERE / "raw/HEADS.sha256").read_text().splitlines() if l.strip())
        if hashlib.sha256(head).hexdigest() != want[name]:
            raise SystemExit(f"REFUSED: {name} differs from raw/HEADS.sha256")
        version, kv, tensors, data_start = parse(head)
        tensors.sort(key=lambda t: t["offset"])
        for i, t in enumerate(tensors):
            end = tensors[i + 1]["offset"] if i + 1 < len(tensors) else size - data_start
            extent = end - t["offset"]
            tname, n, pred = predicted(t)
            # The next tensor starts at the aligned end of this one.
            align = kv.get("general.alignment", 32)
            if not (pred <= extent < pred + align):
                raise SystemExit(f"REFUSED: {path} {t['name']} extent {extent} vs predicted {pred}")
            out["tensors"].append({"file": path, "name": t["name"], "type": tname, "dims": t["dims"],
                                   "elements": n, "bytes": pred})
        out["files"].append({"path": path, "size": size, "sha256": sha, "gguf_version": version,
                             "tensors": len(tensors), "data_start": data_start,
                             "tensor_bytes": sum(predicted(t)[2] for t in tensors),
                             "architecture": kv.get("general.architecture"),
                             "expert_count": kv.get("step35.expert_count"),
                             "expert_used_count": kv.get("step35.expert_used_count"),
                             "block_count": kv.get("step35.block_count"),
                             "split": [kv.get("split.no"), kv.get("split.count")]})
    trunk = [t for t in out["tensors"] if not t["file"].startswith("Step3.7-flash-mtp")]
    mtp = [t for t in out["tensors"] if t["file"].startswith("Step3.7-flash-mtp")]
    bank = [t for t in trunk if t["name"].endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"))]
    experts = out["files"][0]["expert_count"]
    layers = sorted({int(t["name"].split(".")[1]) for t in bank})
    per_layer = {}
    for t in bank:
        per_layer.setdefault(int(t["name"].split(".")[1]), 0)
        per_layer[int(t["name"].split(".")[1])] += t["bytes"]
    slices = sorted({t["bytes"] // experts for t in bank})
    types = sorted({t["type"] for t in bank})
    out["summary"] = {
        "trunk_tensor_bytes": sum(t["bytes"] for t in trunk),
        "trunk_file_bytes": sum(f["size"] for f in out["files"][:3]),
        "expert_bank_bytes": sum(t["bytes"] for t in bank),
        "expert_bank_tensors": len(bank), "expert_layers": len(layers), "first_expert_layer": layers[0],
        "expert_count": experts, "expert_bank_types": types,
        "expert_slice_bytes": slices, "expert_bank_bytes_per_layer": sorted(set(per_layer.values())),
        "non_expert_trunk_bytes": sum(t["bytes"] for t in trunk) - sum(t["bytes"] for t in bank),
        "mtp_file_bytes": out["files"][3]["size"], "mtp_tensor_bytes": sum(t["bytes"] for t in mtp),
        "mtp_expert_bytes": sum(t["bytes"] for t in mtp if "_exps." in t["name"]),
        "tensors": {"trunk": len(trunk), "mtp": len(mtp)},
    }
    (HERE / "census.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    sys.exit(main())
