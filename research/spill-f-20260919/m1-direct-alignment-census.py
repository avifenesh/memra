#!/usr/bin/env python3
"""Header-only GGUF census: which per-expert extents can the `direct` spill arm read?

`crates/memra-engine/src/spill_pread.rs::direct_extent_aligned` admits an expert extent to
O_DIRECT only when both its absolute file offset and its length are multiples of 4096; every
other extent is served by the mmap fallback and counted in `fallbacks`. This script reads only
the GGUF header (no tensor payload), derives each stacked expert tensor's per-expert slice
(offset = data_start + tensor_offset + e * slice_bytes, length = slice_bytes) and reports the
aligned share. It predicts the direct arm's resolved-backend mix before any rig time is spent.

Usage: m1-direct-alignment-census.py <file.gguf>   (prints one JSON document)
"""
import hashlib
import json
import struct
import sys

# ggml type id -> (block elements, block bytes). Unknown ids are reported, never guessed.
TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
}
DIRECT_IO_ALIGNMENT = 4096  # spill_pread.rs DIRECT_IO_ALIGNMENT
EXPERT_SUFFIXES = ("_exps.weight",)


class Reader:
    def __init__(self, f):
        self.f = f
        self.pos = 0

    def take(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise SystemExit(f"REFUSED: truncated GGUF header at byte {self.pos}")
        self.pos += n
        return b

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode("utf-8")

    def value(self, t):
        scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
                  10: "<Q", 11: "<q", 12: "<d"}
        if t in scalar:
            fmt = scalar[t]
            return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]
        if t == 8:
            return self.string()
        if t == 9:
            inner, n = self.u32(), self.u64()
            return [self.value(inner) for _ in range(n)]
        raise SystemExit(f"REFUSED: unknown GGUF value type {t}")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    with open(path, "rb") as f:
        r = Reader(f)
        if r.take(4) != b"GGUF":
            raise SystemExit("REFUSED: not a GGUF file")
        version, n_tensors, n_kv = r.u32(), r.u64(), r.u64()
        alignment = 32
        for _ in range(n_kv):
            key, t = r.string(), r.u32()
            v = r.value(t)
            if key == "general.alignment":
                alignment = int(v)
        infos = []
        for _ in range(n_tensors):
            name, nd = r.string(), r.u32()
            dims = [r.u64() for _ in range(nd)]
            infos.append((name, dims, r.u32(), r.u64()))
        header_end = r.pos
        f.seek(0)
        header_sha = hashlib.sha256(f.read(header_end)).hexdigest()
    data_start = (header_end + alignment - 1) // alignment * alignment
    rows, unknown = [], []
    total = aligned = 0
    total_bytes = aligned_bytes = 0
    overread_total = 0
    for name, dims, t, off in infos:
        if not name.endswith(EXPERT_SUFFIXES):
            continue
        if t not in TYPES:
            unknown.append({"tensor": name, "ggml_type": t})
            continue
        tname, blk, bb = TYPES[t]
        elems = 1
        for d in dims:
            elems *= d
        if elems % blk:
            raise SystemExit(f"REFUSED: {name} elements not a multiple of its block")
        nbytes = elems // blk * bb
        n_expert = dims[-1]  # ggml order: innermost first, expert index outermost
        if nbytes % n_expert:
            raise SystemExit(f"REFUSED: {name} bytes not divisible by its expert count")
        slice_bytes = nbytes // n_expert
        base = data_start + off
        ok = sum(1 for e in range(n_expert)
                 if (base + e * slice_bytes) % DIRECT_IO_ALIGNMENT == 0
                 and slice_bytes % DIRECT_IO_ALIGNMENT == 0)
        # OWED 7 over-read: window = align_up(offset + len) - align_down(offset).
        over = []
        for e in range(n_expert):
            o = base + e * slice_bytes
            start = o - o % DIRECT_IO_ALIGNMENT
            end = -(-(o + slice_bytes) // DIRECT_IO_ALIGNMENT) * DIRECT_IO_ALIGNMENT
            over.append(end - start - slice_bytes)
        overread_total += sum(over)
        over = set(over)
        total += n_expert
        aligned += ok
        total_bytes += nbytes
        aligned_bytes += ok * slice_bytes
        rows.append({"tensor": name, "type": tname, "experts": n_expert,
                     "slice_bytes": slice_bytes, "slice_mod_4096": slice_bytes % 4096,
                     "base_mod_4096": base % 4096, "aligned_slices": ok,
                     "overread_bytes_per_slice": sorted(over)})
    print(json.dumps({
        "file": path.rsplit("/", 1)[-1], "gguf_version": version, "tensors": n_tensors,
        "general_alignment": alignment, "header_bytes": header_end,
        "header_sha256": header_sha, "data_start": data_start,
        "direct_io_alignment": DIRECT_IO_ALIGNMENT,
        "expert_tensors": len(rows), "expert_slices": total, "aligned_slices": aligned,
        "expert_bytes": total_bytes, "aligned_bytes": aligned_bytes,
        "aligned_share": (aligned / total) if total else None,
        "overread_bytes_one_read_of_every_slice": overread_total,
        "unknown_type_tensors": unknown, "per_tensor": rows,
    }, indent=1))


if __name__ == "__main__":
    main()
