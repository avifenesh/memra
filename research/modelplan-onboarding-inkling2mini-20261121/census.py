#!/usr/bin/env python3
"""Header census of an Inkling 2 Mini checkpoint directory (no weight bytes are read).

Reads the safetensors headers of the indexed shard and of the auxiliary drafter file
`mtp.safetensors`, plus `hf_quant_config.json`, and writes two files:

  census.tsv           one row per tensor: file, name, dtype, shape, bytes and the
                       hf_quant_config parameter group that declares it
  census-classes.tsv   the same census grouped by name class, losslessly: the script expands
                       its own output and refuses to write unless the expansion reproduces
                       every file, name, dtype and shape
  census-groups.tsv    one row per hf_quant_config parameter group with every field of its
                       declaration (compute_dtype, output_dtype, fwd_output_dtype,
                       weight_transform, weight_block_shape, weight_scale_format,
                       required_activation_quantization), its tensor count, and what the
                       headers store for it: weight dtypes and the shapes of its scale planes

Usage: census.py CHECKPOINT_DIR OUT_DIR
"""
import hashlib
import json
import os
import re
import struct
import sys
from collections import defaultdict

REPO = "thinkingmachines/Inkling-2-Mini"
REV = "ed40e7da625f594fc9c4990f5f66be24873c22d7"
FILES = ("model-00001-of-00001.safetensors", "mtp.safetensors")
QUANT_PART = {"model-00001-of-00001.safetensors": "model", "mtp.safetensors": "mtp"}
GROUP_KEYS = {
    "compute_dtype",
    "output_dtype",
    "fwd_output_dtype",
    "weight_transform",
    "weight_block_shape",
    "weight_scale_format",
    "required_activation_quantization",
    "tensors",
}
INDEXED = re.compile(r"^(.*?(?:layers|attn_queries|mlp_queries))\.(\d+)(\..*)?$")


def header(path):
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(size))
    head.pop("__metadata__", None)
    return head, size


def group_label(group):
    act = group.get("required_activation_quantization")
    label = group["compute_dtype"]
    if group.get("fwd_output_dtype"):
        label += "/fwd=" + group["fwd_output_dtype"]
    if act:
        label += "/act=" + act["format"]
    return label


def ranges(values):
    values = sorted(values)
    out, start = [], values[0]
    for prev, cur in zip(values, values[1:] + [None]):
        if cur != prev + 1:
            out.append(str(start) if start == prev else f"{start}-{prev}")
            start = cur
    return ",".join(out)


def parse_ranges(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def expand(rows):
    names = {}
    for file, pattern, dtype, shape, indices in rows:
        for index in parse_ranges(indices) if indices != "-" else [None]:
            name = pattern if index is None else pattern.replace("{N}", str(index))
            assert (file, name) not in names, name
            names[(file, name)] = (dtype, shape)
    return names


def main(checkpoint, out_dir):
    quant = json.load(open(os.path.join(checkpoint, "hf_quant_config.json")))
    assert quant["quant_method"] == "inkling2", quant["quant_method"]
    declared = {}
    for part in ("model", "mtp"):
        for group in quant[part]["parameter_groups"]:
            for tensor in group["tensors"]:
                assert (part, tensor) not in declared, tensor
                declared[(part, tensor)] = group_label(group)
    groups = []
    group_of = {}
    for part in ("model", "mtp"):
        for number, group in enumerate(quant[part]["parameter_groups"]):
            unknown = set(group) - GROUP_KEYS
            assert not unknown, f"{part} group {number} has undeclared keys {sorted(unknown)}"
            groups.append((part, number, group, set(), set()))
            for tensor in group["tensors"]:
                group_of[(part, tensor)] = len(groups) - 1
    index = json.load(open(os.path.join(checkpoint, "model.safetensors.index.json")))
    rows, classes = [], defaultdict(set)
    totals = {}
    for file in FILES:
        head, header_bytes = header(os.path.join(checkpoint, file))
        if file == FILES[0]:
            assert set(index["weight_map"]) == set(head), "index and shard header differ"
        data = 0
        for name, info in sorted(head.items()):
            size = info["data_offsets"][1] - info["data_offsets"][0]
            data += size
            shape = ",".join(map(str, info["shape"]))
            # A scale, amax or denominator plane is declared through its weight.
            base = re.sub(r"_(scale|amax|outer_scale_denominator)$", "", name)
            group = declared.get((QUANT_PART[file], base))
            assert group is not None, f"{file}:{name} is in no parameter group"
            rows.append((file, name, info["dtype"], shape, str(size), group))
            stored = groups[group_of[(QUANT_PART[file], base)]]
            if name == base:
                stored[3].add(info["dtype"])
            else:
                # A plane: its suffix, dtype and the grid its last dimension gives over the
                # weight's logical input width (two codes per byte for U8 weights).
                weight = head[base]
                inputs = weight["shape"][-1] * (2 if weight["dtype"] == "U8" else 1)
                grid = f"in/{inputs // info['shape'][-1]}" if len(info["shape"]) == 3 else "per-expert" if info["shape"] != [1] else "scalar"
                stored[4].add(f"{name[len(base):]}:{info['dtype']}:{grid}")
            m = INDEXED.match(name)
            pattern, layer = (f"{m.group(1)}.{{N}}{m.group(3) or ''}", int(m.group(2))) if m else (name, None)
            classes[(file, pattern, info["dtype"], shape)].add(layer)
        totals[file] = (len(head), data, header_bytes)
    assert not {k for k in declared if not any(r[0] == FILES[0 if k[0] == "model" else 1] and r[1] == k[1] for r in rows)}, "a declared tensor is absent"
    class_rows = []
    for (file, pattern, dtype, shape), layers in sorted(classes.items()):
        class_rows.append((file, pattern, dtype, shape, "-" if layers == {None} else ranges(layers)))
    assert expand(class_rows) == {(r[0], r[1]): (r[2], r[3]) for r in rows}, "classes do not reproduce the census"
    os.makedirs(out_dir, exist_ok=True)
    body = "".join("\t".join(row) + "\n" for row in rows)
    with open(os.path.join(out_dir, "census.tsv"), "w") as f:
        f.write(f"# {REPO}@{REV}: every safetensors header tensor. Generated by census.py.\n")
        for file in FILES:
            count, data, header_bytes = totals[file]
            f.write(f"# {file}\ttensors={count}\tdata_bytes={data}\theader_bytes={header_bytes}\n")
        f.write(f"# rows_sha256\t{hashlib.sha256(body.encode()).hexdigest()}\n")
        f.write("file\tname\tdtype\tshape\tbytes\tquant_group\n")
        f.write(body)
    with open(os.path.join(out_dir, "census-classes.tsv"), "w") as f:
        f.write(f"# {REPO}@{REV}: the header census grouped by name class.\n")
        f.write("# Generated by census.py; {N} = layer, drafter block, ViT block or query index.\n")
        for file in FILES:
            f.write(f"# {file}\ttensors={totals[file][0]}\tdata_bytes={totals[file][1]}\n")
        f.write("file\tpattern\tdtype\tshape\tindices\n")
        for row in class_rows:
            f.write("\t".join(row) + "\n")
    with open(os.path.join(out_dir, "census-groups.tsv"), "w") as f:
        f.write(f"# {REPO}@{REV}: the parameter groups of hf_quant_config.json as declared,\n")
        f.write("# beside what the safetensors headers store for each group. Generated by census.py.\n")
        f.write("# stored_planes: suffix:dtype:grid, where in/N is one value per N logical inputs of a row.\n")
        f.write("part\tgroup\ttensors\tcompute_dtype\toutput_dtype\tfwd_output_dtype\tweight_transform\tweight_block_shape\tweight_scale_format\trequired_activation_quantization\tstored_weight_dtypes\tstored_planes\n")
        for part, number, group, dtypes, planes in groups:
            f.write("\t".join([
                part,
                str(number),
                str(len(group["tensors"])),
                group["compute_dtype"],
                group["output_dtype"],
                group.get("fwd_output_dtype", "-"),
                group["weight_transform"],
                json.dumps(group["weight_block_shape"], separators=(",", ":")),
                group.get("weight_scale_format", "-"),
                json.dumps(group["required_activation_quantization"], sort_keys=True, separators=(",", ":")),
                ",".join(sorted(dtypes)),
                ",".join(sorted(planes)) or "-",
            ]) + "\n")
    print(f"{len(rows)} tensors, {len(class_rows)} classes, {len(groups)} groups -> {out_dir}")
    for file in FILES:
        print(file, *totals[file])


if __name__ == "__main__":
    main(*sys.argv[1:])
