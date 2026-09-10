#!/usr/bin/env python3
"""Add calibrated activation scalars to a NEW GGUF, preserving weight bytes.

No weight quantization or re-encoding occurs. The input and output data sections
are independently hashed after writing; an existing output is never replaced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

from calibrate_prefill_fp4 import PROGRAM, projections, sha

# NOT ".input_scale": that suffix is a RESERVED quant auxiliary in the tensor contract
# (QuantAuxTensor::InputScale) and in the safetensors source's is_quant_auxiliary list,
# where it means a ModelOpt static W4A8 activation scale. Minting our scalars under it
# hands the census a different claim about the checkpoint than the one we mean.
SCALE_SUFFIX = ".a4_input_scale"

SOURCE_SHA = "1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a"


def string(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def header(path):
    with path.open("rb") as f:
        def read(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
        def read_string():
            return f.read(read("Q")).decode()
        def value(kind):
            if kind == 8:
                return read_string()
            if kind == 9:
                item, n = read("I"), read("Q")
                return [value(item) for _ in range(n)]
            return read({0:"B", 1:"b", 2:"H", 3:"h", 4:"I", 5:"i", 6:"f", 7:"?", 10:"Q", 11:"q", 12:"d"}[kind])
        if f.read(4) != b"GGUF" or read("I") != 3:
            raise ValueError("requires GGUF v3")
        nt, nm = read("Q"), read("Q")
        metadata, raw = {}, []
        for _ in range(nm):
            start = f.tell()
            name = read_string()
            metadata[name] = value(read("I"))
            end = f.tell()
            f.seek(start)
            raw.append((name, f.read(end-start)))
        start = f.tell()
        tensors = []
        for _ in range(nt):
            name = read_string()
            shape = [read("Q") for _ in range(read("I"))]
            tensors.append({"name": name, "shape": shape, "qtype": read("I"), "offset": read("Q")})
        end = f.tell()
        f.seek(start)
        table = f.read(end-start)
        alignment = metadata.get("general.alignment", 32)
        data_start = (end + alignment-1)//alignment*alignment
        return metadata, raw, tensors, table, data_start, alignment


def range_sha(path, start, length):
    h = hashlib.sha256()
    with path.open("rb") as f:
        f.seek(start)
        while length:
            b = f.read(min(length, 8 << 20))
            if not b:
                raise ValueError("short payload")
            h.update(b)
            length -= len(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--scales", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists() or args.output.resolve() == args.source.resolve():
        raise ValueError("mint requires a new output path")
    record = json.loads(args.scales.read_text())
    # The calibration RECORD is keyed the way the fitting script wrote it; the minted
    # TENSOR carries our own suffix. Keeping the two separate means a re-mint under a
    # different tensor name does not require re-running the fit.
    record_keys = {name+".input_scale" for name in projections().values()}
    expected = {name+SCALE_SUFFIX for name in projections().values()}
    if record["program"] != PROGRAM or set(record["scales"]) != record_keys:
        raise ValueError("activation program must carry exactly 400 declared scalars")
    if sha(args.source) != SOURCE_SHA:
        raise ValueError("served weight artifact hash mismatch")
    meta, raw, tensors, table, data_start, alignment = header(args.source)
    if len(tensors) != 866 or "memra.activation_program" in meta:
        raise ValueError("unexpected source census or already calibrated artifact")
    old_names = {t["name"] for t in tensors}
    if old_names & expected:
        raise ValueError("source already has activation tensors")
    extra = {"memra.activation_program": PROGRAM,
             "memra.activation_calibration_sha256": sha(args.scales),
             "memra.activation_corpus_sha256": record["corpus_lock_sha256"],
             "memra.activation_source_sha256": SOURCE_SHA}
    if set(extra) & set(meta):
        raise ValueError("source has conflicting mint metadata")
    metadata = b"".join((string(n)+struct.pack("<I",8)+string(args.output.stem)) if n == "general.name" else b for n,b in raw)
    metadata += b"".join(string(n)+struct.pack("<I",8)+string(v) for n,v in extra.items())
    old_length = args.source.stat().st_size-data_start
    offset = (old_length+alignment-1)//alignment*alignment
    additions, payload = [], bytearray(offset-old_length)
    for name in sorted(expected):
        record_key = name[: -len(SCALE_SUFFIX)] + ".input_scale"
        multiplier = record["scales"][record_key]["multiplier"]
        if not 0 < multiplier < float("inf"):
            raise ValueError(f"invalid multiplier: {name}")
        encoded = struct.pack("<f", multiplier)
        if not 0 < struct.unpack("<f", encoded)[0] < float("inf"):
            raise ValueError(f"unrepresentable multiplier: {name}")
        additions.append(string(name)+struct.pack("<IQIQ",1,1,0,offset))
        payload.extend(encoded+b"\0"*(alignment-4))
        offset += alignment
    output_header = b"GGUF"+struct.pack("<IQQ",3,len(tensors)+len(additions),len(raw)+len(extra))+metadata+table+b"".join(additions)
    output_start = (len(output_header)+alignment-1)//alignment*alignment
    partial = args.output.with_suffix(args.output.suffix+".partial")
    with partial.open("xb") as dest, args.source.open("rb") as src:
        dest.write(output_header)
        dest.write(b"\0"*(output_start-len(output_header)))
        src.seek(data_start)
        shutil.copyfileobj(src, dest, 8 << 20)
        dest.write(payload)
        dest.flush()
        __import__("os").fsync(dest.fileno())
    old_payload_sha = range_sha(args.source, data_start, old_length)
    if range_sha(partial, output_start, old_length) != old_payload_sha:
        raise ValueError("new artifact changed inherited tensor data")
    _, _, final_tensors, _, final_start, _ = header(partial)
    if final_tensors[:866] != tensors or final_start != output_start or len(final_tensors) != 1266:
        raise ValueError("output header/census mismatch")
    digest = sha(partial)
    # Hard link is an atomic no-overwrite publish, unlike replace/rename.
    __import__("os").link(partial, args.output)
    partial.unlink()
    lock = {"artifact": args.output.name, "sha256": digest, "bytes": args.output.stat().st_size,
            "program": PROGRAM, "source_sha256": SOURCE_SHA, "inherited_payload_sha256": old_payload_sha,
            "inherited_tensor_count": 866, "activation_scale_count": 400, "tensor_count": 1266,
            "calibration_sha256": sha(args.scales), "corpus_lock_sha256": record["corpus_lock_sha256"],
            "mint_script_sha256": sha(__file__), "calibration_script_sha256": record["script_sha256"],
            "qualification": "UNQUALIFIED", "phase_policy": "prefill A4 on selected 400; decode/verify W4A8"}
    args.output.with_suffix(".artifact.lock.json").write_text(json.dumps(lock, indent=2)+"\n")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
