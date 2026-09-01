#!/usr/bin/env python3
"""Patch trained mtp.* tensors into a copy of the official NVFP4 ST checkpoint.

The official `Ornith-1.5-35B-A3B-NVFP4` keeps the whole MTP head BF16, so the
continued-trained head (train_mtp.py export, checkpoint tensor names, BF16)
drops in as a pure tensor replacement. Only shards containing mtp.* tensors are
rewritten; every other file is HARDLINKED from the source dir (no byte copies,
no disk blowup). Every trained tensor must match the original name, shape and
dtype — fail closed on any mismatch.
"""
import argparse
import json
import os
import pathlib
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True, type=pathlib.Path)
    ap.add_argument("--mtp", required=True, type=pathlib.Path, help="trained mtp safetensors")
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    args = ap.parse_args()

    trained = {}
    with safe_open(args.mtp, framework="pt") as f:
        for name in f.keys():
            trained[name] = f.get_tensor(name)
    assert all(k.startswith("mtp.") for k in trained), "trained file must hold mtp.* only"

    idx_path = args.src_dir / "model.safetensors.index.json"
    idx = json.loads(idx_path.read_text())
    wmap = idx["weight_map"]
    src_mtp = {k for k in wmap if k.startswith("mtp.")}
    assert src_mtp == set(trained), (
        f"tensor set mismatch: only-in-src={sorted(src_mtp - set(trained))[:5]} "
        f"only-in-trained={sorted(set(trained) - src_mtp)[:5]}"
    )
    mtp_shards = {wmap[k] for k in src_mtp}
    print(f"{len(trained)} mtp tensors live in shards: {sorted(mtp_shards)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for entry in sorted(os.listdir(args.src_dir)):
        src = args.src_dir / entry
        dst = args.out_dir / entry
        if dst.exists() or src.is_dir() or entry in mtp_shards:
            continue
        os.link(src, dst)  # hardlink: zero-copy share of unchanged files

    for shard in sorted(mtp_shards):
        tensors, meta = {}, None
        with safe_open(args.src_dir / shard, framework="pt") as f:
            meta = f.metadata()
            for name in f.keys():
                if name.startswith("mtp."):
                    orig = f.get_tensor(name)
                    new = trained[name]
                    assert new.shape == orig.shape and new.dtype == orig.dtype, (
                        name, new.shape, orig.shape, new.dtype, orig.dtype
                    )
                    tensors[name] = new
                else:
                    tensors[name] = f.get_tensor(name)
        save_file(tensors, str(args.out_dir / shard), metadata=meta or {"format": "pt"})
        print(f"rewrote {shard}: {sum(1 for n in tensors if n.startswith('mtp.'))} mtp tensors patched")
    print(f"PATCH DONE -> {args.out_dir}")


if __name__ == "__main__":
    main()
