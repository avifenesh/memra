#!/usr/bin/env python3
"""Gate the native cache-aware FastConformer encoder against the pinned NeMo capture.

The native side is `rnnt-stage encoder`: its own subsampling stem, its own relative-position
attention, its own causal convolution and its own two caches, stepped chunk by chunk over the
`[56, 0]` arm. The reference is `tools/nemo_encoder_oracle.py`, captured on CPU in FP32 with
dither off and one thread.

Every oracle file is hash-checked before it is read, so a capture that was edited or rebuilt
cannot quietly become the thing the native run is compared against.
"""
import argparse, hashlib, json, os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    import numpy as np

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--bound", type=float, default=1e-3)
    args = p.parse_args()

    manifest = json.loads((args.oracle / "MANIFEST.json").read_text())
    chunks = manifest["chunks"]
    rows = []
    worst = 0.0
    passed = True
    for chunk in chunks:
        index = chunk["index"]
        entry = manifest["tensors"][f"chunk-{index:03d}-encoder"]
        reference = args.oracle / entry["path"]
        if sha(reference) != entry["sha256"]:
            raise SystemExit(f"oracle encoder hash mismatch on chunk {index}")
        want = np.fromfile(reference, dtype="<f4")
        native_path = args.native / f"chunk-{index:03d}-encoder.f32"
        if not native_path.exists():
            raise SystemExit(f"native run has no chunk {index}")
        got = np.fromfile(native_path, dtype="<f4")
        if got.shape != want.shape:
            raise SystemExit(
                f"chunk {index}: native emitted {got.shape}, reference {want.shape}"
            )
        if not np.isfinite(got).all():
            raise SystemExit(f"chunk {index}: native output is not finite")
        delta = float(np.abs(got - want).max())
        worst = max(worst, delta)
        ok = delta <= args.bound
        passed &= ok
        rows.append(
            {
                "index": index,
                "mel_frames_in": chunk["mel_frames_in"],
                "drop_extra_pre_encoded": chunk["drop_extra_pre_encoded"],
                "output_frames": chunk["output_frames"],
                "cache_last_channel_len": chunk["cache_last_channel_len"],
                "max_abs": delta,
                "status": "passed" if ok else "failed",
            }
        )
        print(f"chunk {index:3d} max_abs {delta:.10g} {rows[-1]['status']}", flush=True)

    native_chunks = len(list(args.native.glob("chunk-*-encoder.f32")))
    if native_chunks != len(chunks):
        raise SystemExit(
            f"native run produced {native_chunks} chunks, reference has {len(chunks)}"
        )

    receipt = {
        "schema": "memra-rnnt-encoder-v1",
        "status": "passed" if passed else "failed",
        "att_context_size": manifest["att_context_size"],
        "bound_abs": args.bound,
        "max_abs": worst,
        "chunks": len(chunks),
        "binary_sha256": sha(args.binary),
        "archive": manifest["archive"],
        "archive_sha256": manifest["archive_sha256"],
        "oracle_manifest_sha256": sha(args.oracle / "MANIFEST.json"),
        "reference": {
            "torch": manifest["torch"],
            "nemo": manifest["nemo"],
            "threads": manifest["threads"],
            "dither": manifest["dither"],
            "eval_mode": manifest["eval_mode"],
        },
        "streaming": manifest["streaming"],
        "rows": rows,
        "scope": "native cache-aware FastConformer encoder against a pinned CPU FP32 NeMo "
        "capture, encoder only. No frontend, no prompt kernel, no predictor, no joint, no "
        "decoding, no GPU, no serving surface.",
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        f"rnnt encoder {receipt['status']}: {len(chunks)} chunks, max abs {worst:.10g}, "
        f"bound {args.bound}"
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
