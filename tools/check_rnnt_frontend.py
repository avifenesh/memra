#!/usr/bin/env python3
"""Gate the native RNNT log-mel frontend against the pinned NeMo capture.

`rnnt-stage frontend` binds the checkpoint's own window and filterbank buffers and computes
the whole clip. The reference is the same capture the encoder gate uses, so the two stages are
scored against one artifact.

With `--compose`, the checker also cuts the streaming chunk windows out of the *native* mel the
same way the reference's streaming buffer cuts them out of its own, writes them, and leaves
them for the encoder gate: that is the frontend and the encoder measured as one path rather
than two stages that were each fed the reference's features.
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
    p.add_argument("--native-mel", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--bound", type=float, default=1e-3)
    p.add_argument(
        "--compose",
        type=Path,
        help="write streaming chunk windows cut from the native mel into this directory",
    )
    args = p.parse_args()

    manifest = json.loads((args.oracle / "MANIFEST.json").read_text())
    entry = manifest["tensors"]["mel"]
    reference_path = args.oracle / entry["path"]
    if sha(reference_path) != entry["sha256"]:
        raise SystemExit("oracle mel hash mismatch")
    want = np.fromfile(reference_path, dtype="<f4").reshape(entry["shape"])
    got = np.fromfile(args.native_mel, dtype="<f4")
    if got.size != want.size:
        raise SystemExit(f"native mel has {got.size} values, reference has {want.size}")
    got = got.reshape(want.shape)
    if not np.isfinite(got).all():
        raise SystemExit("native mel is not finite")

    delta = float(np.abs(got - want).max())
    valid = manifest["chunks"][-1]["buffer_idx"] + 1
    per_column = np.abs(got - want).max(axis=0)
    worst_column = int(per_column.argmax())
    passed = delta <= args.bound

    composed = None
    if args.compose:
        args.compose.mkdir(parents=True, exist_ok=False)
        for chunk in manifest["chunks"]:
            index = chunk["index"]
            start = chunk["buffer_idx"] - chunk["pre_encode_frames"]
            width = chunk["mel_frames_in"]
            window = np.zeros((want.shape[0], width), dtype=np.float32)
            source_start = max(0, start)
            lead = source_start - start
            available = min(width - lead, got.shape[1] - source_start)
            window[:, lead : lead + available] = got[
                :, source_start : source_start + available
            ]
            window.tofile(args.compose / f"chunk-{index:03d}-mel.f32.bin")
        (args.compose / "MANIFEST.json").write_text(
            json.dumps({"composed_from": str(args.native_mel)}, indent=2)
        )
        composed = len(manifest["chunks"])

    receipt = {
        "schema": "memra-rnnt-frontend-v1",
        "status": "passed" if passed else "failed",
        "bound_abs": args.bound,
        "max_abs": delta,
        "mean_abs": float(np.abs(got - want).mean()),
        "worst_column": worst_column,
        "worst_column_max_abs": float(per_column[worst_column]),
        "frames": int(want.shape[1]),
        "valid_frames": valid,
        "binary_sha256": sha(args.binary),
        "archive_sha256": manifest["archive_sha256"],
        "oracle_manifest_sha256": sha(args.oracle / "MANIFEST.json"),
        "reference": {
            "torch": manifest["torch"],
            "nemo": manifest["nemo"],
            "dither": manifest["dither"],
            "preprocessor": manifest["preprocessor"],
        },
        "composed_chunks": composed,
        "scope": "native RNNT log-mel against a pinned CPU FP32 NeMo capture. Frontend only: "
        "no encoder, no decoding, no GPU, no serving surface.",
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        f"rnnt frontend {receipt['status']}: max abs {delta:.6g}, mean {receipt['mean_abs']:.6g}, "
        f"bound {args.bound}, worst column {worst_column}"
    )
    if composed:
        print(f"  composed {composed} chunk windows into {args.compose}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
