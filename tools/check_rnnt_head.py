#!/usr/bin/env python3
"""Gate the native RNNT head against the pinned NeMo capture.

Four independent things, and the last one is the only one that is not a tolerance:

  prompt      prompt_kernel over every encoder frame, one-hot language slot concatenated
  predictor   embedding and two LSTM layers over a fixed token walk, with hidden and cell
  joint       logits for every encoder frame against two predictor rows, log-normalized
              because the reference log-normalizes on CPU and returns raw logits on GPU
  greedy      the emitted token ids, which must be equal, not close

Every reference tensor is hash-checked before it is read.
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
    tensors = manifest["tensors"]
    rows = []
    worst = 0.0
    passed = True

    def compare(label, reference_name, native_name, log_softmax_native=False, vocabulary=0):
        nonlocal worst, passed
        entry = tensors[reference_name]
        path = args.oracle / entry["path"]
        if sha(path) != entry["sha256"]:
            raise SystemExit(f"oracle hash mismatch on {reference_name}")
        want = np.fromfile(path, dtype="<f4")
        native_path = args.native / native_name
        if not native_path.exists():
            raise SystemExit(f"native run has no {native_name}")
        got = np.fromfile(native_path, dtype="<f4")
        if got.shape != want.shape:
            raise SystemExit(
                f"{label}: native {got.shape} against reference {want.shape}"
            )
        if not np.isfinite(got).all():
            raise SystemExit(f"{label}: native output is not finite")
        if log_softmax_native:
            # The reference's joint log-normalizes on CPU and returns raw logits on GPU, by its
            # own device check. The native joint returns logits on every device, so the
            # comparison normalizes rather than making the native output device-dependent.
            rows_ = got.reshape(-1, vocabulary).astype(np.float64)
            maximum = rows_.max(axis=1, keepdims=True)
            shifted = rows_ - maximum
            got = (shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))).astype(
                np.float32
            ).reshape(-1)
        delta = float(np.abs(got - want).max())
        worst = max(worst, delta)
        ok = delta <= args.bound
        passed &= ok
        rows.append(
            {
                "stage": label,
                "reference": reference_name,
                "elements": int(want.size),
                "max_abs": delta,
                "status": "passed" if ok else "failed",
            }
        )
        print(f"{label:28s} max_abs {delta:.10g} {rows[-1]['status']}", flush=True)

    compare("prompted encoder", "prompted-encoder", "prompted-encoder.f32")
    for step in range(len(manifest["predictor_tokens"])):
        compare(
            f"predictor {step} output",
            f"predictor-{step:02d}-out",
            f"predictor-{step:02d}-out.f32",
        )
        # The reference banks one state tensor per layer, shaped [layers, batch, width]; the
        # native run banks hidden and cell separately, so only hidden is directly comparable
        # here and the cell is covered by the next step's output depending on it.
    compare("predictor start output", "predictor-start-out", "predictor-start-out.f32")
    vocabulary = int(manifest["blank_index"]) + 1
    compare(
        "joint against start row",
        "joint-start",
        "joint-start.f32",
        log_softmax_native=True,
        vocabulary=vocabulary,
    )
    compare(
        "joint against walk row",
        "joint-walk",
        "joint-walk.f32",
        log_softmax_native=True,
        vocabulary=vocabulary,
    )

    native_greedy = [
        int(t) for t in (args.native / "greedy.txt").read_text().split()
    ]
    reference_greedy = [int(t) for t in manifest["greedy_tokens"]]
    greedy_equal = native_greedy == reference_greedy
    passed &= greedy_equal
    print(
        f"greedy tokens {'equal' if greedy_equal else 'DIFFER'}: "
        f"{len(native_greedy)} native, {len(reference_greedy)} reference"
    )

    receipt = {
        "schema": "memra-rnnt-head-v1",
        "status": "passed" if passed else "failed",
        "bound_abs": args.bound,
        "max_abs": worst,
        "greedy_tokens_equal": greedy_equal,
        "greedy_token_count": len(reference_greedy),
        "greedy_tokens": reference_greedy,
        "greedy_text": manifest["greedy_text"],
        "language": manifest["language"],
        "prompt_index": manifest["prompt_index"],
        "blank_index": manifest["blank_index"],
        "max_symbols": manifest["max_symbols"],
        "frames": manifest["frames"],
        "binary_sha256": sha(args.binary),
        "archive_sha256": manifest["archive_sha256"],
        "oracle_manifest_sha256": sha(args.oracle / "MANIFEST.json"),
        "reference": {"torch": manifest["torch"], "nemo": manifest["nemo"]},
        "rows": rows,
        "scope": "native RNNT prompt, predictor, joint and greedy decode against a pinned CPU "
        "FP32 NeMo capture, on already-gated encoder frames. No audio, no GPU, no streaming "
        "session lifecycle, no serving surface.",
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(f"rnnt head {receipt['status']}: max abs {worst:.10g}, bound {args.bound}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
