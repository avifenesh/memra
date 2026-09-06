#!/usr/bin/env python3
"""Audit frozen teacher-forced logits and independently recompute their metrics.

This validates measurements, not checkpoint quality or serving admission.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

WINDOWS = ((0, 160), (32768, 1025), (131072, 4097))
STEPS = 64


def log_probs(values):
    if not values or not all(map(math.isfinite, values)):
        raise ValueError("empty or non-finite logits")
    peak = max(values)
    log_sum = math.log(math.fsum(math.exp(x - peak) for x in values))
    return [(x - peak) - log_sum for x in values]


def metrics(a, b, target):
    p, q = log_probs(a), log_probs(b)
    return {
        "reference_nll": -p[target],
        "matrix_nll": -q[target],
        "kl_reference_matrix": math.fsum(math.exp(x) * (x - y) for x, y in zip(p, q)),
        "total_variation": math.fsum(abs(math.exp(x) - math.exp(y)) for x, y in zip(p, q)) / 2,
        "max_logit_delta": max(abs(x - y) for x, y in zip(a, b)),
    }


def audit(root):
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
    if len(rows) != len(WINDOWS) * STEPS:
        raise ValueError("incomplete or extra scored rows")
    keyed = {(row["window"], row["step"]): row for row in rows}
    if set(keyed) != {(w, s) for w in range(len(WINDOWS)) for s in range(STEPS)}:
        raise ValueError("missing, duplicate or unknown scored rows")
    summaries = []
    for window, (offset, prefix) in enumerate(WINDOWS):
        token_bytes = (root / f"window-{window}.tokens.u32le").read_bytes()
        if len(token_bytes) != (prefix + STEPS) * 4:
            raise ValueError("token bank length mismatch")
        tokens = struct.unpack(f"<{prefix + STEPS}I", token_bytes)
        vocab = keyed[window, 0]["vocab"]
        if not isinstance(vocab, int) or not 1 <= vocab <= 1_000_000:
            raise ValueError("invalid vocabulary size")
        paths = [root / f"window-{window}.{arm}.f32le" for arm in ("reference", "matrix")]
        if any(path.stat().st_size != STEPS * vocab * 4 for path in paths):
            raise ValueError("logit bank length mismatch")
        measures = []
        with paths[0].open("rb") as reference, paths[1].open("rb") as matrix:
            for step in range(STEPS):
                row = keyed[window, step]
                if (row["offset"], row["prefix"], row["vocab"], row["target"]) != (
                    offset, prefix, vocab, tokens[prefix + step]
                ):
                    raise ValueError("scored row/input identity mismatch")
                blobs = (reference.read(vocab * 4), matrix.read(vocab * 4))
                for arm, blob in zip(("reference", "matrix"), blobs):
                    if hashlib.sha256(blob).hexdigest() != row[f"{arm}_sha256"]:
                        raise ValueError(f"{arm} row hash mismatch at {window}/{step}")
                a, b = (struct.unpack(f"<{vocab}f", blob) for blob in blobs)
                measured = metrics(a, b, row["target"])
                for name, value in measured.items():
                    if not math.isfinite(row[name]) or not math.isclose(
                        value, row[name], rel_tol=1e-9, abs_tol=1e-9
                    ):
                        raise ValueError(f"metric mismatch {name} at {window}/{step}")
                for arm, values in zip(("reference", "matrix"), (a, b)):
                    if max(range(vocab), key=values.__getitem__) != row[f"{arm}_top1"]:
                        raise ValueError(f"argmax mismatch at {window}/{step}")
                measures.append(measured)
        summary = {
            "window": window, "offset": offset, "prefix": prefix, "rows": STEPS,
            "token_sha256": hashlib.sha256(token_bytes).hexdigest(),
        }
        for name in ("reference_nll", "matrix_nll", "kl_reference_matrix", "total_variation"):
            summary[f"mean_{name}"] = math.fsum(m[name] for m in measures) / STEPS
        summary["mean_nll_delta"] = summary["mean_matrix_nll"] - summary["mean_reference_nll"]
        summary["max_logit_delta"] = max(m["max_logit_delta"] for m in measures)
        summary["max_total_variation"] = max(m["total_variation"] for m in measures)
        summary["top1_agree"] = sum(keyed[window, s]["reference_top1"] == keyed[window, s]["matrix_top1"] for s in range(STEPS))
        summary["reported_sample_agree"] = sum(keyed[window, s]["reference_sample"] == keyed[window, s]["matrix_sample"] for s in range(STEPS))
        summaries.append(summary)
    return {
        "status": "AUDITED_MEASUREMENTS_NOT_QUALITY_ADMISSION",
        "rows": len(rows), "row_hashes_verified": len(rows) * 2,
        "metrics_independently_recomputed": len(rows) * 5,
        "windows": summaries,
        "mean_nll_delta": math.fsum(s["mean_nll_delta"] for s in summaries) / len(summaries),
        "mean_total_variation": math.fsum(s["mean_total_variation"] for s in summaries) / len(summaries),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.receipt_directory), indent=2, allow_nan=False))
