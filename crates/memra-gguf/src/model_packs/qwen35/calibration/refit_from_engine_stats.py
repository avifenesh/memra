#!/usr/bin/env python3
"""Refit the 400 activation multipliers from ENGINE-measured operand statistics.

The v2 record was fitted off-graph (BF16 module inputs) and its ffn_down class landed 3.6x
wrong against the operand the engine quantizes. This consumes the armed-pass statistics JSON
(`qwen-a4-refit-stats`: per-projection amax and a 64-bin log2 histogram of |x| over the W4A8
arithmetic the engine actually runs) and writes TWO mint-compatible records:

  arm "amax"   multiplier = amax / 2688          (the v3 primary, same recipe, right tap)
  arm "p9999"  multiplier = p99.99(|x|) / 2688   (the twin: clip the rare outliers)

The p9999 arm also records the exact clip fraction it chose per projection: the fraction of
values above the implied global clamp 2688*s. Both records carry the same corpus lock as the
v2 fit, so a minted artifact's provenance chain stays closed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PROGRAM = "qwen35-prefill-nvfp4-a4-v1"
F32_MAX = np.finfo(np.float32).max


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def histogram_quantile(hist, zeros, total, q):
    """Quantile of |x| from the 64-bin log2 histogram (bin b covers [2^(b-40), 2^(b-39)))."""
    target = int(np.ceil(total * q))
    seen = zeros
    for b, count in enumerate(hist):
        nxt = seen + count
        if nxt >= target and count > 0:
            frac = max(target - seen, 1) / count
            return float(np.float32(2.0 ** (b - 40 + frac)))
        seen = nxt
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, required=True, help="qwen-a4-refit-stats output json")
    ap.add_argument("--v2-record", type=Path, required=True,
                    help="the off-graph activation-scales.json, for its corpus lock")
    ap.add_argument("--out-prefix", type=Path, required=True,
                    help="writes <prefix>-amax.json and <prefix>-p9999.json")
    args = ap.parse_args()

    stats = json.loads(args.stats.read_text())
    v2 = json.loads(args.v2_record.read_text())
    arms = {
        "amax": {"multiplier": {}, "amax": {}, "clip_fraction": {}, "dropped": 0},
        "p9999": {"multiplier": {}, "amax": {}, "clip_fraction": {}, "p9999": {}, "dropped": 0},
    }
    for row in stats["projections"]:
        stem = row["name"][: -len(".weight")]
        key = stem + ".input_scale"
        amax = np.float32(row["amax"])
        hist, zeros, total = row["hist"], row["zeros"], row["values"]
        for arm in arms.values():
            value = amax
            if arm is arms["p9999"]:
                q = histogram_quantile(hist, zeros, total, 0.9999)
                arm["p9999"][key] = q
                value = np.float32(q if q is not None and q > 0 else amax)
            scale = np.float32(value / 2688.0)
            if not (np.isfinite(scale) and scale > 0):
                raise ValueError(f"{stem}: invalid {value} -> {scale}")
            arm["multiplier"][key] = float(scale)
            arm["amax"][key] = float(amax)
            # Clip fraction at the implied clamp 2688*s: whole histogram bins strictly above it
            # (bin b starts at 2^(b-40)); a straddling bin is excluded, so the number is a
            # slight undercount of the values that will actually clip.
            clamp = float(np.float32(scale * np.float32(6.0 * 448.0)))
            above = sum(c for b, c in enumerate(hist) if (2.0 ** (b - 40)) > clamp)
            arm["clip_fraction"][key] = above / total

    for name, arm in arms.items():
        record = {
            "program": PROGRAM,
            "source": "engine-measured operand statistics (armed W4A8 pass, qwen-a4-refit-stats)",
            "corpus_lock_sha256": v2["corpus_lock_sha256"],
            "script_sha256": sha(__file__),
            "stats_sha256": sha(args.stats),
            "arm": name,
            "linear_count": len(arm["multiplier"]),
            "samples": [{"file": s["file"], "tokens": s["tokens"]} for s in stats["samples"]],
            "scales": {k: {"multiplier": arm["multiplier"][k], "amax": arm["amax"][k],
                           "clip_fraction": arm["clip_fraction"][k]}
                      for k in sorted(arm["multiplier"])},
        }
        out = args.out_prefix.with_name(args.out_prefix.name + f"-{name}.json")
        out.write_text(json.dumps(record, indent=2) + "\n")
        worst = max(arm["clip_fraction"].values())
        print(json.dumps({"arm": name, "out": str(out), "projections": len(arm["multiplier"]),
                          "worst_clip_fraction": worst}))


if __name__ == "__main__":
    main()
