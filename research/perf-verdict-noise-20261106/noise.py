#!/usr/bin/env python3
"""Run-to-run noise of the cell-evidence perf A/B and the false-alarm rate and power of its
REGRESSED verdict, from banked records.

    python3 noise.py extract RECORD_DIR... > runs.jsonl   # per-run decode rates
    python3 noise.py analyze runs.jsonl > noise.out

Each within-arm triple of a record is three runs of one binary in one interleaved window.
Its residuals x/mean - 1, scaled by sqrt(3/2) to undo the shrinkage of centering three
values, are draws of the per-run noise of that device. Resampling six of them as the
A A A and B B B arms of one program gives the verdict's false-alarm rate on identical code;
shifting the B draws down by R percent gives its power for a true R percent regression.
The verdict is tools/cell_evidence.py judge_perf: complete separation (max B < min A) and,
for T > 0, a median drop of at least T percent.
"""
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

TPS = re.compile(r"generated (\d+) tokens in ([0-9.]+)s = ([0-9.]+) tok/s")
THRESHOLDS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
REGRESSIONS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0)


def extract(dirs):
    for arg in dirs:
        root = Path(arg)
        record = json.loads((root / "record.json").read_text())
        perf = record["kinds"].get("perf")
        if not perf:
            continue
        runs = []
        for run in perf["runs"]:
            found = TPS.findall((root / run["log"]).read_text(errors="replace"))
            runs.append({"arm": run["arm"], "model": run["model"], "tps": float(found[-1][2])})
        print(json.dumps({"record": "/".join(root.resolve().parts[-2:]), "device": record["device"]["class"],
                          "tested_commit": record["tested_commit"], "base_commit": perf.get("base_commit"),
                          "baseline": perf.get("baseline"), "runs": runs}))


def triples(path):
    by_device = {}
    for line in open(path):
        record = json.loads(line)
        arms = {}
        for run in record["runs"]:
            arms.setdefault((run["model"], run["arm"]), []).append(run["tps"])
        for xs in arms.values():
            if len(xs) == 3:
                by_device.setdefault(record["device"], []).append(xs)
    return by_device


def regressed(a, b, t):
    return max(b) < min(a) and (statistics.median(b) / statistics.median(a) - 1.0) * 100.0 <= -t


def analyze(path, n_false=400_000, n_power=100_000):
    rng = random.Random(20261106)
    pairs = []
    for line in open(path):
        record = json.loads(line)
        if record["baseline"] != "interleaved":
            continue
        for model in sorted({r["model"] for r in record["runs"]}):
            a = [r["tps"] for r in record["runs"] if r["model"] == model and r["arm"] == "A"]
            b = [r["tps"] for r in record["runs"] if r["model"] == model and r["arm"] == "B"]
            pairs.append((record["device"], model, (statistics.median(b) / statistics.median(a) - 1) * 100,
                          max(b) < min(a)))
    print(f"recorded base vs candidate pairs: {len(pairs)}; separated: {sum(p[3] for p in pairs)}; "
          f"largest |median delta|: {max(abs(p[2]) for p in pairs):.2f}%")
    for device, model, delta, sep in pairs:
        print(f"  {device:8s} {model:32s} {delta:+.2f}% separated={sep}")
    for device, ts in sorted(triples(path).items()):
        resid = [(x / statistics.fmean(xs) - 1.0) * math.sqrt(1.5) for xs in ts for x in xs]
        raw = max(abs(x / statistics.fmean(xs) - 1.0) for xs in ts for x in xs) * 100
        print(f"\n{device}: {len(ts)} triples, per-run relative sd {statistics.pstdev(resid) * 100:.3f}%, "
              f"largest single-run deviation from its arm mean {raw:.2f}%")
        print(f"  false alarms on identical code ({n_false} resampled A/Bs):")
        for t in THRESHOLDS:
            hits = sum(regressed([1 + rng.choice(resid) for _ in range(3)], [1 + rng.choice(resid) for _ in range(3)], t)
                       for _ in range(n_false))
            p = hits / n_false
            print(f"    T={t:3.1f}%  per model {p:.5f}  per two-model head {1 - (1 - p) ** 2:.5f}")
        print(f"  power for a true R% regression ({n_power} resampled A/Bs per cell):")
        for t in (0.0, 1.5, 2.0, 2.5, 3.0):
            row = []
            for r in REGRESSIONS:
                hits = sum(regressed([1 + rng.choice(resid) for _ in range(3)],
                                     [(1 - r / 100) * (1 + rng.choice(resid)) for _ in range(3)], t)
                           for _ in range(n_power))
                row.append(f"R={r:.0f}%:{hits / n_power:.3f}")
            print(f"    T={t:3.1f}%  " + "  ".join(row))


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2:])
    else:
        analyze(sys.argv[2])
