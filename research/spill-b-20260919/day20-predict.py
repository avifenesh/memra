#!/usr/bin/env python3
"""Offline prediction for the day-20 replay: the day-16 shape moved onto the 32-token GDN prime grid
(cohort 1248/1344/1440/1536 ids, loop 10,912 + 160 per turn, budget 1024 MiB), written before the runs.

Reuses day16-predict.py's model of worker.rs (day15-predict.py plus the restore-fence lease correction
day 15 measured) with the exact day-15 `/metrics` entry bytes (156,893,184 B fixed + 29,696 B/token).
Every prompt of the day-20 shape is a multiple of 32, so under the grid-aligned capture
(`seed_capture_boundary`, memra#602) every seed publishes at the prompt end and every restored length
equals the previous prompt: the model's cached_tokens arithmetic is unchanged. Prediction only: the
live runs are authoritative; the numbers are recorded so a deviation is visible.

usage: day20-predict.py [--day16] [-v]
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("day16_predict", HERE / "day16-predict.py")
p16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p16)

SHAPES = dict(p16.SHAPES)
SHAPES["day20"] = {"budget": 1024 << 20, "cohort": [1248, 1344, 1440, 1536], "start": 10912, "grow": 160}
GRID = 32


def predict(shape):
    for n in shape["cohort"] + [shape["start"], shape["grow"]]:
        assert n % GRID == 0, f"{n} is not on the {GRID}-token grid"
    return p16.predict(shape)


def main():
    name = "day16" if "--day16" in sys.argv else "day20"
    shape = SHAPES[name]
    e = p16.p15.entry_bytes
    last = shape["start"] + (p16.TURNS - 1) * shape["grow"]
    b = shape["budget"]
    print(f"{name}: budget={b} protected80={b * 80 // 100} cohort_bytes={sum(e(n) for n in shape['cohort'])} "
          f"e1={e(shape['start'])} e_last={e(last)} two_last={e(last) + e(last - shape['grow'])} "
          f"shares cohort={sum(e(n) for n in shape['cohort']) / b:.3f} e1={e(shape['start']) / b:.3f} e_last={e(last) / b:.3f}")
    out = predict(shape)
    for arm, r in out.items():
        print(f"== {arm}: computed={r['computed']} loop_cold_after_1={r['loop_cold_after_1']} return_cached_sum={r['return_cached']} evictions={r['evictions']}")
        for s, t, p, k, comp in r["rows"]:
            print(f"  {s:10s} {t:9s} prompt={p:6d} cached={k:6d} computed={comp:6d}")
        if "-v" in sys.argv:
            print("\n".join("   " + ln for ln in r["log"]))
    a, bb = out["slru"]["computed"], out["lru"]["computed"]
    print(f"primary: slru={a} lru={bb} diff(slru-lru)={a - bb}")


if __name__ == "__main__":
    main()
