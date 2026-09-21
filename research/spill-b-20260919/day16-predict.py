#!/usr/bin/env python3
"""Offline prediction for the day-16 replay (the day-15 shape scaled to a 1024 MiB budget), written
before the runs. Reuses day15-predict.py's model of worker.rs with the ONE correction day 15 measured:
the restore lease is released at the restore fence, BEFORE the new entry is published, so the hit entry
is the older of the two when the newcomer's publication needs room. Entry bytes are the exact day-15
`/metrics` deltas (156,893,184 B fixed + 29,696 B/token), not the MB-rounded fit.

With that correction the model reproduces day 15's measured runs exactly (slru 132,300 / lru 122,700,
return cached 0 / 8,700, evictions 21 / 19) and predicts the scaled shape. Prediction only: the live runs
are authoritative; the numbers are recorded so a deviation is visible.

usage: day16-predict.py [--day15] [-v]
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("day15_predict", HERE / "day15-predict.py")
p15 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p15)
p15.PER_TOKEN = 29696.0
p15.FIXED = 156893184.0

SHAPES = {
    "day15": {"budget": 2048 << 20, "cohort": [7800, 8000, 8200, 8400], "start": 27300, "grow": 300},
    "day16": {"budget": 1024 << 20, "cohort": [1250, 1350, 1450, 1550], "start": 11000, "grow": 150},
}
TURNS, RETURN_EVERY = 12, 3


def request(self, tenant, prompt_len):
    hit = self.lookup(tenant, prompt_len)
    cached = 0
    if hit is not None:
        self.pin(hit)
        cached = hit["toks"]
        # Day 15's finding: the restore lease ends at the restore fence, before publication.
        self.unpin(hit)
    if not self.has_key(tenant, prompt_len):
        if self.prepare_snapshot(tenant, p15.entry_bytes(prompt_len)):
            self.insert(tenant, prompt_len)
    return cached


p15.Cache.request = request


def predict(shape):
    out = {}
    for arm in ("slru", "lru"):
        rows, c = p15.replay(arm == "slru", shape["budget"], shape["cohort"], shape["start"], shape["grow"], TURNS, RETURN_EVERY)
        out[arm] = {
            "computed": sum(p - k for _, _, p, k in rows),
            "return_cached": sum(k for s, _, _, k in rows if s.startswith("return") or s == "final"),
            "loop_cold_after_1": sum(1 for s, _, _, k in rows if s.startswith("loop") and s != "loop1" and k == 0),
            "evictions": c.evictions,
            "rows": [(s, t, p, k, p - k) for s, t, p, k in rows],
            "log": c.log,
        }
    return out


def main():
    name = "day15" if "--day15" in sys.argv else "day16"
    shape = SHAPES[name]
    e = p15.entry_bytes
    last = shape["start"] + (TURNS - 1) * shape["grow"]
    print(f"{name}: budget={shape['budget']} protected80={shape['budget'] * 80 // 100} cohort_bytes={sum(e(n) for n in shape['cohort'])} "
          f"e1={e(shape['start'])} e_last={e(last)} two_last={e(last) + e(last - shape['grow'])}")
    out = predict(shape)
    for arm, r in out.items():
        print(f"== {arm}: computed={r['computed']} loop_cold_after_1={r['loop_cold_after_1']} return_cached_sum={r['return_cached']} evictions={r['evictions']}")
        for s, t, p, k, comp in r["rows"]:
            print(f"  {s:10s} {t:9s} prompt={p:6d} cached={k:6d} computed={comp:6d}")
        if "-v" in sys.argv:
            print("\n".join("   " + ln for ln in r["log"]))
    a, b = out["slru"]["computed"], out["lru"]["computed"]
    print(f"primary: slru={a} lru={b} diff(slru-lru)={a - b}")


if __name__ == "__main__":
    main()
