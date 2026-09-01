#!/usr/bin/env python3
"""tp2-battery exactness compare (stdlib only).

usage: compare.py A_DIR B_DIR
  *.ids / *.txt  -> byte identity (the decode bar)
  *.f32          -> elementwise max_rel over f32 LE dumps (the prime/near-tie band bar),
                    plus byte-identity flag (a band row that is also byte-identical is
                    reported as such — calibration law: measure first)
exit 0 = all ids/txt byte-identical AND every f32 within BAND (default 2e-4, env BAND);
the caller decides what the bar IS per regime — this tool only reports.
"""
import os, struct, sys

band = float(os.environ.get("BAND", "2e-4"))
a_dir, b_dir = sys.argv[1], sys.argv[2]
shared = sorted(
    set(os.listdir(a_dir)) & set(os.listdir(b_dir))
)
fail = 0
worst = (0.0, None, None)
for f in shared:
    pa, pb = os.path.join(a_dir, f), os.path.join(b_dir, f)
    if not (os.path.isfile(pa) and os.path.isfile(pb)):
        continue
    da, db = open(pa, "rb").read(), open(pb, "rb").read()
    if f.endswith((".ids", ".txt")):
        same = da == db
        print(f"[cmp] {f}: {'IDENTICAL' if same else 'DIFFERS'} ({len(da)} vs {len(db)} bytes)")
        if not same:
            fail = 1
    elif f.endswith(".f32"):
        if da == db:
            print(f"[cmp] {f}: BYTE-IDENTICAL ({len(da)//4} f32)")
            continue
        if len(da) != len(db):
            print(f"[cmp] {f}: LENGTH MISMATCH {len(da)} vs {len(db)}")
            fail = 1
            continue
        n = len(da) // 4
        va = struct.unpack(f"<{n}f", da)
        vb = struct.unpack(f"<{n}f", db)
        mx, mi = 0.0, -1
        for i in range(n):
            x, y = va[i], vb[i]
            d = abs(x - y)
            if d == 0.0:
                continue
            r = d / max(abs(x), abs(y), 1e-30)
            if r > mx:
                mx, mi = r, i
        ok = mx <= band
        print(f"[cmp] {f}: max_rel={mx:.3e} at idx {mi} (band {band:.0e}) {'OK' if ok else 'OVER'}")
        if mx > worst[0]:
            worst = (mx, f, mi)
        if not ok:
            fail = 1
print(f"[cmp] shared files={len(shared)} worst_f32_rel={worst[0]:.3e} ({worst[1]}) fail={fail}")
sys.exit(fail)
