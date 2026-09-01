#!/usr/bin/env bash
# tpd-battery CELL 2 — THE SERVED PP-3 CALIBRATION ROW (timed; caller holds the marker).
# MANDATORY per ep-diet LANE §5.1 and the tp2-battery instrument trap: the engine twin
# under-reads (single-engine walks ~0.8x of their served class; PP arms cannot be priced by
# twins AT ALL — the naive per-step driver read 0.254x of served on the PP-3 placement).
# So the PP-3 comparator in the pricing table is a SERVED row, measured on THIS build, and it
# has to reproduce the banked plain-served baseline (35.36 decode-pool tok/s, tp2-battery
# cell 4) before any diet/PP-3 comparison is stated.
#
# Plain boot, NO extras (byte-identical serving env to the banked calibration boot), then
# run_pool.py timed x1 — which carries its own vendor-default sampled row (serving law).
set -uo pipefail
OUT=/root/out-tpd
mkdir -p "$OUT/served-cal"
echo "######## C2 SERVED PP-3 CALIBRATION BOOT (plain, no extras) ########"
bash "$OUT/serve.sh" start cal || { echo "C2SERVED_EXIT=BOOTFAIL"; exit 1; }
python3 "$OUT/run_pool.py" timed --out "$OUT/served-cal" || { echo "C2SERVED_EXIT=TIMEDFAIL"; bash "$OUT/serve.sh" stop; exit 1; }
bash "$OUT/serve.sh" engage cal || echo "C2SERVED_WARN=engage-red (see logs/boot-cal.engage)"
bash "$OUT/serve.sh" stop
echo "=== C2 SERVED CALIBRATION SUMMARY (vs banked 35.36 pool / 29.99 deep / 0.42s / 2.21s) ==="
python3 - "$OUT/served-cal/timed.json" <<'PY' | tee "$OUT/analysis/verdict-c2-served.txt"
import json, statistics as st, sys
d = json.load(open(sys.argv[1]))
pool = [r for r in d.get("pool_rows", []) if not r.get("err") and r["completion_tokens"] >= 128]
print(f"served PP-3 pool rows n={len(pool)}")
print(f"  decode tok/s median = {st.median(r['decode_tok_s'] for r in pool):.3f}   (banked 35.36)")
print(f"  TTFT median         = {st.median(r['ttft_s'] for r in pool):.3f} s (banked 0.42 s @0.4-0.5k)")
for r in d.get("deep_ttft", []):
    print(f"  deep {str(r.get('tag')):10} ttft={r.get('ttft_s')} decode_tok_s={r.get('decode_tok_s')} "
          f"ct={r.get('completion_tokens')} err={r.get('err')}   (banked deep 29.99 tok/s, TTFT 2.21 s)")
v = d.get("vendor_row")
if v:
    print(f"  VENDOR-DEFAULT sampled row: {v}")
else:
    print("  VENDOR ROW MISSING — serving-law twin absent, report it")
excl = [r for r in d.get("pool_rows", []) if r.get("err") or r["completion_tokens"] < 128]
if excl:
    print("EXCLUDED (128-token floor / error), named:", [(r['tag'], r['completion_tokens'], r.get('err')) for r in excl])
PY
python3 "$OUT/looplaw_screen.py" "$OUT/served-cal" | tee -a "$OUT/analysis/verdict-c2-served.txt"
echo "C2SERVED_DONE"
