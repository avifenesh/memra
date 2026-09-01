#!/usr/bin/env bash
# CELL 6 — K sweep on the winning spec arm (arg $1 = dfl|nat), K in {2,3,5,7}.
# GATED: run ONLY if a spec arm beat plain in cell 4. Each K is a fresh boot (K is a boot pin).
# Timed: decode tok/s on both pools + the deep TTFT rows + one vendor-default row, exactly the
# cell-4 shape so the winning K is comparable to the cell-4 table row.
# Marker: the caller holds /root/TIMING-IN-FLIGHT for this whole cell.
set -uo pipefail
ARM="${1:-dfl}"
OUT=/root/out-3way/c6
mkdir -p "$OUT"
case "$ARM" in
  dfl) SPEC=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1) ;;
  nat) SPEC=(MEMRA_GLM5_MTP=1 MEMRA_GLM5_SPEC=1) ;;
  *) echo "usage: c6_ksweep.sh dfl|nat"; exit 2 ;;
esac

for K in 2 3 5 7; do
  name="c6-$ARM-k$K"
  echo "######## C6 BOOT $name ########"
  /root/out-3way/serve.sh start "$name" "${SPEC[@]}" MEMRA_SPEC_K="$K" \
    || { echo "C6_k${K}_EXIT=BOOTFAIL"; continue; }
  python3 /root/out-3way/run_pool.py sample --out "$OUT/k$K" || { echo "C6_k${K}_EXIT=SAMPLEFAIL"; continue; }
  python3 /root/out-3way/run_pool.py timed --out "$OUT/k$K" --max-tokens 256
  log=/root/out-3way/logs/boot-$name.log
  echo "K receipt:"; grep -m1 -E '\[glm5-spec\] route=spec K=|clamped to' "$log" || true
  grep -m1 'clamped to' "$log" || true
  echo "C6_k${K}_EXIT=0"
done

/root/out-3way/serve.sh stop
echo "=== LOOP-LAW SCREEN ==="
python3 /root/out-3way/looplaw_screen.py "$OUT"/*/
echo "=== K SWEEP TABLE ==="
python3 - <<'PY'
import json, glob, os, statistics as st
def med(xs):
    xs=[x for x in xs if x is not None]
    return st.median(xs) if xs else None
print(f"{'K':>3} {'dec tok/s':>10} {'deep tok/s':>11} {'ttft0.4k':>9} {'ttft3.7k':>9} "
      f"{'acc/cyc':>8} {'tok/cyc':>8} {'vendor t/s':>11}")
for d in sorted(glob.glob("/root/out-3way/c6/k*"), key=lambda p: int(os.path.basename(p)[1:])):
    t=json.load(open(os.path.join(d,"timed.json")))
    dec=[r for r in t["pool_rows"] if r["kind"]!="l3deep" and not r["err"]]
    dp=[r for r in t["pool_rows"] if r["kind"]=="l3deep" and not r["err"]]
    specs=[r["spec"] for r in t["pool_rows"] if r.get("spec")]
    acc=sum(s["accepted"] for s in specs); rnd=sum(s["rounds"] for s in specs)
    apc=acc/rnd if rnd else None
    dt={r["tag"]:r["ttft_s"] for r in t["deep_ttft"]}
    v=(t["vendor_row"] or {}).get("decode_tok_s")
    f=lambda x,n=3: "n/a" if x is None else f"{x:.{n}f}"
    print(f"{os.path.basename(d)[1:]:>3} {f(med([r['decode_tok_s'] for r in dec]),2):>10} "
          f"{f(med([r['decode_tok_s'] for r in dp]),2):>11} {f(dt.get('l3-WARM')):>9} "
          f"{f(dt.get('l3-A4630')):>9} {f(apc):>8} {f(apc+1 if apc else None):>8} {f(v,1):>11}")
PY
echo "C6_ALL_DONE"
