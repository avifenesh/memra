#!/usr/bin/env bash
# STRUCT-BATTERY CELL 2 — THE DEDUP STAT (untimed, count-based; moe-loc LANE §1.6/§4.6):
# ONE instrument boot on the ship recipe with MEMRA_MOE_VROWS_DEDUP_STAT=1 and door D
# pinned =0 (the instrument REQUIRES the host table-build arm), real pools through the
# served path, banking every [moe-vrows-dedup] line. The cumulative counters are
# snapshotted after each pool phase so per-phase deltas (greedy decode pool / l3 deep /
# vendor-default sampled) are recoverable. NO timing number leaves this boot (the
# instrument adds a HashSet per layer-call by design).
set -uo pipefail
OUT=/root/out-struct/c2
LOG=/root/out-struct/logs/boot-c2-dedup.log
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1 MEMRA_SPEC_PMIN=0.7)
STAT=(MEMRA_MOE_VROWS_DEDUP_STAT=1 MEMRA_MOE_VROWS_DEV_TABLES=0)
mkdir -p "$OUT"

snap() { # label
  local last; last=$(grep -- '\[moe-vrows-dedup\]' "$LOG" | tail -1)
  echo "PHASE_SNAP $1: ${last:-<none>}" | tee -a "$OUT/phase-snaps.txt"
}

/root/out-struct/serve.sh start "c2-dedup" "${DFL[@]}" "${STAT[@]}" || { echo "C2_EXIT=BOOTFAIL"; exit 1; }
python3 /root/out-struct/run_pool.py sample --out "$OUT" || { echo "C2_EXIT=SAMPLEFAIL"; exit 1; }
/root/out-struct/serve.sh engage "c2-dedup" "${DFL[@]}" "${STAT[@]}" || { echo "C2_EXIT=ENGAGEFAIL"; exit 1; }
snap "post-sample"
python3 /root/out-struct/run_pool.py cell --out "$OUT/greedy" --pool both --mode greedy --max-tokens 256
snap "post-greedy-pools"
python3 /root/out-struct/run_pool.py cell --out "$OUT/vendor" --pool decode --mode vendor --max-tokens 256
snap "post-vendor-pool"
grep -- '\[moe-vrows-dedup\]' "$LOG" > "$OUT/dedup-lines.txt"
wc -l "$OUT/dedup-lines.txt"
/root/out-struct/serve.sh stop
echo "=== LOOP-LAW SCREEN (c2 tapes; greedy-loop rows would bias routing counts) ==="
python3 /root/out-struct/looplaw_screen.py "$OUT/greedy" "$OUT/vendor"
echo "=== C2 DEDUP VERDICT ARITHMETIC (cumulative + per-phase deltas) ==="
python3 - "$OUT/phase-snaps.txt" "$OUT/dedup-lines.txt" <<'PY'
import re, sys
def parse(line):
    m = re.search(r"layer-calls=(\d+)\s+visits=(\d+)\s+distinct=(\d+)", line)
    return tuple(int(x) for x in m.groups()) if m else None
snaps = []
for line in open(sys.argv[1]):
    p = parse(line)
    if p: snaps.append((line.split(":")[0].replace("PHASE_SNAP ", ""), *p))
lines = [parse(l) for l in open(sys.argv[2])]
lines = [l for l in lines if l]
if lines:
    lc, v, d = lines[-1]
    print(f"FINAL cumulative: layer-calls={lc} visits={v} distinct={d} "
          f"repeat_fraction={1 - d / v:.4f} ({100 * (1 - d / v):.2f}%)")
prev = None
for name, lc, v, d in snaps:
    if prev is not None:
        dv, dd = v - prev[1], d - prev[2]
        if dv > 0:
            print(f"PHASE {name}: dvisits={dv} ddistinct={dd} repeat={1 - dd / dv:.4f} ({100 * (1 - dd / dv):.2f}%)")
    print(f"SNAP {name}: layer-calls={lc} visits={v} distinct={d} repeat={1 - d / v:.4f}")
    prev = (lc, v, d)
print("bounds for the verdict (moe-loc LANE §1.6): independent-routing 3.2% -> +0.9% ship; "
      "10% -> +2.7%; 20% -> +5.6%; 33% -> +9.5%; 70.1% structural ceiling -> +22.7%")
PY
echo "C2_DONE"
