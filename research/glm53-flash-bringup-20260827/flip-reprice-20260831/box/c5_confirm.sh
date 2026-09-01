#!/usr/bin/env bash
# CELL 5 CONFIRM (timed, marker held): x3 interleaved plain vs THE DEPLOYABLE CONFIG —
# DFlash2 + auto K policy (nopin) + MEMRA_SPEC_PMIN=0.7 on the batched walk. This is the
# exact env a flip would ship; the tau arms above were K-pinned single boots.
set -uo pipefail
OUT=/root/out-flip3/c5
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1 MEMRA_SPEC_PMIN=0.7)
mkdir -p "$OUT"
for i in 1 2 3; do
  for arm in plain ship; do
    case $arm in
      plain) extras=() ;;
      ship)  extras=("${DFL[@]}") ;;
    esac
    /root/out-flip3/serve.sh start "c5-$arm$i" ${extras[@]+"${extras[@]}"} || { echo "C5_$arm${i}_EXIT=BOOTFAIL"; continue; }
    python3 /root/out-flip3/run_pool.py sample --out "$OUT/$arm$i" || { echo "C5_$arm${i}_EXIT=SAMPLEFAIL"; continue; }
    python3 /root/out-flip3/run_pool.py timed --out "$OUT/$arm$i" --max-tokens 256
    if [ "$arm" = ship ]; then
      /root/out-flip3/serve.sh walk "c5-$arm$i" batched || echo "C5_$arm${i}_WALK=RED"
      grep -m1 -iE "confidence gate" /root/out-flip3/logs/boot-c5-$arm$i.log || true
      grep -m1 "route=spec" /root/out-flip3/logs/boot-c5-$arm$i.log || true
    fi
    echo "C5_$arm${i}_EXIT=0"
  done
done
# the twin on the shipping config (multi-turn law receipt on the recommended arm)
/root/out-flip3/serve.sh start "c5-twin-ship" "${DFL[@]}" && {
  python3 /root/out-flip3/run_pool.py sample --out "$OUT/twin-ship"
  python3 /root/out-flip3/run_pool.py twin --out "$OUT/twin-ship" --max-tokens 128
  echo "C5_twin_EXIT=0"
}
/root/out-flip3/serve.sh stop
python3 /root/out-flip3/looplaw_screen.py "$OUT"/*/
echo "C5_ALL_DONE"
