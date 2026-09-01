#!/usr/bin/env bash
set -uo pipefail
OUT=/root/out-flip3/c4
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1)
for arm in plain k2; do
  case $arm in
    plain) extras=() ;;
    k2)    extras=("${DFL[@]}" MEMRA_SPEC_K=2) ;;
  esac
  /root/out-flip3/serve.sh start "twin-$arm" ${extras[@]+"${extras[@]}"} || { echo "TWIN_${arm}_EXIT=BOOTFAIL"; continue; }
  python3 /root/out-flip3/run_pool.py sample --out "$OUT/twin-$arm" || { echo "TWIN_${arm}_EXIT=SAMPLEFAIL"; continue; }
  python3 /root/out-flip3/run_pool.py twin --out "$OUT/twin-$arm" --max-tokens 128
  echo "TWIN_${arm}_EXIT=0"
done
/root/out-flip3/serve.sh stop
echo "C4C_ALL_DONE"
