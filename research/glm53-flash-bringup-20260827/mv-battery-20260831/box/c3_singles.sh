#!/usr/bin/env bash
# MV-DOORS CELL 3 — per-door single-flag attribution rows (one boot each on the SHIP
# spec config, untimed-precision: single-boot rows are attribution evidence, the c2
# composed number is the claim). Each boot: sample gate, DOOR gate (own announce
# demanded, other four FORBIDDEN — isolation), timed pool (tok/s delta vs the c2 OFF
# medians), byte identity vs the c2 off-1 tapes (each door alone must byte-hold on the
# served shape too — all five are bit-gated on the rig).
set -uo pipefail
OUT=/root/out-mv/c3
DFL=(MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1 MEMRA_SPEC_PMIN=0.7)
mkdir -p "$OUT"

one_door() {  # name, env
  local name="$1" flag="$2"
  echo "######## C3 BOOT $name ($flag) ########"
  /root/out-mv/serve.sh start "d-$name" "${DFL[@]}" "$flag" || { echo "D_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-mv/run_pool.py sample --out "$OUT/$name-1" || { echo "D_${name}_EXIT=SAMPLEFAIL"; return 1; }
  /root/out-mv/serve.sh doors "d-$name" "${DFL[@]}" "$flag" || { echo "D_${name}_EXIT=DOORFAIL"; return 1; }
  python3 /root/out-mv/run_pool.py timed --out "$OUT/$name-1" --max-tokens 256
  echo "=== IDENTITY $name vs c2 off-1 ==="
  python3 /root/out-mv/run_pool.py compare --a /root/out-mv/c2/off-1 --b "$OUT/$name-1" \
    || echo "D_${name}_IDENTITY_DIVERGENCE — flag it loud"
  echo "D_${name}_EXIT=0"
}

rc=0
one_door twide  MEMRA_BF16_TCOLS_WIDE=1 || rc=1
one_door mpack  MEMRA_MOE_VROWS_PACK=1  || rc=1
one_door xone   MEMRA_BF16_TCOLS_X1=1   || rc=1
one_door kshard MEMRA_TOPK_SHARDS=1     || rc=1
one_door wws    MEMRA_GLM5_VERIFY_WS=1  || rc=1
/root/out-mv/serve.sh stop
echo "=== LOOP-LAW SCREEN (all c3 tapes) ==="
python3 /root/out-mv/looplaw_screen.py "$OUT"/*/
echo "=== per-door summary vs c2 off (single-boot rows; deltas are attribution, not the claim) ==="
# link the c2 OFF boots in as the baseline so mv_check prices each door against them
for d in /root/out-mv/c2/off-*; do ln -sfn "$d" "$OUT/$(basename "$d")"; done
python3 /root/out-mv/mv_check.py --base "$OUT" --baseline off --arms twide,mpack,xone,kshard,wws || true
echo "C3_DONE rc=$rc"
exit "$rc"
