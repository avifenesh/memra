#!/usr/bin/env bash
# DECODE-DIET CELLS 2-5 — per-door single-flag attribution rows (one boot each,
# untimed-precision: single-boot rows are attribution evidence, the composed number is
# the claim). Each boot: sample gate, DOOR gate (own announce demanded, other three
# FORBIDDEN — isolation), timed pool (tok/s delta vs the cell-1 OFF medians), identity
# vs the cell-1 off-1 tapes (each door alone must also be byte-preserving).
set -uo pipefail
OUT=/root/out-diet/c2to5
mkdir -p "$OUT"

one_door() {  # name, env
  local name="$1" flag="$2"
  echo "######## C2-5 BOOT $name ($flag) ########"
  /root/out-diet/serve.sh start "d-$name" "$flag" || { echo "D_${name}_EXIT=BOOTFAIL"; return 1; }
  python3 /root/out-diet/run_pool.py sample --out "$OUT/$name-1" || { echo "D_${name}_EXIT=SAMPLEFAIL"; return 1; }
  /root/out-diet/serve.sh doors "d-$name" "$flag" || { echo "D_${name}_EXIT=DOORFAIL"; return 1; }
  python3 /root/out-diet/run_pool.py timed --out "$OUT/$name-1" --max-tokens 256
  echo "=== IDENTITY $name vs c1 off-1 ==="
  python3 /root/out-diet/run_pool.py compare --a /root/out-diet/c1/off-1 --b "$OUT/$name-1" \
    || echo "D_${name}_IDENTITY_DIVERGENCE — flag it loud"
  echo "D_${name}_EXIT=0"
}

rc=0
one_door hcpre    MEMRA_HC_FUSED_PRE=1     || rc=1
one_door hcws     MEMRA_HC_DECODE_WS=1     || rc=1
one_door kda6     MEMRA_KDA_FUSED_PROJ=1   || rc=1
one_door mlasplit MEMRA_MLA_DECODE_SPLIT=1 || rc=1
/root/out-diet/serve.sh stop
echo "=== LOOP-LAW SCREEN (all c2to5 tapes) ==="
python3 /root/out-diet/looplaw_screen.py "$OUT"/*/
echo "=== per-door summary vs c1 off (single-boot rows; deltas are attribution, not the claim) ==="
# link the cell-1 OFF boots in as the baseline so diet_check prices each door against them
for d in /root/out-diet/c1/off-*; do ln -sfn "$d" "$OUT/$(basename "$d")"; done
python3 /root/out-diet/diet_check.py --base "$OUT" --baseline off --arms hcpre,hcws,kda6,mlasplit || true
echo "C2TO5_DONE rc=$rc"
exit "$rc"
