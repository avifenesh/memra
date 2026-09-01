#!/usr/bin/env bash
# struct-battery engine-probe launcher (cell 5 — the placement A/B; adapted from
# tp2-battery-20260831/box/probe_arm.sh, same instrument = same trap survives: engine
# twins under-read served numbers, so the comparison is map-vs-even RELATIVE only).
# Cards 0,1 (TP-2 pair); port NONE — engine-level; foreground child of this shell.
set -uo pipefail
OUT=${OUT:-/root/out-struct}
BIN=${BIN:-/root/memra-struct/target/release/glm5-tp2-box-probe}
MODEL=/root/models/glm53-nvfp4
MAP=${MAP:-/root/out-struct/maps/agentic-t1-coactivation.json}
mkdir -p "$OUT/logs"

arm=$1; mode=$2; prompts=$3; run=$4; shift 4   # extra env as "$@"

# scrub inherited MEMRA_* (flip-battery discipline), then the arm table
unsets=()
while IFS='=' read -r k _; do case "$k" in MEMRA_*|BOXP_*) unsets+=(-u "$k");; esac; done < <(env)

common=(NVIDIA_TF32_OVERRIDE=0 MEMRA_BF16_MMV=1)
case "$arm" in
  even) armenv=(CUDA_VISIBLE_DEVICES=0,1 MEMRA_GLM5_TP=all@0,1 MEMRA_RP=0 \
                MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=16) ;;
  map)  armenv=(CUDA_VISIBLE_DEVICES=0,1 MEMRA_GLM5_TP=all@0,1 MEMRA_RP=0 \
                MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=16 MEMRA_GLM5_EP_MAP="$MAP") ;;
  *) echo "unknown arm $arm (even|map)"; exit 2 ;;
esac

name="$arm-$mode-$run"
log="$OUT/logs/probe-$name.log"
echo "[sbab] arm=$arm mode=$mode prompts=$prompts run=$run extras=$* bin=$BIN map=${MAP}" | tee "$log"
sha256sum "$BIN" | tee -a "$log"
[ "$arm" = map ] && sha256sum "$MAP" | tee -a "$log"
env "${unsets[@]}" "${common[@]}" "${armenv[@]}" BOXP_MODE="$mode" "$@" \
  "$BIN" "$MODEL" "$prompts" "$OUT/$name" 2>>"$log" | tee -a "$log"
rc=${PIPESTATUS[0]}
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$log"
# arm-identity receipts: the ep-map announce is DEMANDED on the map arm (with the map's
# sha256) and FORBIDDEN on the even arm; peer-slot dispatches are the counter the mint's
# peer_touch_fraction predicts — extracted here per run.
nmap=$(grep -c "ep-map armed" "$log" || true)
peer=$(grep -o "ep-peer-slot-dispatches=[0-9]*" "$log" | tail -1)
if [ "$arm" = map ]; then
  [ "$nmap" -ge 1 ] || { echo "[sbab] ARM-IDENTITY FAIL: map arm without 'ep-map armed'"; rc=1; }
  grep -m1 "ep-map armed" "$log"
else
  [ "$nmap" -eq 0 ] || { echo "[sbab] ARM-IDENTITY FAIL: even arm carries 'ep-map armed'"; rc=1; }
fi
echo "[sbab] rc=$rc arm=$arm mode=$mode run=$run ${peer:-ep-peer-slot-dispatches=<missing>}" | tee -a "$log"
exit $rc
