#!/usr/bin/env bash
# Runs ON gb10-b: memra-server for Qwen3.8-27B under a memory guard (killed if MemAvailable stays
# under 3 GiB for two checks 2 s apart). Derived unified-memory defaults, no hand-set prefix/TTFT.
#   ARM=<arm> BIN=~/memra-run/bin-<sha12> LEASE=<sparklock lease> serve-arm.sh
# Arms (GGUF = memra's NVFP4+Q5_K trunk; NV = NVIDIA's NVFP4 safetensors dir):
#   gguf-mtpmask         GGUF + masked (sxc32768) GGUF MTP draft, documented K table
#   gguf-mtpmask-nobook  as gguf-mtpmask with MEMRA_UNIFIED_PREFIX_BOOKING=0 (insert guard only)
#   gguf-mtpmask-k2      as gguf-mtpmask with MEMRA_SPEC_K_CACHED_LONG_TRIM=2
#   gguf-dflash          GGUF + DFlash2 drafter + FR-Spec trim + restore/capture flags, MEMRA_MTP_SKIP=1
#   gguf-dflash-noskip   as gguf-dflash with the MTP head loaded (receipt boot only)
#   dflash               NV + DFlash2 drafter + FR-Spec trim, MEMRA_MTP_SKIP=1
#   mtpmask              NV + FR-Spec-trimmed MTP head, documented K table
# Extra env: EXTRA="K=V K2=V2" is exported last (tuning arms). Log: ~/memra-run/serve-$ARM.log
set -u
ARM=${ARM:?arm}
BIN=${BIN:?bin dir}
D=$HOME/memra-run
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF
NV=$HOME/models/Qwen3.8-27B-NVFP4
DF=$HOME/models/memra-q38/Qwen3.8-27B-DFlash2-memra
RANKS="$G8/q38-ranks-sxc32768.gguf.txt"
export SPARKLOCK_LEASE=${LEASE:-}
export MEMRA_COMPAT=openai
export MEMRA_MODELS="q38=$NV"
export MEMRA_VISION_DIR="$NV"
export MEMRA_ADDR=127.0.0.1:8090
export MEMRA_CTX=262144
export MEMRA_MAX_SESSIONS=32
export MEMRA_KV_HOST_MB=0
GGUF="q38=$G8/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
MASK="$G8/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf"
case "$ARM" in
  gguf-mtpmask) export MEMRA_MODELS="$GGUF" MEMRA_MTP_DRAFT="$MASK" MEMRA_PREFIX_STABLE_BOUNDARY=1 ;;
  gguf-mtpmask-nobook) export MEMRA_MODELS="$GGUF" MEMRA_MTP_DRAFT="$MASK" MEMRA_PREFIX_STABLE_BOUNDARY=1 \
        MEMRA_UNIFIED_PREFIX_BOOKING=0 ;;
  gguf-mtpmask-k2) export MEMRA_MODELS="$GGUF" MEMRA_MTP_DRAFT="$MASK" MEMRA_PREFIX_STABLE_BOUNDARY=1 \
        MEMRA_SPEC_K_CACHED_LONG_TRIM=2 ;;
  gguf-dflash) export MEMRA_MODELS="$GGUF" MEMRA_MTP_SKIP=1 MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT="$DF" \
        MEMRA_FRSPEC_TRIM="$RANKS" MEMRA_DSPARK_PREFIX_RESTORE=1 MEMRA_DSPARK_BOUNDARY_CAPTURE=1 \
        MEMRA_DSPARK_PARTIAL_RESTORE=1 MEMRA_PREFIX_STABLE_BOUNDARY=1 ;;
  gguf-dflash-noskip) export MEMRA_MODELS="$GGUF" MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT="$DF" \
        MEMRA_FRSPEC_TRIM="$RANKS" MEMRA_DSPARK_PREFIX_RESTORE=1 MEMRA_DSPARK_BOUNDARY_CAPTURE=1 \
        MEMRA_DSPARK_PARTIAL_RESTORE=1 MEMRA_PREFIX_STABLE_BOUNDARY=1 ;;
  dflash) export MEMRA_MTP_SKIP=1 MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT="$DF" MEMRA_FRSPEC_TRIM="$RANKS" ;;
  mtpmask) export MEMRA_FRSPEC_TRIM="$RANKS" ;;
  *) echo "unknown ARM $ARM" >&2; exit 2 ;;
esac
for kv in ${EXTRA:-}; do export "$kv"; done
G=$D/guard-$ARM.log
{ echo "$(date '+%F %T') start ARM=$ARM BIN=$BIN"; sha256sum "$BIN/memra-server"; env | grep '^MEMRA_' | sort; } > "$G"
"$BIN/memra-server" > "$D/serve-$ARM.log" 2>&1 &
pid=$!
echo $pid > "$D/serve-$ARM.pid"
echo $$ > "$D/guard-$ARM.pid"
low=0; minv=999999999
while kill -0 $pid 2>/dev/null; do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  [ "$a" -lt "$minv" ] && minv=$a
  if [ "$a" -lt 3145728 ]; then low=$((low+1)); else low=0; fi
  if [ $low -ge 2 ]; then echo "$(date '+%T') GUARD_KILL MemAvailable ${a}kB < 3GiB twice" >> "$G"; kill -TERM $pid; sleep 5; kill -KILL $pid 2>/dev/null; break; fi
  if [ $((SECONDS % 300)) -lt 2 ]; then echo "$(date '+%T') MemAvailable now ${a}kB min ${minv}kB" >> "$G"; fi
  sleep 2
done
wait $pid; rc=$?
echo "$(date '+%F %T') server exit rc=$rc min MemAvailable ${minv}kB" >> "$G"
