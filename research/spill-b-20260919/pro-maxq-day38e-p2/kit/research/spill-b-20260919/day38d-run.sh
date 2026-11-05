#!/usr/bin/env bash
# DAY38 addendum D boots (1.11; day38-run.sh with shape Xp in place of X): one boot per spec through run-day26-cell.sh, after a bounded idle
# wait (lock free, no compute app, >= 24 GB host memory). Never a signal to anything this lane did not
# start. usage: day38-run.sh <receipt root> <name>:<arm> ...
#   arm = off | on | on-fault-batch | on-fault-nobatch | on-fault-nobatch-red | vmm-off | vmm-on
# env: WT, RIG_LOCK (/tmp/memra-5090.lock), BIN, MODEL, MODEL_KEY, BOOT_CTX (65536; empty = the checkpoint's),
#      LENGTHS (6144,30720), NO_SCOPE.
# Max-Q kit port (kit/DEVIATIONS.md D1, D3, D6): the tip deleted MEMRA_KV_PARK_COMPACT and MEMRA_KV_ALLOCATOR,
# so the arms are binaries: off -> BIN_OFF, on and the green fault arms -> BIN_ON, the red fault arm -> RED_BIN.
# The vmm pair cannot run (no allocator door) and is refused. The caller (kit/run.sh) holds RIG_LOCK for the
# whole run, so idle() checks only the pinned card (SMI_CARD) and host memory, and the cell runs with LOCK=none.
set -uo pipefail
R=${1:?receipt root}; shift
WT=${WT:-$HOME/projects/wt-spill-b}
RIG_LOCK=${RIG_LOCK:-/tmp/memra-5090.lock}
SMI_CARD=${SMI_CARD:?SMI_CARD (the nvidia-smi index of the pinned card)}
BIN_OFF=${BIN_OFF:?BIN_OFF}
BIN_ON=${BIN_ON:?BIN_ON}
RED_BIN=${RED_BIN:?RED_BIN}
LENGTHS=${LENGTHS:-6144,30720}
IDLE_WAIT_S=${IDLE_WAIT_S:-600}
cd "$WT" || exit 1
mkdir -p "$R/boots"
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/run.log"; }
idle() {
  [ -z "$(nvidia-smi -i "$SMI_CARD" --query-compute-apps=pid --format=csv,noheader)" ] || return 1
  [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 24 ] || return 1
}
BOOT_CTX=${BOOT_CTX-65536}
if [ -n "$BOOT_CTX" ]; then export MEMRA_CTX=$BOOT_CTX; else unset MEMRA_CTX; fi
export LOCK=none RIGDIR="$R/boots" N=5 CLIENT=day38-client.py PARSER=day38-parse.py MEMRA_TIMEOUT_MS_MAX=3600000 \
  MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=0
for spec in "$@"; do
  IFS=: read -r name arm <<< "$spec"
  aenv=(-u MEMRA_KV_PARK_COMPACT -u MEMRA_KV_ALLOCATOR -u MEMRA_STEP_OOM_FAULT -u MEMRA_SERVE_BATCH)
  B=$BIN_ON
  args="--lengths $LENGTHS --shapes Xp,A"
  case $arm in
    off) B=$BIN_OFF ;;
    on) ;;
    on-fault-batch) aenv+=(MEMRA_STEP_OOM_FAULT=1); args="--lengths 6144 --shapes Xp --turn4" ;;
    on-fault-nobatch) aenv+=(MEMRA_STEP_OOM_FAULT=1 MEMRA_SERVE_BATCH=0); args="--lengths 6144 --shapes Xp --turn4" ;;
    on-fault-nobatch-red) aenv+=(MEMRA_STEP_OOM_FAULT=1 MEMRA_SERVE_BATCH=0); args="--lengths 6144 --shapes Xp --turn4"; B=$RED_BIN ;;
    vmm-off|vmm-on) log "boot $name: arm $arm needs MEMRA_KV_ALLOCATOR, deleted at the tip; not run"; exit 1 ;;
    *) log "boot $name: unknown arm $arm"; exit 1 ;;
  esac
  [ -x "$B" ] || { log "boot $name: no binary $B; not run"; exit 1; }
  deadline=$((SECONDS + IDLE_WAIT_S)); waited=0
  until idle; do
    [ $SECONDS -ge $deadline ] && { log "boot $name: card $SMI_CARD not idle after $IDLE_WAIT_S s; not run"; exit 3; }
    [ $waited = 0 ] && log "boot $name: waiting for an idle card $SMI_CARD"
    waited=1; sleep 10
  done
  log "boot $name start arm=$arm bin=$(sha256sum "$B" | cut -c1-16) env=[${aenv[*]}] args=[$args]"
  printf 'arm=%s\nbin_sha256=%s\narm_env=%s\nclient_args=%s\n' "$arm" "$(sha256sum "$B" | cut -d' ' -f1)" "${aenv[*]}" "$args" > "$R/boots/$name.arm.txt"
  env "${aenv[@]}" CLIENT_ARGS="$args" bash research/spill-b-20260919/run-day26-cell.sh "$name" AB "$B" > "$R/boots/$name.launch.log" 2>&1
  rc=$?
  log "boot $name rc=$rc $(tail -1 "$R/boots/$name/client.log" 2>/dev/null | cut -c1-120)"
  [ "$rc" = 0 ] || exit "$rc"
  sleep 5
done
log "boots done: $*"
