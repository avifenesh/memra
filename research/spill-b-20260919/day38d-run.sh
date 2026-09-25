#!/usr/bin/env bash
# DAY38 addendum D boots (1.11; day38-run.sh with shape Xp in place of X): one boot per spec through run-day26-cell.sh (it takes the rig lock), after a bounded idle
# wait (at most 7200 s: lock free, no compute app, >= 24 GB host memory). Never a signal to anything this lane did not
# start. usage: day38-run.sh <receipt root> <name>:<arm> ...
#   arm = off | on | on-fault-batch | on-fault-nobatch | on-fault-nobatch-red | vmm-off | vmm-on
# env: WT, RIG_LOCK (/tmp/memra-5090.lock), BIN, MODEL, MODEL_KEY, BOOT_CTX (65536; empty = the checkpoint's),
#      LENGTHS (6144,30720), NO_SCOPE.
set -uo pipefail
R=${1:?receipt root}; shift
WT=${WT:-$HOME/projects/wt-spill-b}
RIG_LOCK=${RIG_LOCK:-/tmp/memra-5090.lock}
BIN=${BIN:-$WT/target/day38d/green/memra-server}
RED_BIN=${RED_BIN:-$WT/target/day38d/red/memra-server}
LENGTHS=${LENGTHS:-6144,30720}
cd "$WT" || exit 1
mkdir -p "$R/boots"
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/run.log"; }
idle() {
  flock -n "$RIG_LOCK" true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
  [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 24 ] || return 1
}
BOOT_CTX=${BOOT_CTX-65536}
if [ -n "$BOOT_CTX" ]; then export MEMRA_CTX=$BOOT_CTX; else unset MEMRA_CTX; fi
export LOCK=$RIG_LOCK RIGDIR="$R/boots" N=5 CLIENT=day38-client.py PARSER=day38-parse.py MEMRA_TIMEOUT_MS_MAX=3600000 \
  MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=0
for spec in "$@"; do
  IFS=: read -r name arm <<< "$spec"
  aenv=(-u MEMRA_KV_PARK_COMPACT -u MEMRA_KV_ALLOCATOR -u MEMRA_STEP_OOM_FAULT -u MEMRA_SERVE_BATCH)
  use_red=0
  args="--lengths $LENGTHS --shapes Xp,A"
  case $arm in
    off) ;;
    on) aenv+=(MEMRA_KV_PARK_COMPACT=1) ;;
    on-fault-batch) aenv+=(MEMRA_KV_PARK_COMPACT=1 MEMRA_STEP_OOM_FAULT=1); args="--lengths 6144 --shapes Xp --turn4" ;;
    on-fault-nobatch) aenv+=(MEMRA_KV_PARK_COMPACT=1 MEMRA_STEP_OOM_FAULT=1 MEMRA_SERVE_BATCH=0); args="--lengths 6144 --shapes Xp --turn4" ;;
    on-fault-nobatch-red) aenv+=(MEMRA_KV_PARK_COMPACT=1 MEMRA_STEP_OOM_FAULT=1 MEMRA_SERVE_BATCH=0); args="--lengths 6144 --shapes Xp --turn4"; use_red=1 ;;
    vmm-off) aenv+=(MEMRA_KV_ALLOCATOR=vmm); args="--lengths 6144 --shapes Xp" ;;
    vmm-on) aenv+=(MEMRA_KV_ALLOCATOR=vmm MEMRA_KV_PARK_COMPACT=1); args="--lengths 6144 --shapes Xp" ;;
    *) log "boot $name: unknown arm $arm"; exit 1 ;;
  esac
  B=$BIN; [ "$use_red" = 1 ] && B=$RED_BIN
  [ -x "$B" ] || { log "boot $name: no binary $B; not run"; exit 1; }
  deadline=$((SECONDS + 7200)); waited=0
  until idle; do
    [ $SECONDS -ge $deadline ] && { log "boot $name: rig not idle after 7200 s; not run"; exit 3; }
    [ $waited = 0 ] && log "boot $name: waiting for an idle rig"
    waited=1; sleep 30
  done
  log "boot $name start arm=$arm bin=$(sha256sum "$B" | cut -c1-16) env=[${aenv[*]}] args=[$args]"
  printf 'arm=%s\nbin_sha256=%s\narm_env=%s\nclient_args=%s\n' "$arm" "$(sha256sum "$B" | cut -d' ' -f1)" "${aenv[*]}" "$args" > "$R/boots/$name.arm.txt"
  env "${aenv[@]}" CLIENT_ARGS="$args" bash research/spill-b-20260919/run-day26-cell.sh "$name" AB "$B" > "$R/boots/$name.launch.log" 2>&1
  log "boot $name rc=$? $(tail -1 "$R/boots/$name/client.log" 2>/dev/null | cut -c1-120)"
  sleep 5
done
log "boots done: $*"
