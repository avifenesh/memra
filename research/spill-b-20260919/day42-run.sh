#!/usr/bin/env bash
# DAY42 boots (1.4): one boot per spec through run-day26-cell.sh (it takes the rig lock), after a bounded idle wait (at
# most 7200 s: lock free, no compute app, >= 24 GB host memory). Never a signal to anything this lane did not start.
# usage: day42-run.sh <receipt root> <name>:<arm> ...
#   arm = ontick | offtick | ontick-nocontracts | fault-d2h-delay | fault-d2h-source-flip | fault-sources-helper-gone
# env: WT, RIG_LOCK (/tmp/memra-5090.lock), BIN, MODEL, MODEL_KEY, BOOT_CTX (65536; empty = the checkpoint's), NO_SCOPE,
#      HOST_MB (8192), BURST (32).
set -uo pipefail
R=${1:?receipt root}; shift
WT=${WT:-$HOME/projects/wt-spill-b}
RIG_LOCK=${RIG_LOCK:-/tmp/memra-5090.lock}
BIN=${BIN:-$WT/target/day42/tip/memra-server}
HOST_MB=${HOST_MB:-8192}
BURST=${BURST:-32}
# DAY42 addendum C: the warm set, the warm requests' output, and the open-output charge per card.
WARM=${WARM:-8}
WARM_TOKENS=${WARM_TOKENS:-8192}
WARM_MAX_TOKENS=${WARM_MAX_TOKENS:-1}
OPEN_OUTPUT=${OPEN_OUTPUT:-8192}
# DAY42 addendum D: the burst's request-supplied context (0 = open output).
BURST_MAX_CTX=${BURST_MAX_CTX:-0}
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
export LOCK=$RIG_LOCK RIGDIR="$R/boots" N=1 CLIENT=day42-client.py PARSER=day42-parse.py MEMRA_TIMEOUT_MS_MAX=3600000
for spec in "$@"; do
  IFS=: read -r name arm <<< "$spec"
  uenv=(-u MEMRA_ADMIT_RECLAIM_OFFTICK -u MEMRA_KV_HOST_CONTRACTS -u MEMRA_KV_HOST_FAULT -u MEMRA_PREFIX_CACHE_MB)
  aenv=(MEMRA_ADMIT_BY_MEMORY=1 "MEMRA_ADMIT_OPEN_OUTPUT_TOKENS=$OPEN_OUTPUT" "MEMRA_KV_HOST_MB=$HOST_MB")
  case $arm in
    ontick) aenv+=(MEMRA_KV_HOST_CONTRACTS=1) ;;
    offtick) aenv+=(MEMRA_KV_HOST_CONTRACTS=1 MEMRA_ADMIT_RECLAIM_OFFTICK=1) ;;
    ontick-nocontracts) ;;
    fault-d2h-delay) aenv+=(MEMRA_KV_HOST_CONTRACTS=1 MEMRA_ADMIT_RECLAIM_OFFTICK=1 MEMRA_KV_HOST_FAULT=d2h-delay) ;;
    fault-d2h-source-flip) aenv+=(MEMRA_KV_HOST_CONTRACTS=1 MEMRA_ADMIT_RECLAIM_OFFTICK=1 MEMRA_KV_HOST_FAULT=d2h-source-flip) ;;
    fault-sources-helper-gone) aenv+=(MEMRA_KV_HOST_CONTRACTS=1 MEMRA_ADMIT_RECLAIM_OFFTICK=1 MEMRA_KV_HOST_FAULT=sources-helper-gone) ;;
    *) log "boot $name: unknown arm $arm"; exit 1 ;;
  esac
  [ -x "$BIN" ] || { log "boot $name: no binary $BIN; not run"; exit 1; }
  deadline=$((SECONDS + 7200)); waited=0
  until idle; do
    [ $SECONDS -ge $deadline ] && { log "boot $name: rig not idle after 7200 s; not run"; exit 3; }
    [ $waited = 0 ] && log "boot $name: waiting for an idle rig"
    waited=1; sleep 30
  done
  args="--burst $BURST --warm $WARM --warm-tokens $WARM_TOKENS --warm-max-tokens $WARM_MAX_TOKENS --burst-max-ctx $BURST_MAX_CTX"
  log "boot $name start arm=$arm bin=$(sha256sum "$BIN" | cut -c1-16) env=[${aenv[*]}] args=[$args]"
  printf 'arm=%s\nbin_sha256=%s\nenv=%s\nclient_args=%s\n' "$arm" "$(sha256sum "$BIN" | cut -d' ' -f1)" "${aenv[*]}" "$args" \
    > "$R/boots/$name.arm.txt"
  env "${uenv[@]}" "${aenv[@]}" CLIENT_ARGS="$args" bash research/spill-b-20260919/run-day26-cell.sh "$name" AB "$BIN" \
    > "$R/boots/$name.launch.log" 2>&1
  log "boot $name rc=$? $(tail -1 "$R/boots/$name/client.log" 2>/dev/null | cut -c1-160)"
  sleep 5
done
log "boots done: $*"
