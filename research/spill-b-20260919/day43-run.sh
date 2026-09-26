#!/usr/bin/env bash
# DAY43 boots (1.4): one boot per spec through run-day26-cell.sh (it takes the rig lock), after a bounded idle
# wait (at most 7200 s: lock free, no compute app, >= 24 GB host memory). Never a signal to anything this lane did not
# start. usage: day43-run.sh <receipt root> <name>:<arm>:<route>:<shape> ...
#   arm = unset | clamp | offprev      route = spec | plain      shape = RX
# env: WT, RIG_LOCK (/tmp/memra-5090.lock), BIN, PREV_BIN, MODEL, MODEL_KEY, BOOT_CTX (65536; empty = the checkpoint's),
#      LENGTHS (6144,30720), NO_SCOPE.
set -uo pipefail
R=${1:?receipt root}; shift
WT=${WT:-$HOME/projects/wt-spill-b}
RIG_LOCK=${RIG_LOCK:-/tmp/memra-5090.lock}
BIN=${BIN:-$WT/target/day43/tip/memra-server}
PREV_BIN=${PREV_BIN:-$WT/target/day43/offprev/memra-server}
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
export LOCK=$RIG_LOCK RIGDIR="$R/boots" N=5 CLIENT=day41-client.py PARSER=day41-parse.py MEMRA_TIMEOUT_MS_MAX=3600000
for spec in "$@"; do
  IFS=: read -r name arm route shape <<< "$spec"
  uenv=(-u MEMRA_SPEC_BUDGET_CLAMP -u MEMRA_RESUME_GRID_REWIND -u MEMRA_SERVE_SPEC -u MEMRA_PREFIX_CACHE_MB)
  aenv=()
  B=$BIN
  case $arm in
    unset) ;;
    clamp) aenv+=(MEMRA_SPEC_BUDGET_CLAMP=1) ;;
    offprev) B=$PREV_BIN ;;
    *) log "boot $name: unknown arm $arm"; exit 1 ;;
  esac
  case $route in
    plain) aenv+=(MEMRA_SERVE_SPEC=0) ;;
    spec) ;;
    *) log "boot $name: unknown route $route"; exit 1 ;;
  esac
  case $shape in
    RX) aenv+=(MEMRA_PREFIX_CACHE_MB=0); args="--shapes RX --lengths $LENGTHS" ;;
    RX6) aenv+=(MEMRA_PREFIX_CACHE_MB=0); args="--shapes RX --lengths 6144" ;;
    FX) args="--shapes FX" ;;
    *) log "boot $name: unknown shape $shape"; exit 1 ;;
  esac
  [ -x "$B" ] || { log "boot $name: no binary $B; not run"; exit 1; }
  deadline=$((SECONDS + 7200)); waited=0
  until idle; do
    [ $SECONDS -ge $deadline ] && { log "boot $name: rig not idle after 7200 s; not run"; exit 3; }
    [ $waited = 0 ] && log "boot $name: waiting for an idle rig"
    waited=1; sleep 30
  done
  log "boot $name start arm=$arm route=$route shape=$shape bin=$(sha256sum "$B" | cut -c1-16) env=[${aenv[*]}] args=[$args]"
  printf 'arm=%s\nroute=%s\nshape=%s\nbin_sha256=%s\nenv=%s\nclient_args=%s\n' "$arm" "$route" "$shape" \
    "$(sha256sum "$B" | cut -d' ' -f1)" "${aenv[*]}" "$args" > "$R/boots/$name.arm.txt"
  env "${uenv[@]}" "${aenv[@]}" CLIENT_ARGS="$args" bash research/spill-b-20260919/run-day26-cell.sh "$name" AB "$B" \
    > "$R/boots/$name.launch.log" 2>&1
  log "boot $name rc=$? $(tail -1 "$R/boots/$name/REPORT.txt" 2>/dev/null | cut -c1-160)"
  sleep "${YIELD_S:-5}" # the lane yields the card between cells when YIELD_S is set (lead, 2026-09-26)
done
log "boots done: $*"
