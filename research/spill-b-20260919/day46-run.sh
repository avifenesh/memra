#!/usr/bin/env bash
# DAY46 boots (1.1, 1.2, addendum A): one boot per spec through run-day26-cell.sh (it takes the rig lock), after a bounded
# idle wait (at most 7200 s: lock free, no compute app, >= 24 GB host memory). Never a signal to anything this lane did
# not start. usage: day46-run.sh <receipt root> <name>:<arm> ...
#   arm = shadow (the predictive door logs only; run-day26-cell.sh arms MEMRA_ADMIT_PREDICT_SHADOW=1 on every boot) |
#   enforce (MEMRA_ADMIT_PREDICT_ENFORCE=1) | enforce-wrel (the same plus MEMRA_ADMIT_W_RELEASE=1).
#   MEMRA_ADMIT_BY_MEMORY and MEMRA_ADMIT_PREDICT_BUDGET_MB unset on every arm (the boot-derived budget).
#   env CLIENT_EXTRA passes the burst shape to day46-client.py.
# env: WT, RIG_LOCK (/tmp/memra-5090.lock), BIN (required), MODEL, MODEL_KEY, BOOT_CTX (65536; empty = the
#      checkpoint's), NO_SCOPE, YIELD_S.
set -uo pipefail
R=${1:?receipt root}; shift
WT=${WT:-$HOME/projects/wt-spill-b}
RIG_LOCK=${RIG_LOCK:-/tmp/memra-5090.lock}
BIN=${BIN:?the day-46 binary}
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
export LOCK=$RIG_LOCK RIGDIR="$R/boots" N=5 CLIENT=day46-client.py PARSER=day46-parse.py MEMRA_TIMEOUT_MS_MAX=3600000
for spec in "$@"; do
  IFS=: read -r name arm <<< "$spec"
  uenv=(-u MEMRA_ADMIT_W_RELEASE -u MEMRA_ADMIT_PREDICT_ENFORCE -u MEMRA_ADMIT_PREDICT_BUDGET_MB -u MEMRA_TTFT_TRACE
        -u MEMRA_ADMIT_BY_MEMORY -u MEMRA_ADMIT_OPEN_OUTPUT_TOKENS)
  aenv=(MEMRA_ADMIT_PREDICT_SHADOW=1)
  B=$BIN
  case $arm in
    shadow) ;;
    enforce) aenv+=(MEMRA_ADMIT_PREDICT_ENFORCE=1) ;;
    enforce-wrel) aenv+=(MEMRA_ADMIT_PREDICT_ENFORCE=1 MEMRA_ADMIT_W_RELEASE=1) ;;
    *) log "boot $name: unknown arm $arm"; exit 1 ;;
  esac
  args="${CLIENT_EXTRA:-}"
  [ -x "$B" ] || { log "boot $name: no binary $B; not run"; exit 1; }
  deadline=$((SECONDS + 7200)); waited=0
  until idle; do
    [ $SECONDS -ge $deadline ] && { log "boot $name: rig not idle after 7200 s; not run"; exit 3; }
    [ $waited = 0 ] && log "boot $name: waiting for an idle rig"
    waited=1; sleep 30
  done
  log "boot $name start arm=$arm bin=$(sha256sum "$B" | cut -c1-16) env=[${aenv[*]}] args=[$args]"
  printf 'arm=%s\nbin_sha256=%s\nenv=%s\nclient_args=%s\n' "$arm" \
    "$(sha256sum "$B" | cut -d' ' -f1)" "${aenv[*]}" "$args" > "$R/boots/$name.arm.txt"
  env "${uenv[@]}" "${aenv[@]}" CLIENT_ARGS="$args" bash research/spill-b-20260919/run-day26-cell.sh "$name" AB "$B" \
    > "$R/boots/$name.launch.log" 2>&1
  rc=$?
  log "boot $name rc=$rc $(tail -1 "$R/boots/$name/REPORT.txt" 2>/dev/null | cut -c1-160)"
  sleep "${YIELD_S:-5}" # the lane yields the card between cells when YIELD_S is set (lead, 2026-09-26)
done
log "boots done: $*"
