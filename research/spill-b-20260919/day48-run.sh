#!/usr/bin/env bash
# DAY48 boots (1.2, addenda A and D): one boot per spec. Addendum D: the runner holds the rig lock first (blocking,
# bounded by the boot's 7200 s deadline), checks the rig idle under the hold (no compute app, >= 24 GB host memory;
# rig-hold.sh), then boots through run-day26-cell.sh with LOCK=none and the hold's fd closed, and releases the hold after
# the cell. Never a signal to anything this lane did not start. usage: day48-run.sh <receipt root> <name>:<arm> ...
#   arm = enforce (MEMRA_ADMIT_PREDICT_ENFORCE=1) | enforce-vg (the same plus MEMRA_ADMIT_PREDICT_VG_DEBT=1); the default
#   (spec) route; run-day26-cell.sh arms MEMRA_ADMIT_PREDICT_SHADOW=1 on every boot.
#   MEMRA_ADMIT_BY_MEMORY and MEMRA_ADMIT_PREDICT_BUDGET_MB unset on every arm (the boot-derived budget).
#   env CLIENT_EXTRA passes the burst shape to day48-client.py.
# env: WT, RIG_LOCK (/tmp/memra-5090.lock), BIN (required), MODEL, MODEL_KEY, BOOT_CTX (65536; empty = the
#      checkpoint's), NO_SCOPE, YIELD_S.
set -uo pipefail
R=${1:?receipt root}; shift
WT=${WT:-$HOME/projects/wt-spill-b}
RIG_LOCK=${RIG_LOCK:-/tmp/memra-5090.lock}
BIN=${BIN:?the day-48 binary}
cd "$WT" || exit 1
mkdir -p "$R/boots"
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/run.log"; }
# shellcheck source=rig-hold.sh
. research/spill-b-20260919/rig-hold.sh
BOOT_CTX=${BOOT_CTX-65536}
if [ -n "$BOOT_CTX" ]; then export MEMRA_CTX=$BOOT_CTX; else unset MEMRA_CTX; fi
export LOCK=$RIG_LOCK RIGDIR="$R/boots" N=5 CLIENT=day48-client.py PARSER=day48-parse.py MEMRA_TIMEOUT_MS_MAX=3600000
for spec in "$@"; do
  IFS=: read -r name arm <<< "$spec"
  uenv=(-u MEMRA_ADMIT_W_RELEASE -u MEMRA_ADMIT_PREDICT_VG_DEBT -u MEMRA_SERVE_SPEC -u MEMRA_ADMIT_PREDICT_ENFORCE -u MEMRA_ADMIT_PREDICT_BUDGET_MB -u MEMRA_TTFT_TRACE
        -u MEMRA_ADMIT_BY_MEMORY -u MEMRA_ADMIT_OPEN_OUTPUT_TOKENS)
  aenv=(MEMRA_ADMIT_PREDICT_SHADOW=1)
  B=$BIN
  case $arm in
    enforce) aenv+=(MEMRA_ADMIT_PREDICT_ENFORCE=1) ;;
    enforce-vg) aenv+=(MEMRA_ADMIT_PREDICT_ENFORCE=1 MEMRA_ADMIT_PREDICT_VG_DEBT=1) ;;
    *) log "boot $name: unknown arm $arm"; exit 1 ;;
  esac
  args="${CLIENT_EXTRA:-}"
  [ -x "$B" ] || { log "boot $name: no binary $B; not run"; exit 1; }
  log "boot $name: waiting for $RIG_LOCK, then an idle rig under the hold (at most 7200 s)"
  rig_hold "$name" $((SECONDS + 7200))
  r=$?
  [ $r = 0 ] || { log "boot $name: rig not held idle within 7200 s; not run"; exit 3; }
  held_at=$(date -u +%FT%TZ)
  log "boot $name start arm=$arm bin=$(sha256sum "$B" | cut -c1-16) env=[${aenv[*]}] args=[$args]"
  printf 'arm=%s\nbin_sha256=%s\nenv=%s\nclient_args=%s\nlock_hold=%s held by day48-run.sh from %s\n' "$arm" \
    "$(sha256sum "$B" | cut -d' ' -f1)" "${aenv[*]}" "$args" "$RIG_LOCK" "$held_at" > "$R/boots/$name.arm.txt"
  env "${uenv[@]}" "${aenv[@]}" LOCK=none CLIENT_ARGS="$args" bash research/spill-b-20260919/run-day26-cell.sh "$name" AB \
    "$B" > "$R/boots/$name.launch.log" 2>&1 8>&-
  rc=$?
  rig_release "$name"
  log "boot $name rc=$rc $(tail -1 "$R/boots/$name/REPORT.txt" 2>/dev/null | cut -c1-160)"
  sleep "${YIELD_S:-5}" # the lane yields the card between cells when YIELD_S is set (lead, 2026-09-26)
done
log "boots done: $*"
