#!/usr/bin/env bash
# WP-B day 39 (DAY39.md 1.3): run named boots of the day-35 cell (day33-client.py, day31-parse.py) with day 39's red
# and green binaries, one at a time, each under its own rig-lock hold (run-day26-cell.sh takes it). Before each boot a
# bounded idle wait (at most 7200 s: lock free, no compute app, at least 24 GB of host memory available). Never a
# signal to anything.
# usage: day39-run.sh <receipt root> <boot spec> ...     boot spec = <name>:<red|green>:<shape>
#        shape = G2 (ON, 8192) | L64 (ON, 32768) | R64 (ON, 32768, the target card's name) | off
# env: WT, BINS (<bins>/<role>/memra-server; the file is named memra-server for the cell's stop step), RIG_LOCK
#      (/tmp/memra-5090.lock), BOOT_CTX (65536; empty = the checkpoint's), MODEL, MODEL_KEY, NO_SCOPE (the cell's).
set -uo pipefail
R=${1:?receipt root}; shift
WT=${WT:-$HOME/projects/wt-spill-b}
BINS=${BINS:-$WT/target/day39}
RIG_LOCK=${RIG_LOCK:-/tmp/memra-5090.lock}
BOOT_CTX=${BOOT_CTX-65536}
cd "$WT" || exit 1
mkdir -p "$R/boots"
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/run.log"; }
idle() {
  flock -n "$RIG_LOCK" true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
  [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 24 ] || return 1
}
if [ -n "$BOOT_CTX" ]; then export MEMRA_CTX=$BOOT_CTX; else unset MEMRA_CTX; fi
export CLIENT=day33-client.py PARSER=day31-parse.py N=5 RIGDIR="$R/boots" MEMRA_TIMEOUT_MS_MAX=3600000 LOCK=$RIG_LOCK
for spec in "$@"; do
  IFS=: read -r name role shape <<< "$spec"
  BIN=$BINS/$role/memra-server
  [ -x "$BIN" ] || { log "boot $name: no binary $BIN; not run"; exit 1; }
  deadline=$((SECONDS + 7200)); waited=0
  until idle; do
    [ $SECONDS -ge $deadline ] && { log "boot $name: rig not idle after 7200 s; not run"; exit 3; }
    [ $waited = 0 ] && log "boot $name: waiting for an idle rig"
    waited=1; sleep 30
  done
  case $shape in
    G2) v=8192; burst=64 ;; L64|R64) v=32768; burst=64 ;; off) v=; burst=0 ;;
    *) log "boot $name: unknown shape $shape"; exit 1 ;;
  esac
  log "boot $name start role=$role shape=$shape bin=$(sha256sum "$BIN" | cut -c1-16)"
  if [ -z "$v" ]; then
    env -u MEMRA_ADMIT_BY_MEMORY -u MEMRA_ADMIT_OPEN_OUTPUT_TOKENS CLIENT_ARGS="--chars 5000 --skip-long --burst 0" \
      bash research/spill-b-20260919/run-day26-cell.sh "$name" AB "$BIN" > "$R/boots/$name.launch.log" 2>&1
  else
    env MEMRA_ADMIT_BY_MEMORY=1 MEMRA_ADMIT_OPEN_OUTPUT_TOKENS=$v CLIENT_ARGS="--chars 5000 --skip-long --burst $burst" \
      bash research/spill-b-20260919/run-day26-cell.sh "$name" AB "$BIN" > "$R/boots/$name.launch.log" 2>&1
  fi
  log "boot $name rc=$? $(grep -h '^DAY31 V-BOOT' "$R/boots/$name/REPORT.txt" 2>/dev/null)"
  sleep "${YIELD_S:-5}" # the lane yields the card between cells when YIELD_S is set (lead, 2026-09-26)
done
log "run done: $*"
