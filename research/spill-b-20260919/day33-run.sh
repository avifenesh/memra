#!/usr/bin/env bash
# WP-B day 33 (DAY33.md 1.3): run named boots of the memra#680 cell, one at a time, each under its own
# /tmp/memra-5090.lock hold (run-day26-cell.sh takes it). Before each boot: a bounded idle wait (at most 7200 s: lock
# free, no compute app, at least 24 GB of host memory available). Never a signal to anything.
# usage: day33-run.sh <receipt root> <boot spec> ...     boot spec = <name>:<red|green>:<shape>
#        shape = G1 | G2 | R32 | off (DAY33.md 1.3); binaries target/day33/memra-server-<red|green>
set -uo pipefail
R=${1:?receipt root}; shift
WT=$HOME/projects/wt-spill-b
cd "$WT" || exit 1
mkdir -p "$R/boots"
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/run.log"; }
idle() {
  flock -n /tmp/memra-5090.lock true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
  [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 24 ] || return 1
}
export CLIENT=day33-client.py PARSER=day31-parse.py N=5 RIGDIR="$R/boots" MEMRA_TIMEOUT_MS_MAX=3600000 \
  MEMRA_CTX=65536 LOCK=/tmp/memra-5090.lock
for spec in "$@"; do
  IFS=: read -r name role shape <<< "$spec"
  BIN=$WT/target/day33/memra-server-$role
  [ -x "$BIN" ] || { log "boot $name: no binary $BIN; not run"; exit 1; }
  deadline=$((SECONDS + 7200)); waited=0
  until idle; do
    [ $SECONDS -ge $deadline ] && { log "boot $name: rig not idle after 7200 s; not run"; exit 3; }
    [ $waited = 0 ] && log "boot $name: waiting for an idle rig"
    waited=1; sleep 30
  done
  case $shape in
    G1) v=2048; burst=64 ;; G2) v=8192; burst=64 ;; R32) v=32768; burst=32 ;; off) v=; burst=0 ;;
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
  sleep 5
done
log "run done: $*"
