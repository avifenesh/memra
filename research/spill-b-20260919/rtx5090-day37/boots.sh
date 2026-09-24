#!/usr/bin/env bash
# DAY37 serving boots on the local RTX 5090 (1.6, addendum B): one boot per spec, each under its own
# /tmp/memra-5090.lock hold (run-day26-cell.sh takes it), after a bounded idle wait (at most 7200 s: lock free, no
# compute app, >= 24 GB host memory). Never a signal to anything this lane did not start.
# usage: boots.sh <receipt root> <name>:<arm>:<kind> ...
#   arm  = pooled | vmm | vmm-mapperfault | vmm-ensurefault | vmm-buildfault1 | vmm-buildfault64 | main
#   kind = mixspec | mixplain | stream | g2 | l64 | boff
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
LANE_BIN=$WT/target/day37/lane/memra-server
MAIN_BIN=$WT/target/day37/main/memra-server
build64=$(seq -s, 1 64 | sed 's/\([0-9]*\)/build:\1/g')
export MEMRA_CTX=65536 LOCK=/tmp/memra-5090.lock RIGDIR="$R/boots" N=5 MEMRA_TIMEOUT_MS_MAX=3600000
for spec in "$@"; do
  IFS=: read -r name arm kind <<< "$spec"
  BIN=$LANE_BIN
  aenv=()
  case $arm in
    pooled) ;;
    vmm) aenv=(MEMRA_KV_ALLOCATOR=vmm) ;;
    vmm-mapperfault) aenv=(MEMRA_KV_ALLOCATOR=vmm MEMRA_KV_VMM_FAULT=mapper:all) ;;
    vmm-ensurefault) aenv=(MEMRA_KV_ALLOCATOR=vmm MEMRA_KV_VMM_FAULT=mapper:all,ensure:1,ensure:2) ;;
    vmm-buildfault1) aenv=(MEMRA_KV_ALLOCATOR=vmm MEMRA_KV_VMM_FAULT=build:1) ;;
    vmm-buildfault64) aenv=(MEMRA_KV_ALLOCATOR=vmm "MEMRA_KV_VMM_FAULT=$build64") ;;
    main) BIN=$MAIN_BIN ;;
    *) log "boot $name: unknown arm $arm"; exit 1 ;;
  esac
  kenv=(); unset CLIENT CLIENT_ARGS PARSER
  case $kind in
    # The day-26 mix as day 26 ran it: the default request deadline (a 90 s cut is a wall-time event, recorded and
    # excluded from the digest comparison with its count, as day 27 did).
    mixspec) kenv=(-u MEMRA_SERVE_SPEC -u MEMRA_TIMEOUT_MS_MAX) ;;
    mixplain) kenv=(-u MEMRA_TIMEOUT_MS_MAX MEMRA_SERVE_SPEC=0) ;;
    stream) kenv=(-u MEMRA_SERVE_SPEC CLIENT=day37-stream-client.py PARSER=day37-stream-parse.py "CLIENT_ARGS=--arm $arm --requests 32 --concurrency 8") ;;
    g2) kenv=(CLIENT=day33-client.py PARSER=day31-parse.py MEMRA_ADMIT_BY_MEMORY=1 MEMRA_ADMIT_OPEN_OUTPUT_TOKENS=8192 "CLIENT_ARGS=--chars 5000 --skip-long --burst 64") ;;
    l64) kenv=(CLIENT=day33-client.py PARSER=day31-parse.py MEMRA_ADMIT_BY_MEMORY=1 MEMRA_ADMIT_OPEN_OUTPUT_TOKENS=32768 "CLIENT_ARGS=--chars 5000 --skip-long --burst 64") ;;
    boff) kenv=(-u MEMRA_ADMIT_BY_MEMORY -u MEMRA_ADMIT_OPEN_OUTPUT_TOKENS CLIENT=day33-client.py PARSER=day31-parse.py "CLIENT_ARGS=--chars 5000 --skip-long --burst 64") ;;
    *) log "boot $name: unknown kind $kind"; exit 1 ;;
  esac
  [ -x "$BIN" ] || { log "boot $name: no binary $BIN; not run"; exit 1; }
  deadline=$((SECONDS + 7200)); waited=0
  until idle; do
    [ $SECONDS -ge $deadline ] && { log "boot $name: rig not idle after 7200 s; not run"; exit 3; }
    [ $waited = 0 ] && log "boot $name: waiting for an idle rig"
    waited=1; sleep 30
  done
  log "boot $name start arm=$arm kind=$kind bin=$(sha256sum "$BIN" | cut -c1-16) env=[${aenv[*]} ${kenv[*]}]"
  mkdir -p "$R/boots"
  printf 'arm=%s\nkind=%s\nbin_sha256=%s\narm_env=%s\nkind_env=%s\n' "$arm" "$kind" "$(sha256sum "$BIN" | cut -d' ' -f1)" \
    "${aenv[*]}" "${kenv[*]}" > "$R/boots/$name.arm.txt"
  env "${kenv[@]}" "${aenv[@]}" bash research/spill-b-20260919/run-day26-cell.sh "$name" AB "$BIN" > "$R/boots/$name.launch.log" 2>&1
  rc=$?
  on=$(grep -c "\[kv-vmm\] door=ON" "$R/boots/$name/server.log" 2>/dev/null); off=$(grep -c "\[kv-vmm\] door=OFF" "$R/boots/$name/server.log" 2>/dev/null)
  log "boot $name rc=$rc door_on=$on door_off=$off $(grep -hE '^DAY37 STREAM |^DAY31 V-BOOT' "$R/boots/$name/REPORT.txt" 2>/dev/null | head -1 | cut -c1-200)"
  sleep 5
done
log "boots done: $*"
