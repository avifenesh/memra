#!/usr/bin/env bash
# DAY49 addendum D local half (the 5090, the 9B): the target chain's steps on this card. Worktrees green (D) and red (D
# with day49d-noreap.patch) built first at nice 19 under 600%. Then, each unit after an idle check with the lock taken
# here (fd 9; a lock not taken within 60 s goes back to the idle wait): arm j twice from green, arm j under the VMM door
# from green and from red; then the serving shape (day49d-run.sh takes the lock per boot); then the readers. The lock
# stays free for 240 s after every unit. Every exit is captured on its own line. Never a signal to anything.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919/rtx5090-day49d
SD=${SD:-8926ccfb3e8a27cf8f91aa747e2b088251215c7a}
LOCK=/tmp/memra-5090.lock
G=$WT/target/wt-day49d-green; RD=$WT/target/wt-day49d-red
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/run.log"; }
W=(systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=20G nice -n 19)
echo "$(date -u +%FT%TZ) WP-B DAY49D worktree builds (nice 19, CPUQuota=600%)" >> "$WT/research/spill-b-20260919/cpu-concurrency.log"
: > "$D/binaries.sha256"
for side in green red; do
  T=$G; [ "$side" = red ] && T=$RD
  [ -d "$T" ] || git worktree add --detach "$T" "$SD" > "$D/$side-worktree.log" 2>&1 || { log "$side: worktree failed"; exit 2; }
  if [ "$side" = red ] && ! git -C "$T" diff --quiet; then :; elif [ "$side" = red ]; then
    git -C "$T" apply "$WT/research/spill-b-20260919/day49d-noreap.patch" || { log "red: patch does not apply"; exit 2; }
  fi
  ( cd "$T" && "${W[@]}" cargo build --release -p memra-server ) > "$D/$side-build.log" 2>&1
  rc=$?
  [ $rc = 0 ] || { log "$side: build failed rc=$rc"; exit 2; }
  echo "$SD $(sha256sum "$T/target/release/memra-server" | cut -d' ' -f1) $side" >> "$D/binaries.sha256"
  log "$side built"
done
idle() { flock -n "$LOCK" true && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
unit() { # <label> <tree> <env...>
  local label=$1 tree=$2; shift 2
  local deadline=$((SECONDS + 28800)) rc
  while :; do
    until idle; do
      [ $SECONDS -ge $deadline ] && { log "$label: card not idle after 28800 s; not run"; return 3; }
      sleep 30
    done
    exec 9>"$LOCK"
    flock -w 60 9 && break
    exec 9>&-
    log "$label: lock taken by another runner inside the window; back to the idle wait"
  done
  log "$label start HEAD=$(git -C "$tree" rev-parse HEAD) dirty=$(git -C "$tree" diff --stat | tail -1)"
  env "$@" HFG_ARMS=j HFG_OUT="$D/$label" "$tree/tools/health-fault-gate.sh" > "$D/$label.log" 2>&1
  rc=$?
  exec 9>&-
  log "$label rc=$rc $(tail -1 "$D/$label.log")"
  sleep 240
}
unit j1 "$G"
unit j2 "$G"
unit vmm-green "$G" MEMRA_KV_ALLOCATOR=vmm
unit vmm-red "$RD" MEMRA_KV_ALLOCATOR=vmm
BIN=$G/target/release/memra-server FAULT=batch:1 YIELD_S=240 CLIENT_EXTRA="--warm-n 0 --burst 8 --length 6144 --max-tokens 64" \
  bash "$WT/research/spill-b-20260919/day49d-run.sh" "$D" O1-off:off O1-on:on O2-on:on O2-off:off \
  P1-off:off-plain P1-on:on-plain P2-on:on-plain P2-off:off-plain
rc=$?
log "boots rc=$rc"
python3 "$WT/research/spill-b-20260919/day49d-read.py" serve rtx5090 "$D" > "$D/read-serve.log" 2>&1
python3 "$WT/research/spill-b-20260919/day49d-read.py" vmm rtx5090 "$D/vmm-green" "$D/vmm-red" > "$D/read-vmm.log" 2>&1
git worktree remove --force "$G" >> "$D/run.log" 2>&1; git worktree remove --force "$RD" >> "$D/run.log" 2>&1
log "done"
