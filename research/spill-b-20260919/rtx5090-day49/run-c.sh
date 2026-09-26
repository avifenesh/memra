#!/usr/bin/env bash
# DAY49 addendum C local cell i-vmm (the 5090, the 9B): the gate's arm i under MEMRA_KV_ALLOCATOR=vmm on the fix
# (green, 95d35c383) and on 02dbdfa40 (red). Each side runs the gate from its own detached worktree, built there first
# at nice 19 under 600% (so the gate's own build finds it current), then under /tmp/memra-5090.lock after an idle
# wait, with the lock free for 240 s after each run. Receipts rtx5090-day49c/. Never a signal to anything.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919/rtx5090-day49c
mkdir -p "$D"
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/run.log"; }
W=(systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=20G nice -n 19)
SIDES=(green:95d35c3838853056d02ec356a06add1ed180d320 red:02dbdfa408a5d0215248c0ec2849cd599ca64083)
echo "$(date -u +%FT%TZ) WP-B DAY49C worktree builds (nice 19, CPUQuota=600%)" >> "$WT/research/spill-b-20260919/cpu-concurrency.log"
: > "$D/binaries.sha256"
for s in "${SIDES[@]}"; do
  name=${s%%:*}; sha=${s#*:}; T=$WT/target/wt-day49c-$name
  [ -d "$T" ] || git worktree add --detach "$T" "$sha" > "$D/$name-worktree.log" 2>&1 || { log "$name: worktree failed"; exit 2; }
  ( cd "$T" && "${W[@]}" cargo build --release -p memra-server ) > "$D/$name-build.log" 2>&1 || { log "$name: build failed"; exit 2; }
  echo "$sha $(sha256sum "$T/target/release/memra-server" | cut -d' ' -f1) $name" >> "$D/binaries.sha256"
  log "$name built at $sha"
done
idle() { flock -n /tmp/memra-5090.lock true && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
for s in "${SIDES[@]}"; do
  name=${s%%:*}; T=$WT/target/wt-day49c-$name
  deadline=$((SECONDS + 14400))
  until idle; do [ $SECONDS -ge $deadline ] && { log "$name: not idle; not run"; exit 3; }; sleep 30; done
  log "$name start HEAD=$(git -C "$T" rev-parse HEAD)"
  MEMRA_KV_ALLOCATOR=vmm HFG_ARMS=i HFG_OUT="$D/$name" flock -w 600 /tmp/memra-5090.lock "$T/tools/health-fault-gate.sh" \
    > "$D/$name.log" 2>&1
  log "$name rc=$? $(tail -1 "$D/$name.log")"
  sleep 240
done
python3 "$WT/research/spill-b-20260919/day49c-read.py" rtx5090 "$D/green" "$D/red" > "$D/read.log" 2>&1
for s in "${SIDES[@]}"; do git worktree remove --force "$WT/target/wt-day49c-${s%%:*}" >> "$D/run.log" 2>&1; done
log "done"
