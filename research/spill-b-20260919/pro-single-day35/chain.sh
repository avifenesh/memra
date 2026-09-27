#!/usr/bin/env bash
# WP-B day 35 target-card chain (DAY35.md 1.3), run ON BOX4 after lane A's LANE-A-BOX4-DONE: one RTX PRO 6000 Blackwell
# WS, Qwen3.8-27B NVFP4-Q5K MTP, MEMRA_CTX unset. Refuses to start unless the model's sha256 is day 32's and the card is
# idle. Builds red (main 25bbb91f5, the day-33 tree as merged) and green (the day-35 fix 809c16444) once each in detached
# worktrees, then runs red-R64, red-off, green-R64-r1, green-off, green-R64-r2 (day33-client.py), each under
# /tmp/memra-gpu.lock after a bounded idle wait. Never a signal to anything. Receipts under
# /root/spill-receipts/b-day35, mirrored to pro-single-day35/box/.
set -uo pipefail
R=/root/spill-receipts/b-day35; mkdir -p "$R/bins" "$R/boots"
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/chain.log"; }
cd /root/wt-b || { log "no /root/wt-b"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD >> "$R/fetch.log" 2>&1 \
  || { log "fetch/ff failed"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
log "chain start HEAD=$(git rev-parse HEAD)"
have=$(sha256sum "$MODEL" | cut -d' ' -f1)
[ "$have" = "$WANT_MODEL" ] || { log "model sha256 $have is not day 32's $WANT_MODEL; not run"; exit 1; }
idle() {
  flock -n /tmp/memra-gpu.lock true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
}
wait_idle() {
  local deadline=$((SECONDS + 7200))
  until idle; do [ $SECONDS -ge $deadline ] && { log "$1: card not idle after 7200 s; not run"; exit 3; }; sleep 30; done
}
build() { # <role> <sha>
  local W=/root/wt-b35-$1
  git worktree add --detach "$W" "$2" >> "$R/fetch.log" 2>&1 || { log "worktree $1 failed"; exit 1; }
  mkdir -p "$R/bins/$1"
  echo "cd $W && CARGO_TARGET_DIR=/root/wt-b/target nice -n 10 cargo build --release -p memra-server" > "$R/bins/$1/build-line.txt"
  git -C "$W" rev-parse HEAD > "$R/bins/$1/build-source.txt"
  ( cd "$W" && CARGO_TARGET_DIR=/root/wt-b/target nice -n 10 cargo build --release -p memra-server ) > "$R/bins/$1/build.log" 2>&1
  local rc=$?; echo "exit=$rc" >> "$R/bins/$1/build.log"
  [ $rc = 0 ] || { log "build $1 failed rc=$rc"; exit 1; }
  cp /root/wt-b/target/release/memra-server "$R/bins/$1/memra-server"   # named memra-server for the cell's stop step
  sha256sum "$R/bins/$1/memra-server" > "$R/bins/$1/binary.sha256"
  git worktree remove --force "$W" >> "$R/fetch.log" 2>&1
  log "built $1 from $2: $(cut -c1-16 "$R/bins/$1/binary.sha256")"
}
build red 25bbb91f5443e3e7aa2079c852d50a088fa6df52
build green 809c16444ad07aa9edd5d293e7d96d6a91a09fdc
export MODEL MODEL_KEY=q38 NO_SCOPE=1 WT=/root/wt-b CLIENT=day33-client.py PARSER=day31-parse.py N=5 \
  RIGDIR="$R/boots" MEMRA_TIMEOUT_MS_MAX=3600000 LOCK=/tmp/memra-gpu.lock
for spec in red-R64:red:on red-off:red:off green-R64-r1:green:on green-off:green:off green-R64-r2:green:on; do
  IFS=: read -r name role arm <<< "$spec"
  wait_idle "boot $name"
  log "boot $name start role=$role"
  if [ "$arm" = off ]; then
    env -u MEMRA_ADMIT_BY_MEMORY -u MEMRA_ADMIT_OPEN_OUTPUT_TOKENS CLIENT_ARGS="--chars 5000 --skip-long --burst 0" \
      bash research/spill-b-20260919/run-day26-cell.sh "$name" AB "$R/bins/$role/memra-server" > "$R/boots/$name.launch.log" 2>&1
  else
    env MEMRA_ADMIT_BY_MEMORY=1 MEMRA_ADMIT_OPEN_OUTPUT_TOKENS=32768 CLIENT_ARGS="--chars 5000 --skip-long --burst 64" \
      bash research/spill-b-20260919/run-day26-cell.sh "$name" AB "$R/bins/$role/memra-server" > "$R/boots/$name.launch.log" 2>&1
  fi
  log "boot $name rc=$? $(grep -h '^DAY31 V-BOOT' "$R/boots/$name/REPORT.txt" 2>/dev/null)"
  sleep 5
done
log "chain done"
