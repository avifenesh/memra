#!/usr/bin/env bash
# WP-B day 35 addendum (DAY35.md section 3), run ON BOX4 after day 36's box half: the review fix, same acceptance. Builds
# the integ55 tip 34a4b7f23 (the day-35 seed term capped at the budget the prefix cache has left) once in a detached
# worktree, then runs green-R64-a1, green-off-a and green-R64-a2 (day33-client.py, day 35's shapes) under
# /tmp/memra-gpu.lock after a bounded idle wait. Read with day33-compare.py against day 35's own red-R64 and red-off.
# Never a signal to anything. Receipts under /root/spill-receipts/b-day35-review, mirrored to pro-single-day35/box-review/.
set -uo pipefail
R=/root/spill-receipts/b-day35-review; mkdir -p "$R/bins/green" "$R/boots"
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
SHA=34a4b7f23
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/chain.log"; }
cd /root/wt-b || { log "no /root/wt-b"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD >> "$R/fetch.log" 2>&1 \
  || { log "fetch/ff failed"; exit 1; }
git fetch -q origin lane/spill-integ55-20260924 >> "$R/fetch.log" 2>&1 || { log "fetch integ55 failed"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
log "chain start HEAD=$(git rev-parse HEAD) review=$(git rev-parse $SHA)"
have=$(sha256sum "$MODEL" | cut -d' ' -f1)
[ "$have" = "$WANT_MODEL" ] || { log "model sha256 $have is not day 32's $WANT_MODEL; not run"; exit 1; }
W=/root/wt-b35-review
git worktree add --detach "$W" "$SHA" >> "$R/fetch.log" 2>&1 || { log "worktree failed"; exit 1; }
echo "cd $W && CARGO_TARGET_DIR=/root/wt-b/target nice -n 10 cargo build --release -p memra-server" > "$R/bins/green/build-line.txt"
git -C "$W" rev-parse HEAD > "$R/bins/green/build-source.txt"
( cd "$W" && CARGO_TARGET_DIR=/root/wt-b/target nice -n 10 cargo build --release -p memra-server ) > "$R/bins/green/build.log" 2>&1
rc=$?; echo "exit=$rc" >> "$R/bins/green/build.log"
[ $rc = 0 ] || { log "build failed rc=$rc"; exit 1; }
cp /root/wt-b/target/release/memra-server "$R/bins/green/memra-server"   # named memra-server for the cell's stop step
sha256sum "$R/bins/green/memra-server" > "$R/bins/green/binary.sha256"
git worktree remove --force "$W" >> "$R/fetch.log" 2>&1
log "built green from $SHA: $(cut -c1-16 "$R/bins/green/binary.sha256")"
idle() {
  flock -n /tmp/memra-gpu.lock true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
}
export MODEL MODEL_KEY=q38 NO_SCOPE=1 WT=/root/wt-b CLIENT=day33-client.py PARSER=day31-parse.py N=5 \
  RIGDIR="$R/boots" MEMRA_TIMEOUT_MS_MAX=3600000 LOCK=/tmp/memra-gpu.lock
for spec in green-R64-a1:on green-off-a:off green-R64-a2:on; do
  IFS=: read -r name arm <<< "$spec"
  deadline=$((SECONDS + 7200))
  until idle; do [ $SECONDS -ge $deadline ] && { log "boot $name: card not idle after 7200 s; not run"; exit 3; }; sleep 30; done
  log "boot $name start"
  if [ "$arm" = off ]; then
    env -u MEMRA_ADMIT_BY_MEMORY -u MEMRA_ADMIT_OPEN_OUTPUT_TOKENS CLIENT_ARGS="--chars 5000 --skip-long --burst 0" \
      bash research/spill-b-20260919/run-day26-cell.sh "$name" AB "$R/bins/green/memra-server" > "$R/boots/$name.launch.log" 2>&1
  else
    env MEMRA_ADMIT_BY_MEMORY=1 MEMRA_ADMIT_OPEN_OUTPUT_TOKENS=32768 CLIENT_ARGS="--chars 5000 --skip-long --burst 64" \
      bash research/spill-b-20260919/run-day26-cell.sh "$name" AB "$R/bins/green/memra-server" > "$R/boots/$name.launch.log" 2>&1
  fi
  log "boot $name rc=$? $(grep -h '^DAY31 V-BOOT' "$R/boots/$name/REPORT.txt" 2>/dev/null)"
  sleep 5
done
log "chain done"
