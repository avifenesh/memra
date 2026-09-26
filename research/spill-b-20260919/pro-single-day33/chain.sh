#!/usr/bin/env bash
# WP-B day 33 target-card chain (DAY33.md 1.3): one RTX PRO 6000 Blackwell, Qwen3.8-27B NVFP4-Q5K MTP, MEMRA_CTX
# unset. Started only after lane A's LANE-A-PRO-DONE. Builds red (main c3eb41d12) and green (the fix, 30a5ab697) in
# detached worktrees, then runs red-R64, red-off, green-R64, green-off, each under /tmp/memra-gpu.lock (the cell takes
# it with flock -w 3600) after a bounded idle wait (lock free, no compute app). Never a signal to anything.
# Receipts under /root/spill-receipts/b-day33, mirrored to pro-single-day33/box/.
set -uo pipefail
R=/root/spill-receipts/b-day33; mkdir -p "$R/boots"
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/chain.log"; }
cd /root/wt-b || exit 1
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD >> "$R/fetch.log" 2>&1 \
  || { log "fetch/ff failed"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
log "chain start HEAD=$(git rev-parse HEAD)"
for pair in red:c3eb41d1218a288dc325196834c1a8f5233cfb76 green:30a5ab697d84ea92210e7910e9cb2c3014503e22; do
  role=${pair%%:*}; sha=${pair#*:}; W=/root/wt-b33-$role
  git worktree add --detach "$W" "$sha" >> "$R/fetch.log" 2>&1 || { log "worktree $role failed"; exit 1; }
  mkdir -p "$R/$role"
  echo "cd $W && CARGO_TARGET_DIR=/root/wt-b/target nice -n 10 cargo build --release -p memra-server" > "$R/$role/build-line.txt"
  git -C "$W" rev-parse HEAD > "$R/$role/build-source.txt"
  ( cd "$W" && CARGO_TARGET_DIR=/root/wt-b/target nice -n 10 cargo build --release -p memra-server ) > "$R/$role/build.log" 2>&1
  rc=$?; echo "exit=$rc" >> "$R/$role/build.log"
  [ $rc = 0 ] || { log "build $role failed rc=$rc"; exit 1; }
  cp /root/wt-b/target/release/memra-server "$R/$role/memra-server"
  sha256sum "$R/$role/memra-server" > "$R/$role/binary.sha256"
  log "built $role from $sha: $(cut -c1-16 "$R/$role/binary.sha256")"
done
idle() {
  flock -n /tmp/memra-gpu.lock true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
}
export CLIENT=day33-client.py PARSER=day31-parse.py N=5 RIGDIR="$R/boots" MEMRA_TIMEOUT_MS_MAX=3600000 \
  LOCK=/tmp/memra-gpu.lock NO_SCOPE=1 WT=/root/wt-b \
  MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf MODEL_KEY=q38
for spec in red-R64:red:on red-off:red:off green-R64:green:on green-off:green:off; do
  IFS=: read -r name role arm <<< "$spec"
  deadline=$((SECONDS + 7200))
  until idle; do [ $SECONDS -ge $deadline ] && { log "boot $name: card not idle after 7200 s; not run"; exit 3; }; sleep 30; done
  log "boot $name start role=$role"
  if [ "$arm" = off ]; then
    env -u MEMRA_ADMIT_BY_MEMORY -u MEMRA_ADMIT_OPEN_OUTPUT_TOKENS CLIENT_ARGS="--chars 5000 --skip-long --burst 0" \
      bash research/spill-b-20260919/run-day26-cell.sh "$name" AB "$R/$role/memra-server" > "$R/boots/$name.launch.log" 2>&1
  else
    env MEMRA_ADMIT_BY_MEMORY=1 MEMRA_ADMIT_OPEN_OUTPUT_TOKENS=32768 CLIENT_ARGS="--chars 5000 --skip-long --burst 64" \
      bash research/spill-b-20260919/run-day26-cell.sh "$name" AB "$R/$role/memra-server" > "$R/boots/$name.launch.log" 2>&1
  fi
  log "boot $name rc=$? $(grep -h '^DAY31 V-BOOT' "$R/boots/$name/REPORT.txt" 2>/dev/null)"
  sleep 5
done
for role in red green; do git worktree remove --force "/root/wt-b33-$role" >> "$R/fetch.log" 2>&1; done
log "chain done"
