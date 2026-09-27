#!/usr/bin/env bash
# WP-B day 34 target-card chain (DAY34.md 1.9), run ON the restored BOX3 once the lead names its host: one RTX PRO 6000
# Blackwell, Qwen3.8-27B NVFP4-Q5K MTP, MEMRA_CTX unset. Refuses to start unless the model's sha256 is day 32's and the
# card is idle. Part A: memra#680's owed boots (DAY33.md 1.3: red-R64, red-off, green-R64, green-off; red = main
# c3eb41d12, green = the fix 30a5ab697). Part B: the day-34 cell, both orders, on the lane tip 9f335ac48. Every binary is
# built once here in a detached worktree. Each boot or order waits (at most 7200 s) for a free /tmp/memra-gpu.lock and no
# compute app; Part B holds the lock per order through the collector. Never a signal to anything.
# Receipts under /root/spill-receipts/b-day34 (part A in p680/), mirrored to pro-single-day34/box/.
# env: MODEL (default /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf), PARTS (default "A B"), ORDER_TIMEOUT (43200).
set -uo pipefail
R=/root/spill-receipts/b-day34; mkdir -p "$R/bins" "$R/p680/boots"
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/chain.log"; }
cd /root/wt-b || { log "no /root/wt-b"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD >> "$R/fetch.log" 2>&1 \
  || { log "fetch/ff failed"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
log "chain start HEAD=$(git rev-parse HEAD) PARTS=${PARTS:-A B}"
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
build() { # <name> <sha>: one binary per commit, copied once
  local W=/root/wt-b34-$1
  git worktree add --detach "$W" "$2" >> "$R/fetch.log" 2>&1 || { log "worktree $1 failed"; exit 1; }
  mkdir -p "$R/bins/$1"
  echo "cd $W && CARGO_TARGET_DIR=/root/wt-b/target nice -n 10 cargo build --release -p memra-server" > "$R/bins/$1/build-line.txt"
  git -C "$W" rev-parse HEAD > "$R/bins/$1/build-source.txt"
  ( cd "$W" && CARGO_TARGET_DIR=/root/wt-b/target nice -n 10 cargo build --release -p memra-server ) > "$R/bins/$1/build.log" 2>&1
  local rc=$?; echo "exit=$rc" >> "$R/bins/$1/build.log"
  [ $rc = 0 ] || { log "build $1 failed rc=$rc"; exit 1; }
  # named memra-server: run-day26-cell.sh stops its own server with `pgrep -x memra-server`
  cp /root/wt-b/target/release/memra-server "$R/bins/$1/memra-server"
  sha256sum "$R/bins/$1/memra-server" > "$R/bins/$1/binary.sha256"
  git worktree remove --force "$W" >> "$R/fetch.log" 2>&1
  log "built $1 from $2: $(cut -c1-16 "$R/bins/$1/binary.sha256")"
}
export MODEL MODEL_KEY=q38 NO_SCOPE=1 WT=/root/wt-b
case " ${PARTS:-A B} " in *" A "*)
  build red c3eb41d1218a288dc325196834c1a8f5233cfb76
  build green 30a5ab697d84ea92210e7910e9cb2c3014503e22
  for spec in red-R64:red:on red-off:red:off green-R64:green:on green-off:green:off; do
    IFS=: read -r name role arm <<< "$spec"
    wait_idle "part A boot $name"
    log "part A boot $name start role=$role"
    if [ "$arm" = off ]; then
      env -u MEMRA_ADMIT_BY_MEMORY -u MEMRA_ADMIT_OPEN_OUTPUT_TOKENS CLIENT=day33-client.py PARSER=day31-parse.py N=5 \
        RIGDIR="$R/p680/boots" MEMRA_TIMEOUT_MS_MAX=3600000 LOCK=/tmp/memra-gpu.lock \
        CLIENT_ARGS="--chars 5000 --skip-long --burst 0" \
        bash research/spill-b-20260919/run-day26-cell.sh "$name" AB "$R/bins/$role/memra-server" > "$R/p680/boots/$name.launch.log" 2>&1
    else
      env MEMRA_ADMIT_BY_MEMORY=1 MEMRA_ADMIT_OPEN_OUTPUT_TOKENS=32768 CLIENT=day33-client.py PARSER=day31-parse.py N=5 \
        RIGDIR="$R/p680/boots" MEMRA_TIMEOUT_MS_MAX=3600000 LOCK=/tmp/memra-gpu.lock \
        CLIENT_ARGS="--chars 5000 --skip-long --burst 64" \
        bash research/spill-b-20260919/run-day26-cell.sh "$name" AB "$R/bins/$role/memra-server" > "$R/p680/boots/$name.launch.log" 2>&1
    fi
    log "part A boot $name rc=$? $(grep -h '^DAY31 V-BOOT' "$R/p680/boots/$name/REPORT.txt" 2>/dev/null)"
    sleep 5
  done ;;
esac
case " ${PARTS:-A B} " in *" B "*)
  build tip 9f335ac4848076fd25fa5d9068bba913ce9c8ad2
  for order in O1 O2; do
    wait_idle "part B order $order"
    log "part B order $order: collector start"
    rm -rf "$R/collector-$order"
    env R="$R" RIG_LOCK=/tmp/memra-gpu.lock BIN="$R/bins/tip/memra-server" BURST=64 \
      python3 tools/tier-battery.py --rig pro-single --timeout "${ORDER_TIMEOUT:-43200}" --out "$R/collector-$order" --external-lock --execute \
      bash research/spill-b-20260919/day31-order.sh @COLLECTOR_LOCK_FD@ $order > "$R/collector-$order.log" 2>&1
    log "part B order $order: collector exit=$?"
  done ;;
esac
log "chain done"
