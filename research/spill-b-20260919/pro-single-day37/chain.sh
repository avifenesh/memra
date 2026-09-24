#!/usr/bin/env bash
# WP-B day 37 target-card sitting (DAY37.md 1.8 step 6, addendum B): one RTX PRO 6000 Blackwell Workstation Edition,
# Qwen3.8-27B NVFP4-Q5K MTP (sha256 below), MEMRA_CTX unset (the checkpoint's 262,144). In order, each GPU step under
# /tmp/memra-gpu.lock (the collector or run-day26-cell.sh takes it) after a bounded idle wait:
#   0. fetch the lane tip (fast-forward only), refuse unless the model's sha256 is day 32's;
#   1. build the lane's memra-server, kv-tier-gate and vmm-call-cost, and main 17dceb981's memra-server (detached);
#   2. stage 0: vmm-call-cost through the collector; its GROW-PLACEMENT line selects MEMRA_KV_VMM_GROW for this sitting
#      (the measurement seam; the class default in code moves only on this receipt, after the sitting);
#   3. A2: the grow series (run-grow.sh pro-single);
#   4. A1: the gate set per arm under one collector hold each (gates.sh);
#   5. the serving boots (boots.sh): mix both paths both orders, A5 faults, A6 main, A7 bursts, A3 stream pairs;
#   6. the reader (day37-read.py pro6000).
# Never a signal to anything this lane did not start. Receipts under /root/spill-receipts/b-day37 (mirrored to
# pro-single-day37/box/ by the lane after the sitting, sha256-checked against a box manifest).
set -uo pipefail
R=/root/spill-receipts/b-day37; mkdir -p "$R/bins" "$R/boots"
WT=/root/wt-b
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
MAIN_SHA=17dceb981
LANE_SHA=${LANE_SHA:-c6f9282c26e71349c0263b88dc7beb59c4a57797}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/chain.log"; }
cd "$WT" || { log "no $WT"; exit 1; }
if [ "${DRY_RUN:-0}" = 1 ]; then log "dry run: the checks only"; fi
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD >> "$R/fetch.log" 2>&1 \
  || { log "fetch/ff failed"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
log "chain start HEAD=$(git rev-parse HEAD)"
have=$(sha256sum "$MODEL" | cut -d' ' -f1)
[ "$have" = "$WANT_MODEL" ] || { log "model sha256 $have is not day 32's $WANT_MODEL; not run"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,memory.total --format=csv > "$R/card.csv"
command grep -q "RTX PRO 6000 Blackwell" "$R/card.csv" || { log "not an RTX PRO 6000 Blackwell: $(tail -1 "$R/card.csv")"; exit 1; }
[ "${DRY_RUN:-0}" = 1 ] && { log "dry run done"; exit 0; }
idle() {
  flock -n /tmp/memra-gpu.lock true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
}
wait_idle() {
  local deadline=$((SECONDS + 7200))
  until idle; do [ $SECONDS -ge $deadline ] && { log "$1: card not idle after 7200 s; not run"; exit 3; }; sleep 30; done
}
# 1. builds
# The deciding cell's binaries are the 5090's r4 source (addenda C to E), pinned: the scripts run from the tip.
if [ ! -x "$R/bins/lane/memra-server" ]; then
  W=/root/wt-b37-lane; mkdir -p "$R/bins/lane"
  git worktree add --detach "$W" "$LANE_SHA" >> "$R/fetch.log" 2>&1 || { log "lane worktree failed"; exit 1; }
  ( cd "$W" && CARGO_TARGET_DIR=/root/wt-b/target nice -n 10 cargo build --release -p memra-server --bin memra-server \
      -p memra-engine --bin kv-tier-gate --bin vmm-call-cost ) > "$R/bins/lane/build.log" 2>&1 || { log "lane build failed"; exit 1; }
  cp /root/wt-b/target/release/memra-server "$R/bins/lane/memra-server"
  cp /root/wt-b/target/release/kv-tier-gate /root/wt-b/target/release/vmm-call-cost "$R/bins/"
  git -C "$W" rev-parse HEAD > "$R/bins/lane/build-source.txt"
  cp "$R/bins/lane/build-source.txt" "$R/bins/lane/source.commit"   # gates.sh runs serve-smoke at this source
  git worktree remove --force "$W" >> "$R/fetch.log" 2>&1
  log "built lane from $(cat "$R/bins/lane/build-source.txt")"
fi
if [ ! -x "$R/bins/main/memra-server" ]; then
  W=/root/wt-b37-main; mkdir -p "$R/bins/main"
  git worktree add --detach "$W" "$MAIN_SHA" >> "$R/fetch.log" 2>&1 || { log "main worktree failed"; exit 1; }
  ( cd "$W" && CARGO_TARGET_DIR=/root/target-b37-main nice -n 10 cargo build --release -p memra-server --bin memra-server ) \
    > "$R/bins/main/build.log" 2>&1 || { log "main build failed"; exit 1; }
  cp /root/target-b37-main/release/memra-server "$R/bins/main/memra-server"
  git -C "$W" rev-parse HEAD > "$R/bins/main/build-source.txt"
  git worktree remove --force "$W" >> "$R/fetch.log" 2>&1; rm -rf /root/target-b37-main
  log "built main from $(cat "$R/bins/main/build-source.txt")"
fi
sha256sum "$R/bins/lane/memra-server" "$R/bins/main/memra-server" "$R/bins/kv-tier-gate" "$R/bins/vmm-call-cost" > "$R/bins/SHA256SUMS"
# 2. stage 0
if [ ! -s "$R/stage0/probe.log" ]; then
  wait_idle "stage 0"; mkdir -p "$R/stage0"
  python3 tools/tier-battery.py --rig pro-single --timeout 1800 --out "$R/stage0/collector" --execute \
    bash -c "'$R/bins/vmm-call-cost' --reps 20 --queue-ms 60 --extents 1,4,16,64 | tee '$R/stage0/probe.log'" > "$R/stage0/collector.log" 2>&1
  log "stage 0 rc=$? $(command grep -h '^GROW-PLACEMENT\|^RELEASE-PLACEMENT' "$R/stage0/probe.log" | tr '\n' ' ')"
fi
placement=$(command grep -h '^GROW-PLACEMENT' "$R/stage0/probe.log" | sed 's/.*-> //')
[ "$placement" = helper ] || [ "$placement" = inline ] || { log "stage 0 printed no placement; not run"; exit 1; }
export MEMRA_KV_VMM_GROW=$placement
echo "$placement" > "$R/stage0/PLACEMENT"
log "placement for this sitting: MEMRA_KV_VMM_GROW=$placement (stage 0's rule)"
# 3. A2
if [ ! -e "$R/grow-32768-r4/collector.exit" ]; then
  wait_idle "A2"
  env -u MEMRA_KV_VMM_GROW WT="$WT" ROOT="$R" GATE_BIN="$R/bins/kv-tier-gate" ARTIFACT="$MODEL" \
    bash research/spill-b-20260919/rtx5090-day37/run-grow.sh pro-single grow-32768-r4 > "$R/grow.log" 2>&1
  log "A2 rc=$? $(head -1 "$R/grow-32768-r4/receipt/GROW.txt" 2>/dev/null)"
fi
# 4. A1 gates
export WT RIG_LOCK=/tmp/memra-gpu.lock MODEL MODEL_TWIN=$MODEL BIN="$R/bins/lane/memra-server" HOSTGATE_MB=256
for arm in pooled vmm; do
  out=$R/gates-r4-$arm
  if [ -e "$out/battery.log" ] && command grep -q "gates done arm=$arm" "$out/battery.log"; then continue; fi
  wait_idle "gates $arm"; mkdir -p "$out"
  python3 tools/tier-battery.py --rig pro-single --timeout 10800 --out "$out/collector" --external-lock --execute \
    bash research/spill-b-20260919/rtx5090-day37/gates.sh @COLLECTOR_LOCK_FD@ "$arm" "$out" > "$out/collector.log" 2>&1
  log "gates $arm rc=$?"
done
# 5. serving boots
export LANE_BIN="$R/bins/lane/memra-server" MAIN_BIN="$R/bins/main/memra-server" MODEL_KEY=q38 BOOT_CTX='' STREAM_CONC=16 NO_SCOPE=1
B=research/spill-b-20260919/rtx5090-day37/boots.sh
bash $B "$R" \
  mix-spec-O1-pooled:pooled:mixspec mix-spec-O1-vmm:vmm:mixspec mix-plain-O1-pooled:pooled:mixplain mix-plain-O1-vmm:vmm:mixplain \
  mix-spec-O2-vmm:vmm:mixspec mix-spec-O2-pooled:pooled:mixspec mix-plain-O2-vmm:vmm:mixplain mix-plain-O2-pooled:pooled:mixplain \
  fault-mapper:vmm-mapperfault:mixspec off-main:main:mixspec \
  burst-g2-pooled:pooled:g2 burst-g2-vmm:vmm:g2 burst-l64-vmm:vmm:l64 burst-l64-pooled:pooled:l64 burst-boff-pooled:pooled:boff burst-boff-vmm:vmm:boff \
  fault-ensure:vmm-ensurefault:g2 fault-build1:vmm-buildfault1:g2 fault-build64:vmm-buildfault64:g2
args=()
for k in 1 2 3 4 5; do args+=("stream-O1-$k-pooled:pooled:stream" "stream-O1-$k-vmm:vmm:stream"); done
for k in 1 2 3 4 5; do args+=("stream-O2-$k-vmm:vmm:stream" "stream-O2-$k-pooled:pooled:stream"); done
bash $B "$R" "${args[@]}"
# 6. reader
python3 research/spill-b-20260919/day37-read.py pro6000 "$R" gates-r4 > "$R/read.log" 2>&1
( cd "$R" && find . -type f ! -path './bins/*' -print0 | sort -z | xargs -0 sha256sum > "$R/MANIFEST.sha256" )
log "LANE-B-DAY37-BOX-DONE"
