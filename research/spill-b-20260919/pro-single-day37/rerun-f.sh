#!/usr/bin/env bash
# WP-B DAY37 addendum F target-card rerun (DAY37.md 1.15): one RTX PRO 6000 Blackwell Workstation Edition, the 27B,
# MEMRA_CTX unset. The r4 lane source (c6f9282c2) and main 17dceb981 built in detached worktrees; stage 0 again as a
# recorded reading; MEMRA_KV_VMM_GROW=inline pinned from the class's first stage-0 receipt; the gate set of both arms
# (gates-r4b-*); the boots off-lane, fault-mapper, off-main, burst-g2-vmm, fault-ensure (boots-f.sh); the corrected
# reader. Refuses to start without ss or lsof. Never a signal to anything this lane did not start. Receipts under
# /root/spill-receipts/b-day37f, mirrored to pro-single-day37/box-f/.
set -uo pipefail
R=/root/spill-receipts/b-day37f; mkdir -p "$R/bins" "$R/boots"
WT=/root/wt-b
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
MAIN_SHA=17dceb981
LANE_SHA=${LANE_SHA:-c6f9282c26e71349c0263b88dc7beb59c4a57797}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/chain.log"; }
cd "$WT" || { log "no $WT"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD >> "$R/fetch.log" 2>&1 || { log "fetch/ff failed"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
command -v ss >/dev/null || command -v lsof >/dev/null || { log "neither ss nor lsof on PATH: the gates cannot check their ports; not run"; exit 1; }
[ "$(sha256sum "$MODEL" | cut -d' ' -f1)" = "$WANT_MODEL" ] || { log "model sha256 mismatch; not run"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,memory.total --format=csv > "$R/card.csv"
command grep -q "RTX PRO 6000 Blackwell Workstation" "$R/card.csv" || { log "not an RTX PRO 6000 Blackwell Workstation: $(tail -1 "$R/card.csv")"; exit 1; }
[ "${DRY_RUN:-0}" = 1 ] && { log "dry run done"; exit 0; }
idle() { flock -n /tmp/memra-gpu.lock true || return 1; [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1; }
wait_idle() { local deadline=$((SECONDS + 7200)); until idle; do [ $SECONDS -ge $deadline ] && { log "$1: card not idle after 7200 s; not run"; exit 3; }; sleep 30; done; }
if [ ! -x "$R/bins/lane/memra-server" ]; then
  W=/root/wt-b37f-lane; mkdir -p "$R/bins/lane"
  git worktree add --detach "$W" "$LANE_SHA" >> "$R/fetch.log" 2>&1 || { log "lane worktree failed"; exit 1; }
  ( cd "$W" && CARGO_TARGET_DIR=/root/wt-b/target nice -n 10 cargo build --release -p memra-server --bin memra-server \
      -p memra-engine --bin vmm-call-cost ) > "$R/bins/lane/build.log" 2>&1 || { log "lane build failed"; exit 1; }
  cp /root/wt-b/target/release/memra-server "$R/bins/lane/memra-server"; cp /root/wt-b/target/release/vmm-call-cost "$R/bins/"
  git -C "$W" rev-parse HEAD > "$R/bins/lane/source.commit"
  git worktree remove --force "$W" >> "$R/fetch.log" 2>&1
fi
if [ ! -x "$R/bins/main/memra-server" ]; then
  W=/root/wt-b37f-main; mkdir -p "$R/bins/main"
  git worktree add --detach "$W" "$MAIN_SHA" >> "$R/fetch.log" 2>&1 || { log "main worktree failed"; exit 1; }
  ( cd "$W" && CARGO_TARGET_DIR=/root/target-b37f-main nice -n 10 cargo build --release -p memra-server --bin memra-server ) \
    > "$R/bins/main/build.log" 2>&1 || { log "main build failed"; exit 1; }
  cp /root/target-b37f-main/release/memra-server "$R/bins/main/memra-server"
  git -C "$W" rev-parse HEAD > "$R/bins/main/source.commit"
  git worktree remove --force "$W" >> "$R/fetch.log" 2>&1; rm -rf /root/target-b37f-main
fi
sha256sum "$R/bins/lane/memra-server" "$R/bins/main/memra-server" "$R/bins/vmm-call-cost" > "$R/bins/SHA256SUMS"
log "chain start HEAD=$(cat "$R/source.txt") lane=$(cat "$R/bins/lane/source.commit") $(tr '\n' ' ' < "$R/bins/SHA256SUMS" | cut -c1-200)"
if [ ! -s "$R/stage0/probe.log" ]; then
  wait_idle "stage 0"; mkdir -p "$R/stage0"
  python3 tools/tier-battery.py --rig pro-single --timeout 1800 --out "$R/stage0/collector" --execute \
    bash -c "'$R/bins/vmm-call-cost' --reps 20 --queue-ms 60 --extents 1,4,16,64 | tee '$R/stage0/probe.log'" > "$R/stage0/collector.log" 2>&1
  log "stage 0 (a reading) rc=$? $(command grep -h '^GROW-PLACEMENT' "$R/stage0/probe.log" | tr '\n' ' ')"
fi
export MEMRA_KV_VMM_GROW=inline
export WT RIG_LOCK=/tmp/memra-gpu.lock MODEL MODEL_TWIN=$MODEL BIN="$R/bins/lane/memra-server" HOSTGATE_MB=256
for arm in pooled vmm; do
  out=$R/gates-r4b-$arm
  if [ -e "$out/battery.log" ] && command grep -q "gates done arm=$arm" "$out/battery.log"; then continue; fi
  wait_idle "gates $arm"; mkdir -p "$out"
  python3 tools/tier-battery.py --rig pro-single --timeout 10800 --out "$out/collector" --external-lock --execute \
    bash research/spill-b-20260919/rtx5090-day37/gates.sh @COLLECTOR_LOCK_FD@ "$arm" "$out" > "$out/collector.log" 2>&1
  log "gates $arm rc=$?"
done
export LANE_BIN="$R/bins/lane/memra-server" MAIN_BIN="$R/bins/main/memra-server" MODEL_KEY=q38 BOOT_CTX='' STREAM_CONC=16 NO_SCOPE=1
bash research/spill-b-20260919/rtx5090-day37/boots-f.sh "$R" off-lane:pooled:mixspec fault-mapper:vmm-mapperfault:mixspec \
  off-main:main:mixspec burst-g2-vmm:vmm:g2 fault-ensure:vmm-ensurefault:g2
python3 research/spill-b-20260919/day37-read.py pro6000 "$R" gates-r4b > "$R/read.log" 2>&1
( cd "$R" && find . -type f ! -path './bins/*' -print0 | sort -z | xargs -0 sha256sum > "$R/MANIFEST.sha256" )
log "LANE-B-DAY37F-BOX-DONE"
