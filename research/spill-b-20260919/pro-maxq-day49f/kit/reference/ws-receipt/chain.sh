#!/usr/bin/env bash
# WP-B DAY49 addendum D target-card half: one RTX PRO 6000 Blackwell Workstation Edition, the 27B. Two detached
# worktrees built first: green at D (the site-aimed fault, arm j) and red at D with day49d-noreap.patch (the batch-OOM
# reap reverted). Then, each under the box lock: the gate's arm j twice from green (j1, j2); arm j under
# MEMRA_KV_ALLOCATOR=vmm from green (vmm-green) and from red (vmm-red); the serving shape (day49d-run.sh, burst 8 x
# 6,144, no warm, MEMRA_STEP_OOM_FAULT=batch:1) on green's binary, spec default then MEMRA_SERVE_SPEC=0, off and on in
# both orders; then day49d-read.py. Refuses to start without ss or lsof. Never a signal to anything this lane did not
# start. Every step's exit is captured on its own line before it is logged. Receipts /root/spill-receipts/b-day49d.
# BOOTS_ONLY=1 (the rerun after the fourteenth sitting's boots did not start: day49d-run.sh got no WT): builds green
# only and runs the serving boots and their reader, receipts /root/spill-receipts/b-day49d-boots.
set -uo pipefail
BOOTS_ONLY=${BOOTS_ONLY:-0}
R=/root/spill-receipts/b-day49d; [ "$BOOTS_ONLY" = 1 ] && R=/root/spill-receipts/b-day49d-boots
mkdir -p "$R"
# Exported: day49d-run.sh reads WT and RIG_LOCK from the environment (its defaults are the local rig's).
export WT=/root/wt-b RIG_LOCK=/tmp/memra-gpu.lock
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
SD=${SD:-8926ccfb3e8a27cf8f91aa747e2b088251215c7a}
PATCH=research/spill-b-20260919/day49d-noreap.patch
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/chain.log"; }
cd "$WT" || { log "no $WT"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD >> "$R/fetch.log" 2>&1 || { log "fetch/ff failed"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
command -v ss >/dev/null || command -v lsof >/dev/null || { log "neither ss nor lsof on PATH; not run"; exit 1; }
[ "$(sha256sum "$MODEL" | cut -d' ' -f1)" = "$WANT_MODEL" ] || { log "model sha256 mismatch; not run"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,memory.total --format=csv > "$R/card.csv"
command grep -q "RTX PRO 6000 Blackwell Workstation" "$R/card.csv" || { log "not an RTX PRO 6000 Blackwell Workstation: $(tail -1 "$R/card.csv")"; exit 1; }
git cat-file -e "$SD^{commit}" || { log "no commit $SD; not run"; exit 1; }
[ "${DRY_RUN:-0}" = 1 ] && { log "dry run done"; exit 0; }
G=$WT/target/wt-day49d-green; RD=$WT/target/wt-day49d-red
: > "$R/binaries.sha256"
SIDES="green red"; [ "$BOOTS_ONLY" = 1 ] && SIDES=green
for side in $SIDES; do
  T=$G; [ "$side" = red ] && T=$RD
  [ -d "$T" ] || git worktree add --detach "$T" "$SD" > "$R/$side-worktree.log" 2>&1 || { log "$side: worktree failed"; exit 1; }
  if [ "$side" = red ]; then git -C "$T" apply "$WT/$PATCH" || { log "red: patch does not apply"; exit 1; }; sha256sum "$PATCH" > "$R/red.patch.sha256"; fi
  ( cd "$T" && nice -n 10 cargo build --release -p memra-server ) > "$R/$side-build.log" 2>&1
  rc=$?
  [ $rc = 0 ] || { log "$side: build failed rc=$rc"; exit 1; }
  echo "$SD $(sha256sum "$T/target/release/memra-server" | cut -d' ' -f1) $side" >> "$R/binaries.sha256"
done
log "chain start HEAD=$(cat "$R/source.txt") $(tr '\n' ' ' < "$R/binaries.sha256" | cut -c1-240)"
gate() { # <label> <tree> <env...>
  local label=$1 tree=$2; shift 2
  env "$@" HFG_ARMS=j HFG_OUT="$R/$label" HFG_READY_WAIT_S=900 flock -w 7200 /tmp/memra-gpu.lock "$tree/tools/health-fault-gate.sh" "$MODEL" \
    > "$R/$label.log" 2>&1
  local rc=$?
  log "$label rc=$rc $(tail -1 "$R/$label.log")"
}
if [ "$BOOTS_ONLY" != 1 ]; then
  gate j1 "$G"
  gate j2 "$G"
  gate vmm-green "$G" MEMRA_KV_ALLOCATOR=vmm
  gate vmm-red "$RD" MEMRA_KV_ALLOCATOR=vmm
fi
[ -d "$WT/research/spill-b-20260919" ] || { log "no lane tree at WT=$WT; boots not run"; exit 1; }
export BIN="$G/target/release/memra-server" MODEL MODEL_KEY=q38 BOOT_CTX='' NO_SCOPE=1 \
  FAULT=batch:1 CLIENT_EXTRA="--warm-n 0 --burst 8 --length 6144 --max-tokens 64"
bash research/spill-b-20260919/day49d-run.sh "$R" O1-off:off O1-on:on O2-on:on O2-off:off \
  P1-off:off-plain P1-on:on-plain P2-on:on-plain P2-off:off-plain
rc=$?
[ $rc = 0 ] || log "boots stopped rc=$rc"
python3 research/spill-b-20260919/day49d-read.py serve pro6000 "$R" > "$R/read-serve.log" 2>&1
[ "$BOOTS_ONLY" = 1 ] || python3 research/spill-b-20260919/day49d-read.py vmm pro6000 "$R/vmm-green" "$R/vmm-red" > "$R/read-vmm.log" 2>&1
for T in "$G" "$RD"; do [ -d "$T" ] && git worktree remove --force "$T" >> "$R/chain.log" 2>&1; done
( cd "$R" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > "$R/MANIFEST.sha256" )
if [ "$BOOTS_ONLY" = 1 ]; then log "LANE-B-DAY49D-BOOTS-DONE"; else log "LANE-B-DAY49D-BOX-DONE"; fi
