#!/usr/bin/env bash
# DAY68 section 1: one owed 5090 half (r1 or l2) on the local RTX 5090 Laptop GPU, fixed before it runs.
#   rtx5090-half.sh prepare <half>  from the lane worktree: the scratch tree at the half's tip under
#                                   /home/avifenesh/spill-a-cells/<half>/tree, the derived scripts and this file copied
#                                   to <half>/scripts (the frozen copy), then the derived build.sh (outside any hold).
#   rtx5090-half.sh card <half>     from the frozen copy only: ONE bounded hold of /tmp/memra-5090.lock (180 x 120 s
#                                   behind the other lanes; then no compute app and >= 20000 MiB free, 15 x 60 s;
#                                   another project's process is waited out, never touched), the target sitting's
#                                   cells in its order through fd 9, each under the collector's timeout; the host load
#                                   and compute apps at each cell's start and a 5 s host-load log through the hold; the
#                                   hold released; then the tip tree's reader.
#   rtx5090-half.sh clean <half>    from the lane worktree, after the receipts are banked: the scratch tree and
#                                   target dir (the bins stay until the lane closes).
# Executed-not-qualified. No host, id or price here.
set -uo pipefail
ACT=$1; HALF=$2
S=/home/avifenesh/spill-a-cells/$HALF
case $HALF in
  r1) TIP=15d7ed351; BASE=40821db01 ;;
  l2) TIP=21984b527; BASE=217ace3fd ;;
  *) echo "half $HALF"; exit 2 ;;
esac
MODEL=/home/avifenesh/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
export A_OUT=$S/receipts A_TREE=$S/tree CARGO_TARGET_DIR=$S/target MEMRA_DAY38_MODEL=$MODEL
case $ACT in
prepare)
  HERE=$(cd "$(dirname "$0")/../.." && pwd)
  [ -e "$HERE/.git" ] || { echo "run prepare from the lane worktree's copy"; exit 2; }
  [ -e "$S" ] && { echo "$S exists"; exit 2; }
  mkdir -p "$S/scripts" "$A_OUT"
  cp "$HERE/research/spill-a-20260919/rtx5090-$HALF/"*.sh "$HERE/research/spill-a-20260919/rtx5090-half.sh" "$S/scripts/"
  git -C "$HERE" rev-parse HEAD > "$S/scripts/lane-tree.sha"
  git -C "$HERE" worktree add -q --detach "$A_TREE" "$TIP" || { echo "rc=2 (worktree)" >> "$A_OUT/build.log"; exit 2; }
  bash "$S/scripts/build.sh" "$TIP" "$BASE" > "$S/build.out" 2>&1
  echo "$HALF build: $(tail -1 "$A_OUT/build.log")"
  ;;
card)
  [ "$(cd "$(dirname "$0")" && pwd)" = "$S/scripts" ] || { echo "run the frozen copy: $S/scripts/rtx5090-half.sh"; exit 2; }
  cd "$A_TREE" || exit 1
  grep -q '^rc=0$' "$A_OUT/build.log" 2>/dev/null || { echo "build receipt missing"; exit 2; }
  [ "$(git rev-parse HEAD)" = "$(git rev-parse "$TIP")" ] && [ -z "$(git status --porcelain --untracked-files=no)" ] \
    || { echo "the tree is not the clean tip"; exit 2; }
  log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$A_OUT/run.log"; }
  log "model sha256 (outside the hold): $(sha256sum "$MODEL" | cut -d' ' -f1) $(basename "$MODEL")"
  exec 9>/tmp/memra-5090.lock
  held=0
  for attempt in $(seq 1 180); do
    if flock -n 9; then held=1; break; fi
    log "lock busy, attempt $attempt of 180, waiting 120 s"; sleep 120
  done
  [ "$held" = 1 ] || { log "NOT RUN: the 5090 lock stayed busy"; exit 2; }
  free_ok=0
  for attempt in $(seq 1 15); do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>&1 | head -1)
    if [ -z "$apps" ] && [ "${free:-0}" -ge 20000 ] 2>/dev/null; then free_ok=1; break; fi
    log "card not idle under the hold (apps=[${apps//$'\n'/; }] free=${free} MiB), attempt $attempt of 15"; sleep 60
  done
  [ "$free_ok" = 1 ] || { log "NOT RUN: the card never went idle under the hold"; exit 2; }
  log "hold taken"
  ( while :; do echo "$(date -u +%FT%TZ) $(cut -d' ' -f1-4 /proc/loadavg)"; sleep 5; done ) > "$A_OUT/host-load-5s.log" 2>&1 &
  LOADER=$!
  trap 'kill "$LOADER" 2>/dev/null || true' EXIT
  cellrun() { # name timeout script args...
    local name=$1 to=$2; shift 2
    log "$name start: host load $(cut -d' ' -f1-3 /proc/loadavg) compute apps [$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 | tr '\n' ';')]"
    timeout "$to" bash "$@" > "$A_OUT/$name.out" 2>&1
    log "$name rc=$?"
  }
  if [ "$HALF" = r1 ]; then
    cellrun gates-cell 10800 "$S/scripts/gates.sh" 9
    cellrun ab-r1-cell 21600 "$S/scripts/ab.sh" 9 seam retire-seam-nosource retire-seam prime 448
  else
    cellrun unit-cell 3600 "$S/scripts/unit-cells.sh" 9
    cellrun gates-cell 10800 "$S/scripts/gates.sh" 9
    cellrun gates-red-cell 3600 "$S/scripts/gates-red.sh" 9
    cellrun ab-chain-cell 10800 "$S/scripts/ab.sh" 9 chain
    cellrun ab-demote-cell 10800 "$S/scripts/ab.sh" 9 demote
  fi
  kill "$LOADER" 2>/dev/null || true
  flock -u 9; exec 9>&-
  log "hold released"
  python3 "research/spill-a-20260919/$HALF-reading.py" "$A_OUT" > "$A_OUT/reading-$HALF.log" 2>&1
  log "reading rc=$? $(tail -1 "$A_OUT/reading-$HALF.log")"
  ;;
clean)
  HERE=$(cd "$(dirname "$0")/../.." && pwd)
  [ -e "$HERE/.git" ] || { echo "run clean from the lane worktree's copy"; exit 2; }
  git -C "$HERE" worktree remove --force "$A_TREE" && rm -rf "$CARGO_TARGET_DIR"
  ;;
*) echo "action $ACT"; exit 2 ;;
esac
