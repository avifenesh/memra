#!/usr/bin/env bash
# WP-B day 35 local chain (DAY35.md 1.3): red-G2 and red-L64 on main 25bbb91f5; the local red shape is the first of the
# two whose red boot reads R-OOM RED (any CUDA_ERROR_OUT_OF_MEMORY line, park receipts included, or any 503), read by
# day33-compare.py itself; then green-<shape>-r1, red-off, green-off, green-<shape>-r2 on the fix. With no local red the
# green boots run on G2. When the shape is L64 the gate runs once on red and twice on green at L64 (the move of its
# defaults is committed only if that holds). Every boot or gate run waits for an idle rig and holds the lock alone.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
R=$D/rtx5090-day35
MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/run.log"; }
idle() {
  flock -n /tmp/memra-5090.lock true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
  [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 24 ] || return 1
}
wait_idle() {
  local deadline=$((SECONDS + 14400))
  until idle; do [ $SECONDS -ge $deadline ] && { log "$1: rig not idle after 14400 s; not run"; exit 3; }; sleep 30; done
}
red_of() { # <boot>: RED or NOT-RED by day33-compare.py's R-OOM line
  python3 "$D/day33-compare.py" --card rtx5090 "red:X:$R/boots/$1" 2>/dev/null | grep -q "R-OOM .* -> RED$" && echo RED || echo NOT-RED
}
bash "$D/day35-run.sh" "$R" red-G2:red:G2 red-L64:red:L64 || exit $?
shape=""
for s in G2 L64; do [ -z "$shape" ] && [ "$(red_of red-$s)" = RED ] && shape=$s; done
log "local red shape: ${shape:-none} (red-G2 $(red_of red-G2), red-L64 $(red_of red-L64))"
gs=${shape:-G2}
bash "$D/day35-run.sh" "$R" "green-$gs-r1:green:$gs" red-off:red:off green-off:green:off "green-$gs-r2:green:$gs" || exit $?
if [ "$shape" = L64 ]; then
  mkdir -p "$R/gate"
  for run in red:red green-1:green green-2:green; do
    name=${run%%:*}; role=${run#*:}
    wait_idle "gate $name"
    log "gate $name start role=$role shape=L64"
    AMB_OPEN=32768 AMB_BURST=64 AMB_CTX=65536 flock -w 7200 /tmp/memra-5090.lock \
      tools/admit-mem-burst-gate.sh "$MODEL" "$WT/target/day35/$role/memra-server" "$R/gate/$name" > "$R/gate/$name.log" 2>&1
    log "gate $name rc=$? $(tail -1 "$R/gate/$name.log")"
  done
fi
log "local chain done"
