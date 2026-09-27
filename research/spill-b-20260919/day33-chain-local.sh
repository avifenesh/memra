#!/usr/bin/env bash
# WP-B day 33 local chain after the red phase (DAY33.md 1.3): the gate script once on red, the green boots
# (green-G2-r1, green-R32, green-off, green-G2-r2), then the gate script twice on green. G2 is the gate shape (the
# first of G1, G2, R32 whose red boot read R-OOM RED). Each gate run holds /tmp/memra-5090.lock for its one boot
# (flock -w 7200) after an idle wait; the boots go through day33-run.sh. Never a signal to anything.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
R=$WT/research/spill-b-20260919/rtx5090-day33
MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/run.log"; }
idle() {
  flock -n /tmp/memra-5090.lock true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
  [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 24 ] || return 1
}
wait_idle() {
  local deadline=$((SECONDS + 7200))
  until idle; do [ $SECONDS -ge $deadline ] && { log "$1: rig not idle after 7200 s; not run"; exit 3; }; sleep 30; done
}
gate() { # <name> <role>
  wait_idle "gate $1"
  log "gate $1 start role=$2 bin=$(sha256sum "$WT/target/day33/$2/memra-server" | cut -c1-16)"
  AMB_OPEN=8192 AMB_BURST=64 AMB_CTX=65536 flock -w 7200 /tmp/memra-5090.lock \
    tools/admit-mem-burst-gate.sh "$MODEL" "$WT/target/day33/$2/memra-server" "$R/gate/$1" > "$R/gate/$1.log" 2>&1
  log "gate $1 rc=$? $(tail -1 "$R/gate/$1.log")"
}
mkdir -p "$R/gate"
until grep -q "run done: red-G2:red:G2 red-R32:red:R32 red-off:red:off\|red-off: rig not idle\|red-R32: rig not idle" "$R/run.log"; do sleep 60; done
gate red "red"
bash research/spill-b-20260919/day33-run.sh "$R" green-G2-r1:green:G2 green-R32:green:R32 green-off:green:off green-G2-r2:green:G2
gate green-1 green
gate green-2 green
log "local chain done"
