#!/usr/bin/env bash
# DAY37 A1 gate set on one allocator arm (1.6), the lead's integ58 cell list plus the twin gate OFF/ON, every cell with
# MEMRA_KV_ALLOCATOR set for the arm. Run under the collector's hold:
#   python3 tools/tier-battery.py --rig rtx5090 --external-lock --execute bash gates.sh @COLLECTOR_LOCK_FD@ <pooled|vmm> <out_root>
# Gates with an --external-lock arm inherit the FD; the rest run inside the hold as integ58 ran them. serve-smoke builds
# memra-server itself (its own rule), so its cell runs that build and the binary is hashed again after it. After each
# cell the door lines in its logs are counted: an arm whose server logs do not say door=ON (vmm) or door=OFF (pooled)
# is a cell that did not run the arm, and the reader refuses it. Executed-not-qualified.
set -uo pipefail
fd=$1; ARM=$2; ROOT=$3
WT=${WT:-$HOME/projects/wt-spill-b}
MODEL=${MODEL:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}
BIN=${BIN:-$WT/target/day37/lane/memra-server}
cd "$WT" || exit 1
mkdir -p "$ROOT"
RIG_LOCK=${RIG_LOCK:-/tmp/memra-5090.lock}
export MEMRA_GPU_LOCK=$RIG_LOCK
case $ARM in
  pooled) ALLOC="" ;;
  vmm) ALLOC="MEMRA_KV_ALLOCATOR=vmm" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
python3 tools/tier-lock-proof.py --fd "$fd" --lock "$RIG_LOCK" --owner collector > "$ROOT/LOCK.json" 2>&1
TREE=$(git rev-parse HEAD)
sha256sum "$BIN" > "$ROOT/binary.sha256"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ROOT/battery.log"; }
run() { # $1 name $2 env-string $3.. command
  local name=$1 envs="$ALLOC $2"; shift 2
  local OUT=$ROOT/$name; mkdir -p "$OUT"
  nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv > "$OUT/card.before.csv" 2>&1
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/compute-apps.before.csv" 2>&1
  log "start $name env=[$envs]"
  # shellcheck disable=SC2086
  env $envs "$@" > "$OUT/gate.log" 2>&1; local rc=$?
  echo "$rc" > "$OUT/gate.exit"
  nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv > "$OUT/card.after.csv" 2>&1
  local on off
  on=$(grep -rh "\[kv-vmm\] door=ON" "$OUT" 2>/dev/null | wc -l); off=$(grep -rh "\[kv-vmm\] door=OFF" "$OUT" 2>/dev/null | wc -l)
  { echo "cell=$name arm=$ARM cmd=$*"; echo "env=$envs"; echo "lock=$RIG_LOCK owner=collector-fd$fd"
    echo "tree=$TREE"; echo "binary_sha256=$(cut -d' ' -f1 "$ROOT/binary.sha256")"; echo "model=$(basename "$MODEL")"
    echo "door_on_lines=$on door_off_lines=$off"; echo "status=executed-not-qualified"; } > "$OUT/CELL.txt"
  log "done $name rc=$rc door_on=$on door_off=$off $(grep -hE 'GATE: |serve-smoke: |PREFIX-NEWEST-TURN-FITS' "$OUT/gate.log" | tail -1 | cut -c1-220)"
}
run serve-smoke "" bash tools/serve-smoke.sh "$MODEL"
sha256sum "$BIN" > "$ROOT/binary.sha256"
sha256sum "$WT/target/release/memra-server" > "$ROOT/serve-smoke-binary.sha256"
run identity-default-on "MEMRA_HOSTGATE_CACHE_MB=64 MEMRA_KV_HOST_CONTRACTS=1" bash tools/kv-host-spill-identity-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$ROOT/identity-default-on/ev"
run fault-default "MEMRA_HOSTGATE_CACHE_MB=64" bash tools/kv-host-contract-fault-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$ROOT/fault-default/ev"
run fault-plain "MEMRA_HOSTGATE_CACHE_MB=64 MEMRA_SERVE_SPEC=0" bash tools/kv-host-contract-fault-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$ROOT/fault-plain/ev"
run hit-off "" bash tools/spec-on-cache-hit-gate.sh --external-lock "$fd" qwen "$MODEL" "$BIN" "$ROOT/hit-off/ev"
run hit-on "MEMRA_KV_HOST_CONTRACTS=1" bash tools/spec-on-cache-hit-gate.sh --external-lock "$fd" qwen "$MODEL" "$BIN" "$ROOT/hit-on/ev"
run twin-off "" python3 tools/prefix-newest-turn-fits-gate.py --external-lock "$fd" --model "$MODEL" --bin "$BIN" --out "$ROOT/twin-off/ev"
run twin-on "MEMRA_KV_HOST_CONTRACTS=1" python3 tools/prefix-newest-turn-fits-gate.py --external-lock "$fd" --model "$MODEL" --bin "$BIN" --out "$ROOT/twin-on/ev"
run admit-mem-burst "" bash tools/admit-mem-burst-gate.sh "$MODEL" "$BIN" "$ROOT/admit-mem-burst/ev"
run spec-ctx-edge "" bash tools/spec-ctx-edge-gate.sh "$MODEL" "$BIN" "$ROOT/spec-ctx-edge/ev"
log "gates done arm=$ARM"
