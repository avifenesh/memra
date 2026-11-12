#!/usr/bin/env bash
# Laptop, inside a gpulock lease on the RTX 5090 (memra#1173): one run of
# tools/drafter-admission-gate.py from the #1173 worktree against an arm's binary.
# usage: gate-5090.sh NAME ARM DRAFT_DIR|- [extra gate args]
# The gate inherits the lease's legacy flock on FD 9 (--external-lock 9). The power limit is
# recorded, never changed.
set -u
NAME=$1; ARM=$2; DRAFT=$3; shift 3
S=$HOME/.cache/memra-1173
MODEL=${CELL_MODEL:-<models>/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_GPU_LOCK$'); do unset "$v"; done
OUT=$S/run/5090/gates/$NAME
mkdir -p "$(dirname "$OUT")"
LOG=$S/run/5090/gates/$NAME.log
{ date -Is; sha256sum "$S/bin/memra-server-$ARM-120a"
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,power.draw,enforced.power.limit,memory.used --format=csv,noheader
  uptime; } > "$LOG"
DARGS=()
[ "$DRAFT" != "-" ] && DARGS=(--draft "$DRAFT")
setsid python3 "$S/wt/tools/drafter-admission-gate.py" --external-lock 9 --model "$MODEL" "${DARGS[@]}" \
  --bin "$S/bin/memra-server-$ARM-120a" --out "$OUT" --gpu-lock /tmp/memra-5090.lock "$@" >> "$LOG" 2>&1 &
GP=$!
echo "$GP" > "$S/run/5090/gates/$NAME.pgid"
wait "$GP"; rc=$?
# The gate stops its own server and ballast; make sure nothing of its group outlives the lease.
kill -- -"$GP" 2>/dev/null
echo "gate $NAME rc=$rc $(date -Is)" | tee -a "$LOG"
tail -n 8 "$LOG"
exit $rc
