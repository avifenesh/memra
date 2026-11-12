#!/usr/bin/env bash
# Laptop, inside a gpulock lease on the RTX 5090 (memra#1255): one run of
# tools/same-pass-admission-gate.py (from the fix worktree) against an arm's binary.
# usage: gate-5090.sh NAME ARM [extra gate args]. The power limit is recorded, never changed.
set -u
NAME=$1; ARM=$2; shift 2
S=$HOME/.cache/memra-1255
M=$HOME/models/q38-gguf
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_GPU_LOCK$'); do unset "$v"; done
OUT=$S/run/5090/gates/$NAME
mkdir -p "$OUT"
LOG=$OUT/gate.log
{ date -Is; sha256sum "$S/bin/memra-server-$ARM-120a"
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw,enforced.power.limit,memory.used --format=csv,noheader
  uptime; } > "$LOG"
setsid python3 "$S/wt/tools/same-pass-admission-gate.py" --external-lock 9 --model "$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" \
  --draft "$M/Qwen3.8-27B-DFlash2-memra" --bin "$S/bin/memra-server-$ARM-120a" --out "$OUT" "$@" >> "$LOG" 2>&1 &
GP=$!
echo "$GP" >> "$S/pids.txt"
trap 'kill -TERM -- -"$GP" 2>/dev/null; exit 143' TERM INT
wait "$GP"; rc=$?
kill -- -"$GP" 2>/dev/null
echo "gate $NAME rc=$rc $(date -Is)" | tee -a "$LOG"
tail -n 12 "$LOG"
exit $rc
