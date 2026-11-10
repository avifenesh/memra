#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: one gate run of a tool in box/.
# usage: gb10-gate.sh NAME ARM TOOL [tool args after the binary and the model]
set -u
NAME=$1; ARM=$2; TOOL=$3; shift 3
D=$HOME/.cache/memra-carry
MODEL=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
OUT=$D/run/$NAME
if [ -e "$OUT/runs.jsonl" ]; then mv "$OUT" "$OUT.old.$(date +%s)"; fi
mkdir -p "$OUT"
{ date -Is; sha256sum "$D/bin/memra-server-$ARM"; nvidia-smi --query-gpu=name,driver_version,temperature.gpu --format=csv; uptime; } > "$OUT/guard.log"
python3 "$D/box/$TOOL" "$D/bin/memra-server-$ARM" "$MODEL" --out "$OUT" "$@" 2>&1 | tee "$OUT/gate.log"
rc=${PIPESTATUS[0]}
echo "gate $NAME rc=$rc"
exit $rc
