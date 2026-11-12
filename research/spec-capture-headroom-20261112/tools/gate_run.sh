#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: one exactness gate of memra#1221 on one binary, in
# its own process group, stopped (with every server it booted) when it outlives CAP_MIN.
# KIND: inv (tools/seeded-invariance-gate.py --penalised device) or bnd (tools/spec-burst-boundary-gate.py).
# usage: gate_run.sh KIND BIN_DIR OUT_NAME [CAP_MIN]   -> runs/gates/OUT_NAME{,.log}; exit 0 on PASS
set -u
D=$HOME/.cache/memra-1221; cd "$D" || exit 1
KIND=$1; BIN=$D/bins/$2/memra-server; OUT=$D/runs/gates/$3; CAP=${4:-27}
M=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
mkdir -p "$D/runs/gates"
case $KIND in
  inv) set -- python3 box/seeded-invariance-gate.py "$BIN" "$M" --out "$OUT" --penalised device; PASS='SEEDED INVARIANCE: PASS' ;;
  bnd) set -- python3 box/spec-burst-boundary-gate.py "$BIN" "$M" --out "$OUT"; PASS='SPEC BURST BOUNDARY: PASS' ;;
  *) echo "unknown kind $KIND"; exit 2 ;;
esac
{ date -Is; sha256sum "$BIN" | sed "s|$HOME|~|"; nvidia-smi --query-gpu=name,driver_version,temperature.gpu --format=csv,noheader; uptime; } > "$OUT.guard"
setsid "$@" > "$OUT.log" 2>&1 < /dev/null &
p=$!
if ! waitfor --gone "$p" --timeout "${CAP}m" --quiet; then
  echo "$(date -Is) gate_run: cap ${CAP} min reached; stopping group $p" >> "$OUT.log"
  kill -TERM -"$p"; waitfor --gone "$p" --timeout 1m --quiet || kill -KILL -"$p"
fi
wait "$p"
grep -q "$PASS" "$OUT.log"
