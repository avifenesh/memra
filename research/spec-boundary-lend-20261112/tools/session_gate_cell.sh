#!/usr/bin/env bash
# Runs ON a GB10 unit under the caller's lease (memra#1221 part 2 review round 2, item 3): build
# session-gate from the head tree (src-head with TAG's files over it) if bins/TAG lacks it, then run
# it with MEMRA_GATE_LEND_ABOVE=1 (every trunk layer's rows [0, committed) lent before each
# checkpoint rewind) on the 27B MTP GGUF. -> runs/session-gate-TAG/{gate.log,verdict.txt}
# usage: session_gate_cell.sh CAP_MIN TAG
set -u
R=$HOME/.cache/memra-1221b; CAP=${1:-25}; TAG=$2
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" CARGO_TARGET_DIR=$R/target-head
OUT=$R/runs/session-gate-$TAG; mkdir -p "$OUT" "$R/bins/$TAG"
end=$(( $(date +%s) + CAP * 60 ))
left() { echo $(( end - $(date +%s) )); }
cd "$R/src-head" || exit 1
[ -f ".files-$TAG" ] || { echo "src-head does not hold $TAG's files" >> "$OUT/verdict.txt"; exit 1; }
if [ ! -x "$R/bins/$TAG/session-gate" ]; then
  timeout -s INT "$(left)" nice -n 10 cargo build --release --locked -j 16 -p memra-engine --bin session-gate > "$OUT/build.log" 2>&1 \
    || { echo "$(date -Is) BUILD-STOP session-gate" >> "$OUT/verdict.txt"; exit 1; }
  cp -a "$CARGO_TARGET_DIR/release/session-gate" "$R/bins/$TAG/"
fi
M=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
{ date -Is; sha256sum "$R/bins/$TAG/session-gate" | sed "s|$HOME|~|"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } >> "$OUT/verdict.txt"
[ "$(left)" -gt 120 ] || { echo "cap reached before the gate" >> "$OUT/verdict.txt"; exit 1; }
MEMRA_ENV_AUDIT=warn MEMRA_GATE_LEND_ABOVE=1 timeout "$(( $(left) - 30 ))" "$R/bins/$TAG/session-gate" "$M" > "$OUT/gate.log" 2>&1
rc=$?
echo "rc=$rc $(grep -E 'SESSION GATE|ALL TURNS MATCH' "$OUT/gate.log" | tail -n 2 | tr '\n' ' ')" >> "$OUT/verdict.txt"
exit $rc
