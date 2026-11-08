#!/usr/bin/env bash
# Runs ON b23d under the caller's sparklock lease: served-fup.sh (ledger rows 43 and 45, cells
# B L N N L B), then the 16k argmax-margin gate on the follow-up binary with v3 (the arm the cb33
# lease ended before). usage: b23d-fup.sh LANDED_BIN FOLLOWUP_BIN TREE OUT
set -u
LB=$1; NB=$2; TREE=$3; OUT=$4
D=$HOME/.cache/memra-f8f4v3
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
export TMPDIR=$D/tmp PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
log(){ echo "$(date +%T) $*"; }
mkdir -p "$OUT"
bash "$D/served-fup.sh" "$LB" "$NB" "$OUT/served" > "$OUT/served.out" 2>&1
log "served done"; head -n 30 "$OUT/served/summary.txt"
R=$OUT/argmax16k-v3; rm -rf "$R"; mkdir -p "$R/root/target" "$R/root/research/e2e"
cp -a "$TREE/tools" "$R/root/tools"; cp -a "$TREE/research/e2e/prompts" "$R/root/research/e2e/prompts"
ln -s "$NB" "$R/root/target/release"
(cd "$R/root" && MEMRA_GATE_LOG_ROOT=$R bash tools/argmax-margin-gate.sh "$G8" --prompt research/e2e/prompts/p4-16k.txt --logdir "$R") > "$R/argmax.log" 2>&1
log "argmax-16k v3 rc=$? $(grep -E 'SUMMARY|PASS|FAIL|flip budget' "$R/argmax.log" | tr '\n' ' ')"
rm -rf "$R/root"
echo B23D-FUP-DONE
