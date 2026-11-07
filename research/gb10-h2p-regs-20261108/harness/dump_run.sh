#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: boot the research binary (the ATTN3 dump hook) and send one
# cold 131k prompt; the hook writes the prime-attention inputs of the chosen calls under dumps/.
# usage: dump_run.sh BIN
#   Dumps the first and the last attention layer (call 0 and 15 at each depth) of the chunks at
#   depths 16,384 / 32,768 / 65,536 and of the final chunk (131,051).
set -u
BIN=$1
D=$HOME/.cache/memra-attn3
OUT=$D/runs/dump${TAG:-}
mkdir -p "$D/dumps" "$OUT"
SPEC=${SPEC:-"16384:0,16384:15,32768:0,32768:15,65536:0,65536:15,131051:0,131051:15"}
bash "$D/box/run_arm.sh" "$OUT" "$BIN" "ATTN3_DUMP_DIR=$D/dumps ATTN3_DUMP=$SPEC" "131072" "1" > "$OUT/arm.log" 2>&1
echo "dump rc=$?"
grep ATTN3_DUMP "$OUT/server.log"
ls "$D/dumps"
