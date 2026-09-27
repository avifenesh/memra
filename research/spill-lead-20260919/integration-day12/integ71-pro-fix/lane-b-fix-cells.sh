#!/usr/bin/env bash
# integ71 fix run: lane B's cells for its #844 fixes (B's list), run by drive71f.sh under its hold of
# /tmp/memra-gpu.lock (FD 9); no cell takes the lock. usage: lane-b-fix-cells.sh <out_root>
set -uo pipefail
R=$1; mkdir -p "$R"
HERE=$(cd "$(dirname "$0")/../../../.." && pwd); cd "$HERE" || exit 1
M9=/root/models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
M27=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
SRV=target/release/memra-server
cell() {
    local name=$1 t=$2; shift 2
    mkdir -p "$R/$name"
    echo "$(date -u +%FT%TZ) start $name" | tee -a "$R/cells.log"
    timeout "$t" bash -c "$*" > "$R/$name/cell.log" 2>&1; local rc=$?
    echo "$rc" > "$R/$name/cell.exit"
    echo "$(date -u +%FT%TZ) done $name rc=$rc" | tee -a "$R/cells.log"
}
D44="EXTERNAL_LOCK=1 WT=$HERE RIG_LOCK=/tmp/memra-gpu.lock BIN=$SRV PREV_BIN=$SRV MODEL=$M27 MODEL_KEY=q38 BOOT_CTX= NO_SCOPE=1 YIELD_S=5"
cell c11-day44 5400 "$D44 bash research/spill-b-20260919/day44-run.sh $R/c11-day44/d44 rx-plain-rx-O1-keep:keep:plain:RX6 rx-plain-rx-O1-exact:exact:plain:RX6 rx-spec-rx-O1-keep:keep:spec:RX6 rx-spec-rx-O1-exact:exact:spec:RX6 fault-plain-rxg6:fault:plain:RXg6; python3 research/spill-b-20260919/day44-read.py pro6000 $R/c11-day44/d44"
cell c13-rw 5400 "$D44 bash research/spill-b-20260919/day44-run.sh $R/c13-rw/d44rw rw-plain-O1-keep:keep:plain:RW6 rw-plain-O1-exact:exact:plain:RW6 rw-spec-O1-keep:keep:spec:RW6 rw-spec-O1-exact:exact:spec:RW6; python3 research/spill-b-20260919/day44-read.py pro6000 $R/c13-rw/d44rw"
cell c12x-amb-exact 1800 "MEMRA_RESUME_EXACT=1 tools/admit-mem-burst-gate.sh $M9 $SRV $R/c12x-amb-exact/ev"
{
  echo "c11 $(grep -hE '^DAY44 (E1|E3|E4-FAULT|E5)' $R/c11-day44/cell.log | sed 's/  */ /g' | cut -c1-200 | tr '\n' '|')"
  echo "c13 $(grep -hE '^DAY44 (R1|R2|R3)' $R/c13-rw/cell.log | sed 's/  */ /g' | cut -c1-220 | tr '\n' '|')"
  echo "c12x $(grep -h 'ADMIT-MEM BURST GATE' $R/c12x-amb-exact/cell.log | tail -1)"
} > "$R/result.txt"
echo "$(date -u +%FT%TZ) LANE-B-FIX-CELLS-DONE" | tee -a "$R/cells.log"
