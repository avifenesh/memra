#!/usr/bin/env bash
# integ71: lane B's GPU cells (B's list for integ71, commands as B gave them), run by drive71.sh under its hold of
# /tmp/memra-gpu.lock (FD 9); no cell takes the lock. Each cell: its own dir, log, exit and the pass line B named.
# usage: lane-b-cells.sh <out_root>
set -uo pipefail
R=$1; mkdir -p "$R"
HERE=$(cd "$(dirname "$0")/../../../.." && pwd); cd "$HERE" || exit 1
M9=/root/models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
M27=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
Q35=/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
SRV=target/release/memra-server
B=target/release
cell() { # $1 name $2 timeout $3.. command (run through bash -c so env prefixes and loops work)
    local name=$1 t=$2; shift 2
    mkdir -p "$R/$name"
    echo "$(date -u +%FT%TZ) start $name" | tee -a "$R/cells.log"
    timeout "$t" bash -c "$*" > "$R/$name/cell.log" 2>&1; local rc=$?
    echo "$rc" > "$R/$name/cell.exit"
    echo "$(date -u +%FT%TZ) done $name rc=$rc" | tee -a "$R/cells.log"
}
cell c01-grid-capture 1800 "MEMRA_TEST_QWEN_GGUF=$M9 NVIDIA_TF32_OVERRIDE=0 cargo test -p memra-engine --test grid_capture_gpu -- --ignored --test-threads=1"
cat docs/FLAGS.md docs/TESTING.md > "$R/cont.txt"
cell c02-continuation 600 "NVIDIA_TF32_OVERRIDE=0 $B/qwen-a4-continuation-gate $M9 $R/cont.txt 9296 16 48 80"
tail -c 4000 docs/SERVING.md > "$R/suffix.txt"
cell c03-rewind 4800 "for L in 6144 30720; do for K in 32 256; do timeout 1200 $B/concat-prime-probe $M27 primepath --prompt-a @docs/SERVING.md --prompt-tokens \$L --suffix @$R/suffix.txt --suffix-tokens 64 --hist \$K --rewind --steps 48; echo \"run L=\$L K=\$K rc=\$?\"; done; done"
cell c04-prime-gate 900 "$B/prime-gate $Q35 --prompts-file research/prime-gate-coverage-20260802/prompts-mixed.txt --steps 0"
# Cell 5 as B substituted it: the 35B draft (draft-35b-owntrim-nvfp4head-q4blk.gguf) is on no rig, box or tiyuvta repo, so
# run-spec K=1..8 runs on the 27B's own NextN head (B: the same Qwen hybrid MTP walker as the 9B), and on the 9B too.
cell c05-run-spec-27b 900 "env -u MEMRA_MTP_DRAFT -u MEMRA_PROMPT_DIR -u MEMRA_SPEC_K -u MEMRA_GEN_ONLY -u MEMRA_RESUME_EXACT -u MEMRA_SPEC_BUDGET_CLAMP MEMRA_SPEC_TEMP=0 MEMRA_NGEN=32 MEMRA_PROMPT_FILE=tools/fast-gate/prompts/probe.txt $B/run-spec $M27"
cell c05-run-spec-9b 900 "env -u MEMRA_MTP_DRAFT -u MEMRA_PROMPT_DIR -u MEMRA_SPEC_K -u MEMRA_GEN_ONLY -u MEMRA_RESUME_EXACT -u MEMRA_SPEC_BUDGET_CLAMP MEMRA_SPEC_TEMP=0 MEMRA_NGEN=32 MEMRA_PROMPT_FILE=tools/fast-gate/prompts/probe.txt $B/run-spec $M9"
cell c06-run-gen 600 "MEMRA_NGEN=8 $B/run-gen $M27 --prompt \"\$(head -c 4000 docs/SERVING.md)\""
cell c07-hfg 3600 "HFG_ARMS=g,h,j HFG_OUT=$R/c07-hfg/hfg HFG_READY_WAIT_S=900 tools/health-fault-gate.sh $M27"
cell c08-hfg-vmm 1800 "MEMRA_KV_ALLOCATOR=vmm HFG_ARMS=j HFG_OUT=$R/c08-hfg-vmm/hfg HFG_READY_WAIT_S=900 tools/health-fault-gate.sh $M27"
cell c09-grow 7200 "NVIDIA_TF32_OVERRIDE=0 $B/kv-tier-gate --artifact $M27 --case grow --context 32768 --tiers host --same-program --kv-allocator vmm-ondemand --out $R/c09-grow/grow"
cell c11-day44 5400 "EXTERNAL_LOCK=1 WT=$HERE RIG_LOCK=/tmp/memra-gpu.lock BIN=$SRV PREV_BIN=$SRV MODEL=$M27 MODEL_KEY=q38 BOOT_CTX= NO_SCOPE=1 YIELD_S=5 bash research/spill-b-20260919/day44-run.sh $R/c11-day44/d44 rx-plain-rx-O1-keep:keep:plain:RX6 rx-plain-rx-O1-exact:exact:plain:RX6 rx-spec-rx-O1-keep:keep:spec:RX6 rx-spec-rx-O1-exact:exact:spec:RX6 fault-plain-rxg6:fault:plain:RXg6; python3 research/spill-b-20260919/day44-read.py pro6000 $R/c11-day44/d44"
cell c12-amb-wrel 1800 "MEMRA_ADMIT_W_RELEASE=1 tools/admit-mem-burst-gate.sh $M9 $SRV $R/c12-amb-wrel/ev"
# The pass lines, read from each cell's own log (B's list).
{
  echo "c01 $(grep -hE '^test result' $R/c01-grid-capture/cell.log | tail -1) skipped_lines=$(grep -c 'MEMRA_TEST_QWEN_GGUF unset; skipped' $R/c01-grid-capture/cell.log)"
  echo "c02 $(grep -h 'CONTINUATION GATE' $R/c02-continuation/cell.log | tail -1)"
  echo "c03 exact=$(grep -c 'verdict rewind: EXACT' $R/c03-rewind/cell.log) of 4"
  echo "c04 exit=$(cat $R/c04-prime-gate/cell.exit)"
  echo "c05-27b exit=$(cat $R/c05-run-spec-27b/cell.exit) k_lines=$(grep -cE '^\[generate_spec K=[1-8]\]' $R/c05-run-spec-27b/cell.log) pass_lines=$(grep -c 'self-consistency: PASS' $R/c05-run-spec-27b/cell.log) acc0=$(grep -c 'WARNING: acceptance == 0' $R/c05-run-spec-27b/cell.log) $(grep -h '=== SELF-CONSISTENCY' $R/c05-run-spec-27b/cell.log | tail -1)"
  echo "c05-9b exit=$(cat $R/c05-run-spec-9b/cell.exit) k_lines=$(grep -cE '^\[generate_spec K=[1-8]\]' $R/c05-run-spec-9b/cell.log) pass_lines=$(grep -c 'self-consistency: PASS' $R/c05-run-spec-9b/cell.log) acc0=$(grep -c 'WARNING: acceptance == 0' $R/c05-run-spec-9b/cell.log) $(grep -h '=== SELF-CONSISTENCY' $R/c05-run-spec-9b/cell.log | tail -1)"
  echo "c06 match=$(grep -cw 'MATCH' $R/c06-run-gen/cell.log) mismatch=$(grep -c 'MISMATCH' $R/c06-run-gen/cell.log)"
  echo "c07 $(grep -h '^health-fault-gate:' $R/c07-hfg/cell.log | tail -1) $(grep -ho 'HFG (j)[^|]*chunk_sessions=[0-9]*' $R/c07-hfg/cell.log | grep -o 'chunk_sessions=[0-9]*' | tr '\n' ' ')"
  echo "c08 $(grep -h '^health-fault-gate:' $R/c08-hfg-vmm/cell.log | tail -1) reap_j=$(grep -c '\[kv-vmm\] reap (batch-oom)' $R/c08-hfg-vmm/hfg/j/server.log 2>/dev/null) reap_jred=$(grep -c '\[kv-vmm\] reap (batch-oom)' $R/c08-hfg-vmm/hfg/j-red/server.log 2>/dev/null) door_on=$(cat $R/c08-hfg-vmm/hfg/j/server.log $R/c08-hfg-vmm/hfg/j-red/server.log 2>/dev/null | grep -c '\[kv-vmm\] door=ON')"
  echo "c09 $(head -1 $R/c09-grow/grow/GROW.txt 2>/dev/null)"
  echo "c11 $(grep -hE '^DAY44 (E1|E3|E4-FAULT|E5)' $R/c11-day44/cell.log | sed 's/  */ /g' | cut -c1-200 | tr '\n' '|')"
  echo "c12 $(grep -h 'ADMIT-MEM BURST GATE' $R/c12-amb-wrel/cell.log | tail -1)"
} > "$R/result.txt"
echo "$(date -u +%FT%TZ) LANE-B-CELLS-DONE" | tee -a "$R/cells.log"
