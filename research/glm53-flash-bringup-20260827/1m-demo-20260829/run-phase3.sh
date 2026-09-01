#!/bin/bash
# Phase 3 of the 1M demonstration, on the chunked-ppN binary:
#   0. fixture gate: glm5_hyper_ppn_gate at stages=4 cross-device, monolithic path
#      (default env, P=6 = one chunk) AND chunk loop engaged (MEMRA_PRIME_CHUNK=2, 3 chunks).
#      Both must hold bit-identity; the second is the arm my change adds.
#   1. reboot PP4-1M, re-run the 6k identity rung: door-off chunked (phase1-A-6k, banked)
#      vs PP4 now-chunked must be BYTE-IDENTICAL — the phase-1 divergence was monolithic-vs-
#      chunked schedule mismatch (the documented cuBLASLt m-shape near-tie class), and this
#      re-run is the discriminator that closes it.
#   2. ladder: 16k, 131k, 262k (greedy), then the 1M twin pair (greedy + vendor-default).
set -u
R=$HOME/lane-1mdemo-vast-20260829
W=$HOME/wt-1mdemo
BIN=$W/target/release/memra-server
GATE=$W/target/release/glm5-hyper-ppn-gate
C=$R/corpus-1m.txt
{
  date -u +%FT%TZ
  echo "=== 0. stop any running server first: the fixture gates need the cards ==="
  for pid in $(pgrep -x memra-server 2>/dev/null); do
    exe=$(readlink /proc/$pid/exe 2>/dev/null) || continue
    case "$exe" in
      */memra-server|*/memra-server\ \(deleted\)) echo "  stopping pid $pid"; kill $pid ;;
      *) echo "  skip pid $pid exe=$exe" ;;
    esac
  done
  for i in $(seq 1 90); do pgrep -x memra-server >/dev/null || break; sleep 2; done
  pgrep -x memra-server >/dev/null && { echo "server still alive; abort"; exit 1; }
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  echo "=== 0a. fixture gate, stages=4 cross-device, MONOLITHIC ppN path (one chunk) ==="
  ( cd $W && MEMRA_PP_DEVICES=0,1,2,3 NVIDIA_TF32_OVERRIDE=0 "$GATE" 4 6 8 2>&1 | tail -12 )
  echo
  echo "=== 0b. fixture gate, stages=4 cross-device, CHUNK LOOP engaged (chunk=2 over P=6) ==="
  ( cd $W && MEMRA_PP_DEVICES=0,1,2,3 MEMRA_PRIME_CHUNK=2 NVIDIA_TF32_OVERRIDE=0 "$GATE" 4 6 8 2>&1 | tail -12 )
  echo
  echo "=== 1. BOOT PP4-1M (chunked-ppN binary) ==="
  bash $R/serve-1m.sh PP4-1M-C "$BIN" MEMRA_CTX=1048576 CUDA_VISIBLE_DEVICES=0,1,2,3 \
    MEMRA_PP_STAGES=4 MEMRA_PP_DEVICES=0,1,2,3 MEMRA_MOE_RESIDENT_HEADROOM_GB=36 || exit 1
  bash $R/vramwatch.sh $R/phase3-vram.csv 20 &
  VW=$!
  echo "vramwatch pid $VW"

  echo; echo "=== identity rung: 6k greedy, PP4-chunked vs banked door-off-chunked ==="
  python3 $R/primeprobe.py C-6k "$C" 26000 64 greedy $R/phase3-C-6k.json
  python3 - <<'EOF'
import json
a = json.load(open("/root/lane-1mdemo-vast-20260829/phase1-A-6k.json"))
b = json.load(open("/root/lane-1mdemo-vast-20260829/phase3-C-6k.json"))
print(f"6k rung: doorOFF-chunked vs PP4-chunked byte-identical = {a['output'] == b['output']}")
if a["output"] != b["output"]:
    print("  A:", repr(a["output"][:160]))
    print("  B:", repr(b["output"][:160]))
EOF

  echo; echo "=== R16K ==="
  date -u +%FT%TZ
  python3 $R/primeprobe.py R16K "$C" 64400 64 greedy $R/phase3-R16K.json
  echo; echo "=== R131K ==="
  date -u +%FT%TZ
  python3 $R/primeprobe.py R131K "$C" 527000 128 greedy $R/phase3-R131K.json
  date -u +%FT%TZ
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo; echo "=== R262K ==="
  date -u +%FT%TZ
  python3 $R/primeprobe.py R262K "$C" 1054000 128 greedy $R/phase3-R262K.json
  date -u +%FT%TZ
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  CHARS=$(python3 - <<'EOF'
import json
j = json.load(open("/root/lane-1mdemo-vast-20260829/phase3-R262K.json"))
pt = j["usage"]["prompt_tokens"]
ratio = j["chars"] / pt
print(int(1043000 * ratio))
EOF
)
  echo; echo "=== R1M greedy (chars=$CHARS from R262K ratio) ==="
  date -u +%FT%TZ
  python3 $R/primeprobe.py R1M "$C" "$CHARS" 256 greedy $R/phase3-R1M.json
  date -u +%FT%TZ
  if ! python3 -c "import json,sys; sys.exit(0 if json.load(open('$R/phase3-R1M.json'))['status']==200 and not json.load(open('$R/phase3-R1M.json'))['error'] else 1)"; then
    CHARS=$((CHARS * 985 / 1000))
    echo "R1M failed; ONE retry at chars=$CHARS"
    date -u +%FT%TZ
    python3 $R/primeprobe.py R1M-retry "$C" "$CHARS" 256 greedy $R/phase3-R1M.json
    date -u +%FT%TZ
  fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "--- serve log tail after R1M ---"
  grep -vE "meter|admission" $R/serve-PP4-1M-C.log | tail -12

  echo; echo "=== R1M vendor-default sampled twin ==="
  date -u +%FT%TZ
  python3 $R/primeprobe.py R1M-V "$C" "$CHARS" 256 vendor $R/phase3-R1M-V.json
  date -u +%FT%TZ
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  kill $VW 2>/dev/null && echo "vramwatch $VW stopped"
  echo "--- per-card VRAM peak over phase 3 ---"
  python3 - <<'EOF'
import csv
rows = [r for r in csv.reader(open("/root/lane-1mdemo-vast-20260829/phase3-vram.csv")) if r and r[0] != "ts"]
for g in range(4):
    print(f"  gpu{g}: peak {max(int(r[g+1]) for r in rows)} MiB")
EOF
  echo "--- OOM/panic/error census of the serve log ---"
  grep -cE "out.of.memory|panicked|CUDA_ERROR|engine-error" $R/serve-PP4-1M-C.log || true
  date -u +%FT%TZ
  echo PHASE3DONE
} 2>&1 | tee $R/07-phase3.txt
