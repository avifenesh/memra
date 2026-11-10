#!/usr/bin/env bash
# Rig, inside the caller's gpulock lease (untimed): how deep the RTX 5090 can serve Qwen3.8-27B
# (NVFP4+Q5_K, the frspec MTP draft, one session). For each context in turn, boot memra-server,
# send one greedy request of about CTX-4096 prompt tokens (the served cells' document, 64 max
# tokens) and record whether it served, the prompt tokens and the card's peak memory.
# usage: cap5090.sh BIN_DIR OUTDIR CTX...
set -u
BIN=$1; OUT=$2; shift 2
D=$HOME/.cache/memra-split-rung
M=$HOME/models/q38-gguf
mkdir -p "$OUT"
for ctx in "$@"; do
  o=$OUT/ctx$ctx; mkdir -p "$o"
  nvidia-smi --query-gpu=timestamp,memory.used,temperature.gpu --format=csv,noheader -l 1 > "$o/gpu.csv" 2>&1 &
  smi=$!
  env MEMRA_COMPAT=openai MEMRA_ADDR=127.0.0.1:8097 MEMRA_CTX=$ctx MEMRA_MAX_SESSIONS=1 MEMRA_KV_HOST_MB=0 \
    MEMRA_MODELS="q38=$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" MEMRA_MTP_DRAFT="$M/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf" \
    MEMRA_PREFIX_STABLE_BOUNDARY=1 "$BIN/memra-server" > "$o/server.log" 2>&1 &
  sp=$!
  if waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8097/readyz && exit 0 || exit 1' --fail-if "! kill -0 $sp" --every 5s --timeout 6m > "$o/boot.log" 2>&1; then
    BENCH_BASE=http://127.0.0.1:8097 /usr/bin/python3 - "$o/req.json" "$((ctx - 4096))" > "$o/req.log" 2>&1 <<'EOF'
import json, sys
sys.path.insert(0, __import__("os").path.expanduser("~/.cache/memra-split-rung/box"))
import pf_long_memra as pf
from eos_ab import stream
out, target = sys.argv[1], int(sys.argv[2])
row = stream([{"role": "user", "content": pf.doc(pf.lines_for(target, "fairness-bench-v1"), f"memra-split cap {target}")}], 64, {"temperature": 0.0})
open(out, "w").write(json.dumps(row) + "\n")
print(json.dumps(row)[:400])
EOF
    echo "ctx $ctx served rc=$?" >> "$OUT/summary"
  else
    echo "ctx $ctx boot failed: $(grep -m1 -iE 'error|refus|out of memory' "$o/server.log" | cut -c1-200)" >> "$OUT/summary"
  fi
  kill -TERM $sp 2>/dev/null; waitfor --gone $sp --timeout 2m > /dev/null 2>&1 || kill -KILL $sp
  kill $smi 2>/dev/null
  echo "ctx $ctx peak MiB $(awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0 > m) m = $2+0} END {print m}' "$o/gpu.csv")" >> "$OUT/summary"
done
