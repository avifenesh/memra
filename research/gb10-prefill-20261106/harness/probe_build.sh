#!/usr/bin/env bash
# Runs ON a GB10 host: build the standalone v1/v2 F8F4 probe from probe-src/ (a copy of the
# engine .cu files and the probe, so a cargo build of the tree is never disturbed).
set -u
D=$HOME/.cache/memra-gemm-v2; S=$D/probe-src
export PATH=/usr/local/cuda/bin:$PATH
cd "$S" || exit 1
nice -n 10 nvcc -gencode arch=compute_121a,code=sm_121a -O3 -std=c++17 --expt-relaxed-constexpr \
  -Xptxas -v -I cu -o "$D/f8f4v2_probe" f8f4v2_probe.cu cu/mmq_nvfp4_f8f4.cu > "$D/logs/probe-build.log" 2>&1
echo "nvcc rc=$?"
grep -E "error" "$D/logs/probe-build.log" | head -n 20
python3 - "$D/logs/probe-build.log" <<'PY'
import re, sys
cur = None
for line in open(sys.argv[1]):
    m = re.search(r"Compiling entry function '(\S+)'", line)
    if m: cur = m.group(1); continue
    if cur and "f8f4v2" in cur and ("Used" in line or "spill" in line):
        print(cur[:60], "|", line.strip().split("ptxas info    :")[-1].strip())
PY
