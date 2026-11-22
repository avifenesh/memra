#!/bin/bash
# runs under a gpulock lease: both W8A8 programs on every case, then the comparison
set -u
D=/home/avifenesh/.cache/inkling2mini/actq-fix/w8/data
nvidia-smi --query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem --format=csv,noheader
for c in $D/L*; do
  m=$c/meta.json
  N=$(python3 -c "import json;print(json.load(open('$m'))['N'])"); K=$(python3 -c "import json;print(json.load(open('$m'))['K'])")
  R=$(python3 -c "import json;print(json.load(open('$m'))['R'])"); G=$(python3 -c "import json;print(json.load(open('$m'))['ng'])")
  /home/avifenesh/.cache/inkling2mini/actq-fix/w8/w8probe $c $N $K $R $G || echo "FAILED $c"
done
python3 /home/avifenesh/.cache/inkling2mini/actq-fix/w8/analyze.py $D
