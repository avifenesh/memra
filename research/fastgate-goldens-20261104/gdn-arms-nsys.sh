#!/bin/bash
# S22:F5: do MEMRA_GDN_MMA=1/0 run different GDN kernels on the q9 and q35 fast-gate probes?
cd /home/avifenesh/.cache/memra-fix/engine-dsv4-glm || exit 2
O=/home/avifenesh/.cache/memra-fix/logs-fastgate/gdn-nsys; mkdir -p $O
B=/home/avifenesh/.cache/memra-fix/target-cuda/release/run-gen
P=$PWD/tools/fast-gate/prompts/probe.txt
Q9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
for arm in 1 0; do
    /usr/local/cuda/bin/nsys profile -t cuda -o $O/q9-mma$arm -f true --sample=none --cpuctxsw=none /usr/bin/env MEMRA_GDN_MMA=$arm MEMRA_NGEN=20 MEMRA_NMEASURE=0 MEMRA_PROMPT_FILE=$P $B $Q9 > $O/q9-mma$arm.log 2>&1
    /usr/local/cuda/bin/nsys stats -q -r cuda_gpu_kern_sum -f csv $O/q9-mma$arm.nsys-rep > $O/q9-mma$arm.kern.csv 2>&1
    /usr/local/cuda/bin/nsys profile -t cuda -o $O/q35-mma$arm -f true --sample=none --cpuctxsw=none /usr/bin/env MEMRA_CHAT=1 MEMRA_GDN_MMA=$arm MEMRA_NGEN=20 MEMRA_NMEASURE=0 MEMRA_PROMPT_FILE=$P $B $Q35 > $O/q35-mma$arm.log 2>&1
    /usr/local/cuda/bin/nsys stats -q -r cuda_gpu_kern_sum -f csv $O/q35-mma$arm.nsys-rep > $O/q35-mma$arm.kern.csv 2>&1
done
for m in q9 q35; do
    for arm in 1 0; do grep -E '^tokens:' $O/$m-mma$arm.log | sed "s/^/$m mma$arm /"; done
    python3 - $O/$m-mma1.kern.csv $O/$m-mma0.kern.csv <<'EOF'
import csv, sys
def names(p):
    rows = list(csv.reader(open(p)))
    h = next((i for i, r in enumerate(rows) if 'Name' in r), None)
    if h is None: return set()
    k = rows[h].index('Name')
    return {r[k][:90] for r in rows[h + 1:] if len(r) > k}
a, b = names(sys.argv[1]), names(sys.argv[2])
print('only mma1:', sorted(n for n in a - b if 'gdn' in n.lower() or 'mma' in n.lower() or 'scan' in n.lower() or 'k4' in n.lower() or 'k5' in n.lower())[:12])
print('only mma0:', sorted(n for n in b - a)[:12])
print('kernel names', len(a), len(b))
EOF
done
