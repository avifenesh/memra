#!/usr/bin/env python3
"""Port research/spill-b-20260919/run-day26-cell.sh to the <box> Max-Q kit (kit/DEVIATIONS.md D4, D5, D7).

Exact-match replacements; each must hit once. Idempotent: a file already carrying the port marker is left alone.
"""
import sys

path = sys.argv[1]
src = open(path).read()
MARK = "# Max-Q kit port"
if MARK in src:
    print("already ported")
    sys.exit(0)

subs = [
    (
        "set -uo pipefail\ncell=${1:?cell}; order=${2:?AB|BA}\n",
        "set -uo pipefail\n"
        "# Max-Q kit port (kit/DEVIATIONS.md D4, D5, D7): no git on the box (source.txt is the kit's SOURCE_SHA and\n"
        "# dirty.txt is the kit manifest check, empty when every kit file matches), every nvidia-smi call and the server\n"
        "# pinned to one card (SMI_CARD; CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=SMI_CARD), and the server pid\n"
        "# plus the card's compute apps recorded at ready (the pin check).\n"
        "SMI_CARD=${SMI_CARD:?SMI_CARD (the nvidia-smi index of the pinned card)}\n"
        "cell=${1:?cell}; order=${2:?AB|BA}\n",
    ),
    (
        'git rev-parse HEAD > "$C/source.txt"; git status --short > "$C/dirty.txt"',
        'cat "$WT/SOURCE_SHA" > "$C/source.txt"; sha256sum -c --quiet KIT-MANIFEST.sha256 > "$C/dirty.txt" 2>&1',
    ),
    (
        'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$C/compute-apps-before.csv"\n'
        'nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv > "$C/gpu-before.csv"',
        'nvidia-smi -i "$SMI_CARD" --query-compute-apps=pid,process_name,used_memory --format=csv > "$C/compute-apps-before.csv"\n'
        'nvidia-smi -i "$SMI_CARD" --query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv > "$C/gpu-before.csv"',
    ),
    (
        'MEMRA_COMPAT=openai MEMRA_MODELS="$MODEL_KEY=$MODEL" MEMRA_ADDR=127.0.0.1:$PORT MEMRA_ADMIT_PREDICT_SHADOW=1 \\\n',
        'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$SMI_CARD \\\n'
        'MEMRA_COMPAT=openai MEMRA_MODELS="$MODEL_KEY=$MODEL" MEMRA_ADDR=127.0.0.1:$PORT MEMRA_ADMIT_PREDICT_SHADOW=1 \\\n',
    ),
    (
        'echo "$ready_ms" > "$C/ready_ms.txt"\n',
        'echo "$ready_ms" > "$C/ready_ms.txt"\n'
        'own_server_pid > "$C/server.pid"\n'
        'nvidia-smi -i "$SMI_CARD" --query-compute-apps=pid,process_name,used_memory --format=csv > "$C/compute-apps-ready.csv"\n',
    ),
    (
        'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$C/compute-apps-after.csv"\n'
        'nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv > "$C/gpu-after.csv"',
        'nvidia-smi -i "$SMI_CARD" --query-compute-apps=pid,process_name,used_memory --format=csv > "$C/compute-apps-after.csv"\n'
        'nvidia-smi -i "$SMI_CARD" --query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv > "$C/gpu-after.csv"',
    ),
]
for old, new in subs:
    n = src.count(old)
    if n != 1:
        sys.exit(f"expected one match, found {n}: {old[:80]!r}")
    src = src.replace(old, new)
if "nvidia-smi --" in src:
    sys.exit("an nvidia-smi call without -i is left")
open(path, "w").write(src)
print("ported")
