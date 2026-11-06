#!/usr/bin/env bash
# Runs ON a GB10 host: the served P V V P campaign (round-2 base vs F8F4 v2), all three lengths
# per fresh boot, then the summary. usage: served_ab.sh CAMPAIGN_NAME V_BINDIR
set -u
D=$HOME/.cache/memra-gemm-v2
CD=$D/runs/$1; V=$2
P=$D/bins/r2base-1ed40895ec
bash "$D/campaign.sh" "$CD" "32768 131072 260000" "1 1 1" "P|$P|" "V|$V|" "V|$V|" "P|$P|"
python3 "$D/summarize.py" "$CD" > "$CD/summary.txt" 2>&1
python3 - "$CD" >> "$CD/summary.txt" <<'PY'
import json, pathlib, sys
cd = pathlib.Path(sys.argv[1])
for cell in sorted(p for p in cd.iterdir() if p.is_dir()):
    m = [json.loads(l)["mem_available_kb"] for l in open(cell / "mem.jsonl") if l.strip()]
    print(f"{cell.name}: MemAvailable floor {min(m) / 1048576:.2f} GiB")
PY
echo SERVED-AB-DONE
