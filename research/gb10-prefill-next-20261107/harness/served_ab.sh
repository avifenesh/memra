#!/usr/bin/env bash
# Runs ON a GB10 host: the served main-vs-lane campaign, all three lengths per fresh boot, then the
# summary with the MemAvailable floor per cell. usage: served_ab.sh CAMPAIGN_NAME BASE_BIN LANE_BIN CELLS
#   CELLS: the arm order, e.g. "B L L B B L"
set -u
D=$HOME/.cache/memra-gb10-next
CD=$D/runs/$1; B=$2; LB=$3; ORDER=$4
specs=()
for a in $ORDER; do
  case $a in
    B) specs+=("B|$B|") ;;
    L) specs+=("L|$LB|") ;;
  esac
done
bash "$D/box/campaign.sh" "$CD" "32768 131072 260000" "1 1 1" "${specs[@]}"
python3 "$D/box/summarize.py" "$CD" > "$CD/summary.txt" 2>&1
python3 - "$CD" >> "$CD/summary.txt" <<'PY'
import json, pathlib, sys
cd = pathlib.Path(sys.argv[1])
for cell in sorted(p for p in cd.iterdir() if p.is_dir()):
    m = [json.loads(l)["mem_available_kb"] for l in open(cell / "mem.jsonl") if l.strip()]
    print(f"{cell.name}: MemAvailable floor {min(m) / 1048576:.2f} GiB")
PY
echo SERVED-AB-DONE | tee -a "$CD/summary.txt"
