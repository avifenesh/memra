#!/usr/bin/env bash
# Launch only on the isolated single-PRO lane clone. No GPU call outside collector.
set -euo pipefail
out=${1:?new receipt directory required}
mkdir -p "$out"
deadline=$((SECONDS + 2700))
attempt=0
while :; do
  cell="$out/attempt-$attempt"
  set +e
  python3 tools/tier-battery.py --rig pro-single --timeout 120 --out "$cell" --execute \
    python3 -c 'import subprocess
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"],text=True)
print("COMPUTE INVENTORY: "+repr(apps),flush=True)
if apps.strip(): raise SystemExit("REFUSED: competing GPU process")
for arm in ("0","1"):
 print("SMOKE HOST_BOUNCE="+arm,flush=True)
 subprocess.run(["env","MEMRA_PP_HOST_BOUNCE="+arm,"target/release/pp-transport-smoke"],check=True)
' > "$out/attempt-$attempt.console.log" 2>&1
  code=$?
  set -e
  if [[ $code -eq 0 ]]; then
    printf '%s\n' "$cell" > "$out/completed-cell.txt"
    exit 0
  fi
  # Retry lock contention only, never retry a launched/refused GPU command.
  if [[ -f "$cell/command.log" ]] || ! grep -Eq 'REFUSED: \[Errno (11|35)\]' "$out/attempt-$attempt.console.log"; then
    exit "$code"
  fi
  if (( SECONDS >= deadline )); then
    echo 'REFUSED: canonical GPU lock occupied for 45 minutes' >&2
    exit 2
  fi
  sleep 30
  attempt=$((attempt + 1))
done
