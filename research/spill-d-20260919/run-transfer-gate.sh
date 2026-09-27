#!/usr/bin/env bash
# Day 12: run tier-transfer-gate (A's native conformance bindings, including the two day-11
# rule lines) through the collector on the single-PRO clone, one cell per case, N=1. Lock
# contention is retried on a bounded cadence and every refused attempt is kept; a cell whose
# gate was launched (command.log exists) is never retried, whatever its exit code.
# usage: run-transfer-gate.sh <receipt-root> [binary] [case ...]
set -uo pipefail
root=${1:?new receipt root required}
binary=${2:-/root/wt-d/target/release/tier-transfer-gate}
shift 2 2>/dev/null || shift $#
cases=("$@")
if [[ ${#cases[@]} -eq 0 ]]; then
  cases=(conformance roundtrip)
fi
mkdir -p "$root"
git -C "$(dirname "$(dirname "$binary")")/.." rev-parse HEAD > "$root/source.commit" 2>/dev/null || true
sha256sum "$binary" > "$root/binary.sha256"
for case in "${cases[@]}"; do
  dir="$root/$case"
  mkdir -p "$dir"
  cp "$root/source.commit" "$dir/source.commit" 2>/dev/null || true
  cp "$root/binary.sha256" "$dir/binary.sha256"
  deadline=$((SECONDS + 2700))
  attempt=0
  while :; do
    cell="$dir/collector-$attempt"
    set +e
    python3 tools/tier-battery.py --rig pro-single --timeout 700 --out "$cell" --execute \
      "$binary" "$case" > "$dir/collector-$attempt.log" 2>&1
    code=$?
    set -e
    if [[ -f "$cell/command.log" ]]; then
      echo "$code" > "$dir/collector.exit"
      echo "$attempt" > "$dir/launched-attempt"
      tail -n 1 "$cell/command.log" > "$dir/last-line.txt"
      break
    fi
    set +e
    grep -Eq 'REFUSED: \[Errno (11|35)\]' "$dir/collector-$attempt.log"
    contended=$?
    set -e
    if [[ $contended -ne 0 ]]; then
      echo "$code" > "$dir/collector.exit"
      echo "REFUSED: collector did not launch the gate and the refusal was not lock contention" > "$dir/last-line.txt"
      break
    fi
    if (( SECONDS >= deadline )); then
      echo 2 > "$dir/collector.exit"
      echo 'REFUSED: canonical GPU lock occupied for 45 minutes' > "$dir/last-line.txt"
      break
    fi
    sleep 30
    attempt=$((attempt + 1))
  done
  set +e
  python3 tools/tier-battery.py --validate "$dir" > "$dir/validate.json" 2> "$dir/validate.err"
  echo $? > "$dir/validate.exit"
  set -e
  printf '%s exit=%s last=%s\n' "$case" "$(cat "$dir/collector.exit")" "$(cat "$dir/last-line.txt")"
done
set +e
python3 tools/tier-battery.py --validate "$root" > "$root/validate.json" 2> "$root/validate.err"
echo $? > "$root/validate.exit"
exit 0
