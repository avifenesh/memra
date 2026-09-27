#!/usr/bin/env bash
# Day 11: run every kv-tier-gate fault arm through the collector on the single-PRO clone.
# One cell per arm, N=1, 8k context, pooled allocator (the default program), 1800 s timeout.
# Lock contention is retried on a bounded cadence and every refused attempt is kept; a cell
# whose gate was launched (command.log exists) is never retried, whatever its exit code.
# usage: run-fault-arms.sh <receipt-root> [artifact] [binary] [arm ...]
set -uo pipefail
root=${1:?new receipt root required}
artifact=${2:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
binary=${3:-/root/wt-d/target/release/kv-tier-gate}
shift 3 2>/dev/null || shift $#
arms=("$@")
if [[ ${#arms[@]} -eq 0 ]]; then
  arms=(cancel-demote cancel-restore corrupt-host missing-host host-budget-short device-short require-resident)
fi
mkdir -p "$root"
git -C "$(dirname "$(dirname "$binary")")/.." rev-parse HEAD > "$root/source.commit" 2>/dev/null || true
sha256sum "$binary" > "$root/binary.sha256"
worst=0
for arm in "${arms[@]}"; do
  dir="$root/$arm"
  mkdir -p "$dir"
  cp "$root/source.commit" "$dir/source.commit" 2>/dev/null || true
  cp "$root/binary.sha256" "$dir/binary.sha256"
  deadline=$((SECONDS + 2700))
  attempt=0
  while :; do
    cell="$dir/collector-$attempt"
    set +e
    python3 tools/tier-battery.py --rig pro-single --timeout 1800 --out "$cell" --execute \
      env NVIDIA_TF32_OVERRIDE=0 "$binary" --artifact "$artifact" --case active --context 8192 \
      --tiers host --same-program --kv-allocator pooled --fault "$arm" --out "$dir/receipt" \
      > "$dir/collector-$attempt.log" 2>&1
    code=$?
    set -e
    if [[ -f "$cell/command.log" ]]; then
      # The gate ran: PASS (0), typed refusal (2) or failure. Record; never rerun a launched cell.
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
  printf '%s exit=%s last=%s\n' "$arm" "$(cat "$dir/collector.exit")" "$(cat "$dir/last-line.txt")"
  (( $(cat "$dir/collector.exit") > worst )) && worst=$(cat "$dir/collector.exit")
done
set +e
python3 tools/tier-battery.py --validate "$root" > "$root/validate.json" 2> "$root/validate.err"
echo $? > "$root/validate.exit"
exit 0
