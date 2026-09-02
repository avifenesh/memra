#!/bin/bash
# q4e-loader-stream cell (memra issue #48): the whole acceptance set, unattended.
# Each arm takes /tmp/q48fn-measure.lock -s (shared) on its own, so an exclusive holder on
# this box is never contended; between arms the box is released.
set -uo pipefail
out=/root/realgate/loaderout
q="$out/CELL.log"
say() { printf "[%s] %s\n" "$(date -u +%FT%TZ)" "$*" >> "$q"; }
NEW=/root/realgate/bin/qwen4exp_real_gate.loader
OLD=/root/realgate/bin/qwen4exp_real_gate.prestream

# Headroom guard: never squeeze a co-tenant lane on this shared box.
need_mem() {
  local want=$1 i avail
  for i in $(seq 1 240); do
    avail=$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)
    if [ "$avail" -ge "$want" ]; then echo "$avail"; return 0; fi
    sleep 30
  done
  return 1
}

run_arm() {
  local lbl=$1 bin=$2 cap=$3 want=$4 avail
  if ! avail=$(need_mem "$want"); then
    say "$lbl SKIPPED - MemAvailable never reached ${want}GB in 2h (co-tenant headroom guard)"
    return 0
  fi
  say "$lbl starting (MemAvailable=${avail}GB, cap=$cap, binary=$(basename "$bin"))"
  /root/realgate/loaderout/run-cap.sh "$lbl" "$bin" "$cap"
  say "$lbl rc=$?"
}

run_arm new-cap150 "$NEW" 150G 170
run_arm old-cap150 "$OLD" 150G 170
run_arm old-cap230 "$OLD" 230G 250

say "identity comparison"
/root/realgate/loaderout/compare-identity.sh new-cap150 >> "$q" 2>&1
say "identity rc=$?"
say "CELL DONE"
