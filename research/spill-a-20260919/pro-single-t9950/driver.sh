#!/usr/bin/env bash
# OWED item 3's 9950X-class reading, driver (DAY44.md section 1): waits for the build receipt (rc=3: the host is not
# 9950X-class, nothing runs, the reading stays owed), then fill-survey.sh (the probe; section 1's rule picks the design
# for this host class), item3.sh (design T's cell, (a) and (b)), gates.sh and hitgate.sh ((c) on ft's binary, with the
# arms' tree's tools), each under ONE collector hold. A busy collector lock: bounded retries (60 x 120 s), never a signal to the holder. Every
# cell executed-not-qualified. No host, id or price here.
set -uo pipefail
cd /root/wt-a || exit 1
R=/root/spill-receipts/a-t9950
D=research/spill-a-20260919/pro-single-t9950
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
until grep -q '^rc=' "$R/build.log" 2>/dev/null; do sleep 20; done
if grep '^rc=' "$R/build.log" | tail -1 | grep -q '^rc=3'; then echo "$(date -u +%FT%TZ) host not 9950X-class; nothing run" | tee -a "$R/progress.log"; exit 3; fi
if ! grep '^rc=' "$R/build.log" | tail -1 | grep -q '^rc=0$'; then echo "build failed" | tee "$R/BUILD-FAILED"; exit 1; fi
cell() { # name timeout script args...
  local name=$1 timeout=$2; shift 2
  for try in $(seq 1 60); do
    python3 tools/tier-battery.py --rig pro-single --timeout "$timeout" --out "$R/$name" --external-lock --execute bash "$@" > "$R/$name.collector.log" 2>&1
    rc=$?
    if grep -q "^REFUSED: \[Errno 11\]\|canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$R/$name.collector.log" && [ ! -f "$R/$name/CELL.jsonl" ]; then
      echo "$(date -u +%FT%TZ) $name lock busy, retry $try/60 in 120 s" | tee -a "$R/lock-retries.log"; rm -rf "${R:?}/${name:?}"; sleep 120; continue
    fi
    echo "$(date -u +%FT%TZ) $name rc=$rc" | tee -a "$R/progress.log"; break
  done
}
cell fill-survey 1800 $D/fill-survey.sh @COLLECTOR_LOCK_FD@
cell item3-cell 14400 $D/item3.sh @COLLECTOR_LOCK_FD@
# The gates must be the arms' tree's (the tip's tools carry S2's fault cells, which a pre-S2 binary does not know): the
# tools/ of <g4_sha> exactly for the two gate cells, then the tip's back (the tree checked).
ARMS=$(cat "$R/tree-arms.sha"); TIP=$(cat "$R/tree-tip.sha")
git diff --binary "$TIP" "$ARMS" -- tools | git apply || { echo "$(date -u +%FT%TZ) the arms' tools did not apply; gates not run" | tee -a "$R/progress.log"; exit 2; }
git diff --stat -- tools > "$R/gates-tools.stat"
cell gates 5400 $D/gates.sh @COLLECTOR_LOCK_FD@
bash $D/hitgate.sh >> "$R/progress.log" 2>&1
git checkout -q "$TIP" -- tools; git clean -q -fd tools
cell unit-cell 5400 $D/unit-cells.sh @COLLECTOR_LOCK_FD@
[ -z "$(git status --porcelain --untracked-files=no)" ] || echo "$(date -u +%FT%TZ) the tree did not return to the tip after the gates" | tee -a "$R/progress.log"
echo "$(date -u +%FT%TZ) driver-done" | tee -a "$R/progress.log"
