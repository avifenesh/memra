#!/usr/bin/env bash
# Laptop: retry of quiet-5090.sh after its first A/B was INCONCLUSIVE (another project's Go tests held
# the rig at load 45-66, so the fixed drop rule kept 1 + 1 runs). Holds nothing while it waits:
# it starts only once both the 1-minute and the 5-minute load averages are below 20, then
# takes the build lock, then the 5090, and runs the same pre-registered A/B (same drop rule, same
# decision) and, on ACCEPT, the sm_120a record. usage: quiet-5090b.sh HEAD_SHA ATTEMPT
set -u
HEAD=$1; N=$2
G=~/.cache/memra-gemm-v3
SRC=$G/eng; WORK=$G/cells-5090
cd "$SRC" || exit 1
echo "$(TZ=Asia/Jerusalem date +%T) attempt $N: waiting for a quiet rig"
waitfor --until 'awk "{exit !(\$1 < 20 && \$2 < 20)}" /proc/loadavg' --every 60s --timeout 4h 2>&1 | tail -n 1
echo "$(TZ=Asia/Jerusalem date +%T) quiet; waiting for the build lock, then the 5090 ($(cut -d' ' -f1-3 /proc/loadavg))"
flock "$HOME/.cache/memra-rig-build.lock" \
  gpulock run --ttl 90m --for "memra #1102 quiet decode A/B + sm_120a record, attempt $N (gemm-v2 agent)" --wait 4h -- \
  bash -c '
    set -u
    echo "$(TZ=Asia/Jerusalem date +%T) lease taken ($(cut -d" " -f1-3 /proc/loadavg))"
    python3 '"$G"'/decode-ab.py '"$SRC"' '"$WORK"' '"$G"'/decode-ab-out'"$N"' > '"$G"'/decode-ab'"$N"'.log 2>&1
    echo "$(TZ=Asia/Jerusalem date +%T) decode A/B rc=$?"; tail -n 1 '"$G"'/decode-ab'"$N"'.log
    if grep -q "\"decision\": \"ACCEPT\"" '"$G"'/decode-ab'"$N"'.log; then
      python3 '"$G"'/cells-accept.py '"$SRC"' '"$HEAD"' 259ef0d31035 '"$WORK"' '"$G"'/cells-5090-out'"$N"' '"$G"'/decode-ab-out'"$N"'/decode-ab.json
      echo "$(TZ=Asia/Jerusalem date +%T) record rc=$?"
    else
      echo "decision is not ACCEPT: no record"
    fi'
echo "$(TZ=Asia/Jerusalem date +%T) QUIET-5090-DONE"
