#!/usr/bin/env bash
# flip-reprice attributed build (rebuild-attribution law: real build time + git log -1 +
# binary mtime == BUILD_END + strings probes; a 0.04s "Finished" after a checkout is a
# failed-checkout alarm). Own clone /root/memra-vb @ c62677352 (lane/glm5-verify-batch).
set -uo pipefail
cd /root/memra-vb
LOG=/root/out-flip3/build-c62677352.log
{
  echo "=== BUILD ATTRIBUTION (flip-reprice) ==="
  git log -1 --format="commit %H%nsubject %s%ndate %ci"
  git status --porcelain | head -5
  echo "BUILD_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  T0=$(date +%s)
  PATH=/root/.cargo/bin:$PATH cargo build --release 2>&1 | tail -5
  RC=$?
  T1=$(date +%s)
  echo "BUILD_EXIT=$RC BUILD_REAL_S=$((T1-T0))"
  echo "BUILD_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  BIN=/root/memra-vb/target/release/memra-server
  echo "bin_mtime=$(stat -c %y "$BIN" 2>/dev/null)"
  echo "--- strings probes (the batched-walk head must carry these) ---"
  for lit in "glm5-phase-v" "verify walk BATCHED per layer" "draft source = dflash2 @" "confidence gate armed: PMIN=" "MEMRA_GLM5_VERIFY_BATCH"; do
    n=$(strings -a "$BIN" | grep -cF "$lit")
    echo "probe '$lit': $n"
  done
} | tee "$LOG"
grep -q "BUILD_EXIT=0" "$LOG" || { echo "BUILD FAILED"; exit 1; }
n=$(strings -a /root/memra-vb/target/release/memra-server | grep -cF "glm5-phase-v")
[ "$n" -ge 1 ] || { echo "STRINGS PROBE FAILED: [glm5-phase-v] absent"; exit 1; }
echo "BUILD GREEN"
