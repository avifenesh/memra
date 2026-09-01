#!/usr/bin/env bash
# DECODE-DIET build + attribution (rebuild-after-checkout-attribution law):
# a 0.04s "Finished" after a checkout is a failed-checkout alarm; the receipt carries
# real build wall, binary mtime == BUILD_END, git log -1, and the strings probes for
# the four door announces + the batched verify walk.
set -uo pipefail
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
SRC=/root/memra-diet
BIN=$SRC/target/release/memra-server
LOG=/root/out-diet/build-28cbc1af6.log
mkdir -p /root/out-diet

{
  echo "== checkout =="
  cd "$SRC"
  git fetch origin lane/glm5-decode-diet
  git checkout -q 28cbc1af6
  git log -1 --format='%H %s'
  echo "== build (nice -n19, window owned — no other timed cell in flight) =="
  echo "BUILD_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  t0=$SECONDS
  nice -n19 cargo build --release --bin memra-server --bin launch-econ 2>&1 | tail -20
  rc=$?
  echo "BUILD_RC=$rc BUILD_WALL_S=$((SECONDS-t0))"
  echo "BUILD_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "== attribution =="
  ls -la "$BIN" "$SRC/target/release/launch-econ"
  echo "bin_mtime=$(stat -c %y "$BIN")"
  sha256sum "$BIN" | cut -c1-16
  echo "== strings probes (four door announces + batched walk) =="
  for p in "hc-fused-pre] engaged" "hc-decode-ws] engaged" "kda-fused6] engaged arm=bf16" \
           "mla-decode-split] engaged" "verify walk BATCHED per layer"; do
    n=$(strings "$BIN" | grep -c -- "$p")
    echo "probe '$p' hits=$n"
    [ "$n" -ge 1 ] || echo "PROBE_FAIL: $p"
  done
} 2>&1 | tee "$LOG"
