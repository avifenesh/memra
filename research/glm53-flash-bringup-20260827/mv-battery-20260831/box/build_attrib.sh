#!/usr/bin/env bash
# mv-doors build + attribution (rebuild-after-checkout-attribution law):
# a 0.04s "Finished" after a checkout is a failed-checkout alarm; the receipt carries
# real build wall, binary mtime == BUILD_END, git log -1, and the strings probes for
# the five matvec door announces + the batched verify walk + the vrows pair announce.
set -uo pipefail
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
SRC=/root/memra-mv
BIN=$SRC/target/release/memra-server
LOG=/root/out-mv/build-146b13c33.log
mkdir -p /root/out-mv

{
  echo "== checkout =="
  cd "$SRC"
  git fetch origin lane/glm5-matvec
  git checkout -q 146b13c33
  git log -1 --format='%H %s'
  echo "== build (nice -n19; launched only with no timed cell in flight) =="
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
  echo "== strings probes (five door announces + batched walk + vrows pair) =="
  for p in "bf16-tcols-wide] engaged" "bf16-tcols-x1] engaged" "moe-vrows-pack] engaged" \
           "topk-shards] engaged" "glm5-verify-ws] engaged" \
           "verify walk BATCHED per layer" "verify MoE batched across rows"; do
    n=$(strings "$BIN" | grep -c -- "$p")
    echo "probe '$p' hits=$n"
    [ "$n" -ge 1 ] || echo "PROBE_FAIL: $p"
  done
} 2>&1 | tee "$LOG"
