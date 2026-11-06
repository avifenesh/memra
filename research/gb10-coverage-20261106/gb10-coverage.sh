#!/usr/bin/env bash
# Runs ON spark-b23d under the caller's sparklock lease ($LEASE), after the cold-drafter run.
# Two GB10 coverage checks at main's code (memra-server built from 3bbdf42ff9 = main 5d70118b99):
#   1. a streaming /v1/completions request with a ~150K-token prompt (prefill well past 90 s) gets
#      its headers and its tokens, never an HTTP 408 (gguf-mtpmask serving arm, ctx 262,144);
#   6. a clean build (fresh target dir) with MEMRA_CUDA_ARCH unset auto-detects 121a: the
#      memra-engine build.rs warning line, then memra-probe built the same way and run.
set -u
LEASE=${LEASE:?lease}
D=$HOME/memra-run
O=$D/gb10-coverage
mkdir -p "$O"
L=$O/chain.log
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) [coverage] $*" | tee -a "$L"; }
export SPARKLOCK_LEASE=$LEASE
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
T=$HOME/.cache/memra-cells/gb10/tree-121a
log "start lease=$LEASE"

# ---- check 1: long streaming prefill ----
if ss -ltn | grep -q ':8090 '; then log "port 8090 busy; refusing"; exit 4; fi
(ARM=gguf-mtpmask BIN=$D/bin-main-3bbdf42ff9 LEASE=$LEASE nohup bash $D/serve-batchspec.sh) > $D/serve-coverage.out 2>&1 < /dev/null &
ok=0
for i in $(seq 1 80); do curl -sf http://127.0.0.1:8090/v1/models >/dev/null && { ok=1; break; }; sleep 15; done
[ $ok = 1 ] || { log "boot FAILED"; exit 3; }
log "server up ($(sha256sum $D/bin-main-3bbdf42ff9/memra-server | cut -c1-12))"
timeout 2400 python3 $D/longprefill_client.py http://127.0.0.1:8090 q38 150000 "$O/check1-result.json" > "$O/check1-client.out" 2>&1
log "check1: $(cat "$O/check1-client.out" | tail -n 1)"
pid=$(cat $D/serve-bs-gguf-mtpmask.pid)
kill "$pid"
for i in $(seq 1 90); do [ -e /proc/$pid ] || break; sleep 2; done
cp $D/serve-bs-gguf-mtpmask.log "$O/check1-server.log"
cp $D/guard-bs-gguf-mtpmask.log "$O/check1-guard.log" 2>/dev/null || true
log "check1 server stopped"

# ---- check 6: clean build, MEMRA_CUDA_ARCH unset ----
CT=~/.cache/memra-gb10-clean-target
rm -rf ~/.cache/memra-gb10-clean-target
cd "$T" || exit 2
log "check6: clean build at $(git rev-parse --short=10 HEAD), MEMRA_CUDA_ARCH=${MEMRA_CUDA_ARCH:-unset}"
env -u MEMRA_CUDA_ARCH CARGO_TARGET_DIR="$CT" CARGO_BUILD_JOBS=8 \
  flock "$HOME/.cache/memra-rig-build.lock" nice -n 15 cargo build --release --locked -p memra-engine --lib \
  > "$O/check6-engine-build.log" 2>&1
log "check6 engine build rc=$? : $(grep -a 'auto-detected' "$O/check6-engine-build.log" | tr '\n' ' ')"
env -u MEMRA_CUDA_ARCH CARGO_TARGET_DIR="$CT" CARGO_BUILD_JOBS=8 \
  flock "$HOME/.cache/memra-rig-build.lock" nice -n 15 cargo build --release --locked -p memra-probe --bin memra-probe \
  > "$O/check6-probe-build.log" 2>&1
log "check6 probe build rc=$?"
"$CT/release/memra-probe" > "$O/check6-probe-run.log" 2>&1
log "check6 probe run rc=$? : $(head -n 3 "$O/check6-probe-run.log" | tr '\n' ' ')"
rm -rf ~/.cache/memra-gb10-clean-target
log COVERAGE_DONE
