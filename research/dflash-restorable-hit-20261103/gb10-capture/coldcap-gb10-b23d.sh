#!/usr/bin/env bash
# Runs ON spark-b23d under the caller's sparklock lease ($LEASE): the cold-drafter capture fix
# (c28fb5ee0e) on GB10.
#   1. build memra-server at the fix with the cell build profile (2 jobs, one codegen unit, fat
#      LTO, 121a), in the cell tree whose target already holds main's build;
#   2. the cold-drafter exactness gate with the doors on by device default, plus the serving
#      shape check (--check-log);
#   3. the GGUF DFlash2 agentic cell c1 A/B, main (3bbdf42ff9 code = main 5d70118b99) against the
#      fix, order main/fix/fix/main, fresh boot each, doors on by device default,
#      MEMRA_PREFIX_STABLE_BOUNDARY=0, bench client on the unit (arms_local2.sh).
set -u
LEASE=${LEASE:?lease}
D=$HOME/memra-run
O=$D/coldcap
mkdir -p "$O"
L=$O/chain.log
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) [coldcap] $*" | tee -a "$L"; }
FIX=c28fb5ee0eaf
MAIN_BIN=$D/bin-main-3bbdf42ff9/memra-server   # copied from the cb33 cell bins (sha256 24fc6045841f)
T=$HOME/.cache/memra-cells/gb10/tree-121a
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH SPARKLOCK_LEASE=$LEASE
log "start lease=$LEASE"
cd "$T" || { log "no tree"; exit 2; }
git checkout -q --detach --force c28fb5ee0eaf0f3d9fb843b006d355a62c50389d || { log "checkout failed"; exit 2; }
log "tree at $(git rev-parse --short=10 HEAD)"
env CARGO_BUILD_JOBS=2 CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 CARGO_PROFILE_RELEASE_LTO=fat \
  MEMRA_CUDA_ARCH=121a CARGO_TARGET_DIR="$T/target" \
  flock "$HOME/.cache/memra-rig-build.lock" nice -n 15 cargo build --release --locked --quiet -p memra-server --bin memra-server \
  > "$O/build.log" 2>&1 || { log "build FAILED"; exit 3; }
mkdir -p "$D/bin-fix-$FIX" "$D/bin-main-3bbdf42ff9"
cp "$T/target/release/memra-server" "$D/bin-fix-$FIX/"
log "binaries fix=$(sha256sum "$D/bin-fix-$FIX/memra-server" | cut -c1-12) main=$(sha256sum "$D/bin-main-3bbdf42ff9/memra-server" | cut -c1-12)"
DOORS=default M=$HOME/models/memra-q38 GGUF=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf \
  DF=$HOME/models/memra-q38/Qwen3.8-27B-DFlash2-memra RANKS=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/q38-ranks-sxc32768.gguf.txt \
  timeout 2400 bash "$T/research/dflash-restorable-hit-20261103/cold-drafter-gate.sh" "$D/bin-fix-$FIX/memra-server" "$O/gate-fix" \
  > "$O/gate-fix.out" 2>&1
log "gate: $(grep -E 'GATE|SHAPE|=' "$O/gate-fix/compare.txt" | tr '\n' ' ') $(grep -a 'prefix restore doors default' "$O/gate-fix/serve-arm.log" | head -1)"
LEASE=$LEASE UNIT=spark-b23d SERVE=$D/serve-batchspec.sh bash "$D/arms_local2.sh" "$O" \
  c1-main-a:bin-main-3bbdf42ff9:1:DEF c1-fix-a:bin-fix-$FIX:1:DEF \
  c1-fix-b:bin-fix-$FIX:1:DEF c1-main-b:bin-main-3bbdf42ff9:1:DEF
log COLDCAP_DONE
