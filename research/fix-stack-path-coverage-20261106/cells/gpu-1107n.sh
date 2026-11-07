#!/bin/bash
# RTX 5090 delta run at the final restacked tip (onto main cf85a23ff5: #1077 F8F4 v2 GEMM, GB10-only route, and #1080 data only; plus the probe gate fix 1a6b6eb671)
# after gpu-1107.sh validated the stack at a0a6816ece (onto 65d5887596). The tip's change from a0a6816ece
# is exactly #1077's delta on every layer (restack proof in decisions 05:15). Cells: build + closure,
# kernel-check FULL (incl. the MMQ-F8F4-V2 bit cells), fast-gate's prime/argmax/spec probes as named
# diagnostics (the plan expands to tier 2, which gpu-1107's local-ci covered at a0a6816ece), the
# restorable-hit cell at a context that leaves the restorable entry resident, and serve-smoke.
#   gpu-1107n.sh <tip sha>
set -u
TIP=$1
W=<work>/engine-dsv4-glm
cd $W || exit 2
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "worktree dirty" >&2; exit 2; }
git fetch -q <work>/restack-1105 fix4/gates || exit 2
git checkout -q --detach $TIP || exit 2
export MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=<work>/target-cuda CARGO_BUILD_JOBS=2
L=<work>/logs-gpu-1107n; mkdir -p $L; S=$L/status
C=<work>/cells-1106
H=$(git rev-parse HEAD); echo "head $H start $(date '+%m-%d %H:%M')" > $S
REL=$CARGO_TARGET_DIR/release
B() { flock <cache>/memra-rig-build.lock nice -n 15 "$@"; }
G() { local ttl=$1 why=$2; shift 2; gpulock run --ttl "$ttl" --for "memra fix-stack final tip 1107 (onto cf85a23ff5): $why" --wait 3h -- "$@"; }
note() { echo "$* [$(date +%H:%M)]" >> $S; }
B bash -c "git ls-files -z | xargs -0 touch -c && cargo build --release -p memra-engine --bins -p memra-server --bin memra-server -p memra-tokenizer --bin tok-parity" > $L/build.log 2>&1; note "build rc=$?"
need=$(cargo tree --offline -p memra-server -p memra-engine -p memra-tokenizer -e normal,build --prefix none 2>/dev/null | grep -o -E '^memra-[a-z0-9-]+' | sort -u)
miss=""; for c in $need; do grep -q -E "Compiling $c v[0-9.]+ \($W/crates/$c\)" $L/build.log || miss="$miss $c"; done
note "closure: $(echo $need | wc -w) crates, not compiled from this worktree:${miss:- none}"
note "stamp server=$(grep -a -c "${H:0:12}" $REL/memra-server)"
cp $REL/memra-server $L/memra-server-${H:0:12}; cp $REL/kernel-check $L/kernel-check-${H:0:12}
BIN=$L/memra-server-${H:0:12}
# the probe recovery gate fixed in 1a6b6eb671 (local-ci stopped on its arm A at a0a6816ece: the script ran inside the boot)
G 30m "GPU probe recovery gate (fixed)" python3 tools/gpu-probe-recovery-gate.py --model <hf-models>/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf --bin $BIN --out $L/probe-gate > $L/probe-gate.log 2>&1
note "probe-gate rc=$? $(grep -E 'A recover|B latch|C fatal|GPU-PROBE-RECOVERY' $L/probe-gate.log | tr '\n' ' ' | cut -c1-500) :: $(grep -h armed $L/probe-gate/*/fake-smi-trace.txt 2>/dev/null | tr '\n' ' ')"
G 30m "kernel-check FULL" $L/kernel-check-${H:0:12} > $L/kc.log 2>&1
note "kernel-check rc=$? $(grep -a -E 'ALL GREEN|FAIL' $L/kc.log | tail -n 2 | tr '\n' ' ' | cut -c1-200) f8f4-v2-cells=$(grep -a -c -i 'f8f4.v2\|F8F4-V2' $L/kc.log)"
P=q9,q35,q35slru,q35spec,g31,g26,e4b,k27,kat,g12,g12d,chunkinv,chunkinvc,ptick,ptickc,pbg9,pbg9c,amargin,amarginc,amarginb,amarginbc,gwstress,isogap,isogapc,accept,sstress
G 2h "fast-gate named probes (prime, argmax, spec)" nice -n 15 bash tools/fast-gate/fast-gate.sh --tier 1 --probes $P > $L/fastgate.log 2>&1
note "fast-gate rc=$? $(grep -E 'kernel-check \(|: (PASS|FAIL|SKIP)' $L/fastgate.log | sed 's/(gates green + golden token-identical, [0-9]*s)//' | tr '\n' ' ' | cut -c1-1200)"
G 1h "restorable-hit cell at ctx 4096 (A2/A17a)" env RCTX=4096 python3 $C/serve_cells.py restorable $BIN $L/serve > $L/serve-restorable.log 2>&1
note "serve-restorable rc=$? :: $(grep -a 'CELL' $L/serve-restorable.log | cut -c1-200 | tr '\n' '|' | cut -c1-1400)"
# DSpark cells the a0a6816ece run read with one door only (PREFIX_RESTORE=1, partial and capture off):
# the pair cell's turn 3 went through the plain kv-reuse resume and differed from cold; the drafter-arm
# identity gate's capture-length check failed. Here: both at their designed three doors on this tip, and
# the one-door pair cell on main 65d5887596's own binary and on this tip, to see whether main reads the same.
Q38=<models>/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf; DRAFT=<cache>/memra-fix4/dflash2-q38
MAINBIN=<work>/logs-gpu-1107n-pre/memra-server-main-65d5887596
G 45m "DSpark pair cell, three doors" env EV=$L/dspark-cell-3doors BIN=$BIN MODEL=$Q38 DRAFT=$DRAFT bash research/dspark-partial-restore-20261104/dspark-pair-cell.sh > $L/dspark-cell-3doors.log 2>&1
note "dspark-cell-3doors rc=$? $(grep -E 'warm==cold|DSPARK-PAIR CELL|strict-prefix restores' $L/dspark-cell-3doors.log | tr '\n' ' ')"
for arm in tip:$BIN main:$MAINBIN; do
  G 45m "DSpark pair cell, one door, ${arm%%:*}" env EV=$L/dspark-cell-1door-${arm%%:*} BIN=${arm#*:} MODEL=$Q38 DRAFT=$DRAFT WARM_ENV="MEMRA_DSPARK_PREFIX_RESTORE=1" bash research/dspark-partial-restore-20261104/dspark-pair-cell.sh > $L/dspark-cell-1door-${arm%%:*}.log 2>&1
  note "dspark-cell-1door-${arm%%:*} rc=$? $(grep -E 'warm==cold|DSPARK-PAIR CELL' $L/dspark-cell-1door-${arm%%:*}.log | tr '\n' ' ') :: $(grep -a -h -o 'kv-reuse: [0-9]* of [0-9]* prompt tokens resumed' $L/dspark-cell-1door-${arm%%:*}/warm-server.log | tr '\n' ' ')"
done
G 40m "host spill identity, drafter arm, three doors" env MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=$DRAFT MEMRA_DSPARK_PREFIX_RESTORE=1 MEMRA_DSPARK_PARTIAL_RESTORE=1 MEMRA_DSPARK_BOUNDARY_CAPTURE=1 bash tools/kv-host-spill-identity-gate.sh --external-lock 9 $Q38 $BIN $L/identity-dspark-3doors > $L/identity-dspark-3doors.log 2>&1
note "identity-dspark-3doors rc=$? $(grep -E 'sized:|GATE|FAIL' $L/identity-dspark-3doors.log | tr '\n' ' ' | cut -c1-300)"
G 1h "serve-smoke" bash tools/serve-smoke.sh > $L/serve-smoke.log 2>&1
note "serve-smoke rc=$? $(grep -c '  ok: ' $L/serve-smoke.log) ok, $(grep -c '  FAIL: ' $L/serve-smoke.log) FAIL"
# the full battery (fast-gate tier 2) at the final tip: the a0a6816ece run stopped at the probe gate
G 4h "local-ci full battery at the final tip" nice -n 15 env MEMRA_CI_LOCK_HELD=1 bash tools/local-ci.sh > $L/local-ci.log 2>&1
note "local-ci rc=$? :: $(grep -a -E 'GPU-PROBE-RECOVERY|accept-gate|spec-on-cache-hit|local-ci: (PASS|FAIL|ALL)|FAILED' $L/local-ci.log | head -n 8 | tr '\n' '|' | cut -c1-600)"
note "done"
