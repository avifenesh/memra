#!/bin/bash
# RTX 5090 run at the stack restacked onto main 72f0e26fba (#1082 reclaim spares the restorable tail, #1083 the
# restore grid law + one restore switch). The hand merges sit in serving code (worker.rs): the reclaim spare's
# gate, the host tier's carries_recurrent, the hit-site grid-law line, the deleted partial-restore and
# host-contracts doors. Cells that execute those paths:
#   release battery (kernel-check, argmax, run-spec) at the tip;
#   the restorable-hit cell (A2/A17a): r3's reclaim-on-defer must now spare the tail and the preference run;
#   the DSpark pair cell under the one switch (=1) and at the 5090 device default (switch OFF);
#   the host-spill identity gate's drafter arm (host tier lookup under the grid law);
#   serve-smoke.
#   gpu-1107u.sh <tip sha>
set -u
TIP=$1
W=<work>/engine-dsv4-glm
cd $W || exit 2
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "worktree dirty" >&2; exit 2; }
git fetch -q <work>/restack-1105 fix4/gates || exit 2
git checkout -q --detach $TIP || exit 2
export MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=<work>/target-cuda CARGO_BUILD_JOBS=2
L=<work>/logs-gpu-1107v; mkdir -p $L; S=$L/status
C=<work>/cells-1107u
H=$(git rev-parse HEAD); echo "head $H start $(date '+%m-%d %H:%M')" > $S
REL=$CARGO_TARGET_DIR/release
B() { flock <cache>/memra-rig-build.lock nice -n 15 "$@"; }
G() { local ttl=$1 why=$2; shift 2; gpulock run --ttl "$ttl" --for "memra fix stack on main 72f0e26fba: $why" --wait 3h -- "$@"; }
note() { echo "$* [$(date +%H:%M)]" >> $S; }
B bash -c "git ls-files -z | xargs -0 touch -c && cargo build --release -p memra-engine --bins -p memra-server --bin memra-server -p memra-tokenizer --bin tok-parity" > $L/build.log 2>&1; note "build rc=$?"
need=$(cargo tree --offline -p memra-server -p memra-engine -p memra-tokenizer -e normal,build --prefix none 2>/dev/null | grep -o -E '^memra-[a-z0-9-]+' | sort -u)
miss=""; for c in $need; do grep -q -E "Compiling $c v[0-9.]+ \($W/crates/$c\)" $L/build.log || miss="$miss $c"; done
note "closure: $(echo $need | wc -w) crates, not compiled from this worktree:${miss:- none}"
note "stamp server=$(grep -a -c "${H:0:12}" $REL/memra-server)"
cp $REL/memra-server $L/memra-server-${H:0:12}
BIN=$L/memra-server-${H:0:12}
Q38=<models>/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf; DRAFT=<cache>/memra-fix4/dflash2-q38
for ctx in 8192 4096; do
  G 1h "restorable-hit cell ctx $ctx (A2/A17a, reclaim spare)" env RCTX=$ctx python3 $C/serve_cells.py restorable $BIN $L/serve-$ctx > $L/serve-restorable-$ctx.log 2>&1
  note "serve-restorable-$ctx rc=$? :: $(grep -a 'CELL' $L/serve-restorable-$ctx.log | cut -c1-220 | tr '\n' '|' | cut -c1-1600)"
done
for arm in on:MEMRA_DSPARK_PREFIX_RESTORE=1 default:MEMRA_DSPARK_PREFIX_RESTORE=default; do
  G 45m "DSpark pair cell, switch ${arm%%:*}" env EV=$L/dspark-cell-${arm%%:*} BIN=$BIN MODEL=$Q38 DRAFT=$DRAFT WARM_ENV="${arm#*:}" bash research/dspark-partial-restore-20261104/dspark-pair-cell.sh > $L/dspark-cell-${arm%%:*}.log 2>&1
  note "dspark-cell-${arm%%:*} rc=$? $(grep -E 'warm==cold|DSPARK-PAIR CELL|strict-prefix restores' $L/dspark-cell-${arm%%:*}.log | tr '\n' ' ') :: $(grep -a -h -E 'restore grid law|\[dspark\] (door|switch)|prefix-restore' $L/dspark-cell-${arm%%:*}/warm-server.log 2>/dev/null | head -n 3 | cut -c1-200 | tr '\n' '|')"
done
G 40m "host spill identity, drafter arm, one switch" env MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=$DRAFT MEMRA_DSPARK_PREFIX_RESTORE=1 bash tools/kv-host-spill-identity-gate.sh --external-lock 9 $Q38 $BIN $L/identity-dspark > $L/identity-dspark.log 2>&1
note "identity-dspark rc=$? $(grep -E 'sized:|GATE|FAIL' $L/identity-dspark.log | tr '\n' ' ' | cut -c1-300)"
G 1h "serve-smoke" bash tools/serve-smoke.sh > $L/serve-smoke.log 2>&1
note "serve-smoke rc=$? $(grep -c '  ok: ' $L/serve-smoke.log) ok, $(grep -c '  FAIL: ' $L/serve-smoke.log) FAIL"
G 2h "release battery" bash tools/release-battery.sh --evidence-dir $L/battery-${H:0:12} > $L/battery.log 2>&1
note "battery rc=$? $(grep -m1 -oE 'RELEASE BATTERY (PASS|FAIL)' $L/battery.log) $(grep -a -E 'kernel-check +(PASS|FAIL)' $L/battery.log | head -n 1 | cut -c1-160)"
note "done"
