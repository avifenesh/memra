#!/bin/bash
# RTX 5090 run at the stack restacked onto main c347e782d2 (#1098 memra#1093 part 1: the DSpark pool grid law; #1092, #1089).
# The hand merge kept main's recurrent rewind (grid_rewind_plain) where the stack deletes the grid-rewind door
# (#1062 d6ecb6af77), and #1062 makes park compaction the 5090 default. Cells:
#   memra-server lib tests (CUDA build, no visible device; the full tree has the fixtures);
#   the DSpark pair cell at the 5090 default with turn 2 at max_tokens 96 and 256: the #1062 landing block's
#     lift test (compaction is on by default here; both arms must equal cold, with the grid-rewind line);
#   the pair cell under the one switch; the restorable-hit cell (A2/A17, reclaim spare); host-spill identity
#   (drafter arm); serve-smoke; the release battery.
#   gpu-1107w.sh <tip sha>
set -u
TIP=$1
W=<work>/engine-dsv4-glm
cd $W || exit 2
[ "$(git rev-parse HEAD)" = "$(git rev-parse $TIP)" ] || { echo "worktree not at $TIP" >&2; exit 2; }
export MEMRA_NVCC=/usr/local/cuda/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=<work>/target-cuda CARGO_BUILD_JOBS=2
L=<work>/logs-gpu-1108a; mkdir -p $L; S=$L/status
C=<work>/cells-1107u
PC=<work>/ab-parkcompact/cell
H=$(git rev-parse HEAD); echo "resume (gpu-1108b) at ${H:0:12} $(date '+%m-%d %H:%M'): after the #1102 record; cells from MAX_T2=256 on" >> $S
REL=$CARGO_TARGET_DIR/release
B() { flock <cache>/memra-rig-build.lock nice -n 15 "$@"; }
G() { local ttl=$1 why=$2; shift 2; gpulock run --ttl "$ttl" --for "memra fix stack on main c347e782d2: $why" --wait 3h -- "$@"; }
note() { echo "$* [$(date +%H:%M)]" >> $S; }
BIN=$L/memra-server-${H:0:12}
[ -x $BIN ] && [ "$(grep -a -c "${H:0:12}" $BIN)" -ge 1 ] || { note "no stamped binary at $BIN"; exit 1; }
Q38=<models>/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf; DRAFT=<cache>/memra-fix4/dflash2-q38
for mt in 256; do
  G 30m "#1062 lift: pair cell at the 5090 default, MAX_T2=$mt" env EV=$L/pair-default-mt$mt BIN=$BIN MODEL=$Q38 DRAFT=$DRAFT MAX_T2=$mt WARM_ENV="MEMRA_DSPARK_PREFIX_RESTORE=default" bash $PC/dspark-pair-cell.sh > $L/pair-default-mt$mt.log 2>&1
  note "pair-default-mt$mt rc=$? $(grep -E 'warm==cold' $L/pair-default-mt$mt.log | tr '\n' ' ') :: $(grep -a -h -E 'grid-rewind|park-compact|kv-reuse: [0-9]+ of' $L/pair-default-mt$mt/warm-server.log | cut -c1-130 | tr '\n' '|' | cut -c1-600)"
done
G 30m "pair cell, one switch on" env EV=$L/pair-on BIN=$BIN MODEL=$Q38 DRAFT=$DRAFT WARM_ENV="MEMRA_DSPARK_PREFIX_RESTORE=1" bash $PC/dspark-pair-cell.sh > $L/pair-on.log 2>&1
note "pair-on rc=$? $(grep -E 'warm==cold|DSPARK-PAIR CELL|strict-prefix restores' $L/pair-on.log | tr '\n' ' ')"
G 1h "restorable-hit cell (A2/A17)" python3 $C/serve_cells.py restorable $BIN $L/serve > $L/serve-restorable.log 2>&1
note "serve-restorable rc=$? :: $(grep -a 'CELL' $L/serve-restorable.log | cut -c1-200 | tr '\n' '|' | cut -c1-1400)"
G 40m "host spill identity, drafter arm" env MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=$DRAFT MEMRA_DSPARK_PREFIX_RESTORE=1 bash tools/kv-host-spill-identity-gate.sh --external-lock 9 $Q38 $BIN $L/identity-dspark > $L/identity-dspark.log 2>&1
note "identity-dspark rc=$? $(grep -E 'sized:|GATE|FAIL' $L/identity-dspark.log | tr '\n' ' ' | cut -c1-300)"
G 1h "serve-smoke" bash tools/serve-smoke.sh > $L/serve-smoke.log 2>&1
note "serve-smoke rc=$? $(grep -c '  ok: ' $L/serve-smoke.log) ok, $(grep -c '  FAIL: ' $L/serve-smoke.log) FAIL"
G 2h "release battery" bash tools/release-battery.sh --evidence-dir $L/battery-${H:0:12} > $L/battery.log 2>&1
note "battery rc=$? $(grep -m1 -oE 'RELEASE BATTERY (PASS|FAIL)' $L/battery.log) $(grep -a -E 'kernel-check +(PASS|FAIL)' $L/battery.log | head -n 1 | cut -c1-160)"
note "done"
