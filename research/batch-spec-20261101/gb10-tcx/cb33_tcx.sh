#!/usr/bin/env bash
# Runs ON the GB10 box: design C prototype (exact tensor-core form at 16 rows, UP chains per
# iteration). Bench sweep m=16 vs the default (rpscx), ncu on blk.0 FFN launches, kernel-check.
set -uo pipefail
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
export SPARKLOCK_LEASE=$(cat ~/.cache/memra-cb33-lease.txt)
OUT=$HOME/memra-run/step-tcx
mkdir -p "$OUT"
T=$HOME/.cache/memra-batchspec-target/release
cd ~/projects/memra-batchspec
CARGO_TARGET_DIR=$HOME/.cache/memra-batchspec-target nice -n 5 cargo build --release -j 16 \
  -p memra-engine --bin mmvq-b16-bench --bin kernel-check --bin decode-batch-bench > "$OUT/build.log" 2>&1
echo "BUILD=$?" >> "$OUT/build.log"
G=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
B=$T/mmvq-b16-bench
for round in 1 2 3; do
  for v in auto tc1 tc2 tc4; do
    if [ $v = auto ]; then unset MEMRA_BVAR; else export MEMRA_BVAR=$v; fi
    echo "r$round m16 $v $(timeout 600 "$B" "$G" --layers 4 --m 16 --iters 20 2>&1 | grep B16BENCH)" >> "$OUT/sweep.txt"
    unset MEMRA_BVAR
  done
done
NCU=$(command -v ncu)
M="gpu__time_duration.sum,launch__registers_per_thread,sm__warps_active.avg.per_cycle_active,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warp_latency_per_inst_issued.ratio,smsp__inst_executed.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum"
for v in auto tc1 tc2 tc4; do
  if [ $v = auto ]; then E=""; else E="MEMRA_BVAR=$v"; fi
  sudo -E env PATH="$PATH" $E timeout 600 "$NCU" --clock-control none \
    -k "regex:qmatvec_nvfp4_mmvq_b16_" --launch-skip 2 --launch-count 3 --metrics "$M" --csv \
    "$B" "$G" --layers 1 --m 16 --iters 1 > "$OUT/ncu-$v.csv" 2> "$OUT/ncu-$v.err"
  echo "ncu $v rc=$?" >> "$OUT/done.txt"
done
sudo chown "$USER" "$OUT"/*
for v in tc2 tc4 tc1; do
  MEMRA_BVAR=$v MEMRA_KC_FAST=1 timeout 1500 "$T/kernel-check" "$G" > "$OUT/kc-$v.log" 2>&1
  echo "KC $v rc=$?" >> "$OUT/kc-$v.log"
done
cp "$T/decode-batch-bench" "$OUT/dbb"
run() { # tag env
  MEMRA_DECODE_BATCH_CAP=32 MEMRA_DBB_SAMP=greedy timeout 1800 env $2 "$OUT/dbb" "$G" \
    --seqs 16 --chunk 16 --ctx 12288 --steps 16 --reps 3 > "$OUT/$1.log" 2>&1
  echo "b16 d12288 $1: $(grep CHUNKSWEEP "$OUT/$1.log") $(grep TOKFP "$OUT/$1.log" | sed "s/.*\[//" | md5sum | cut -c1-12)" >> "$OUT/ab.txt"
}
i=0
for a in auto s tc2 tc4 tc4 tc2 s auto; do
  i=$((i + 1))
  case $a in auto) E="MEMRA_X=0";; s) E="MEMRA_G4X=s";; *) E="MEMRA_BVAR=$a";; esac
  run r$i-$a "$E"
done
rm -f "$OUT/dbb"
echo "STEP_DONE $OUT"
