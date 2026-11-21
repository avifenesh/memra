#!/bin/bash
L=/home/avifenesh/.cache/inkling2mini/gpu-logs
T=/home/avifenesh/.cache/inkling2mini/oracle/tapes
BIN=/home/avifenesh/.cache/inkling2mini/gpu-frozen6
M=/home/avifenesh/models/inkling2-mini
M512=$L/ckpt512/snapshots/s
C=/home/avifenesh/.cache/inkling2mini/ceiling/clampcheck.sh
O=$L/b6
mkdir -p $O
cd $O
cat $BIN/SHA256SUMS
echo "loadavg $(cat /proc/loadavg)"
bash $C; rc=$?
echo "CLAMPCHECK_RC=$rc"
if [ $rc -eq 1 ]; then
  sudo -n systemctl restart nvidia-powerd
  waitfor --until "bash $C" --timeout 60s; echo "CLAMPCHECK_AFTER_RESTART_RC=$?"
fi
# exit 2 (UNSURE) is not OK: the timed takes below are valid only if the 250 ms telemetry shows
# every loaded sample at P0 and 14001 MHz (judged after the run, judge.py)
nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,enforced.power.limit,clocks_event_reasons.active,temperature.gpu --format=csv -lms 250 > smi.csv 2>&1 &
SMI=$!
run() { tag=$1; shift; echo "MARK $tag $(date +%T.%N)"; "$@" > $tag.log 2>&1; echo "RC $tag=$?"; grep -v '^ink2-gpu: class' $tag.log | tail -n 6; }
run gate $BIN/ink2-gpu-gate
# the debug build runs the loader's per-name debug assertions over every tensor it reads
run debug-load $BIN/ink2-gpu-debug $M --class declared --tape $T/p1_chat.exact_f32 --gen 2 --out debug-load.json
run p1-float $BIN/ink2-gpu $M --class float --tape $T/p1_chat.exact_f32 --score --gen 32 --chunkings 26,1 --limit 57 --out p1-float.json
run p2-float $BIN/ink2-gpu $M --class float --tape $T/p2_window.exact_f32 --score --chunkings 512,300 --limit 1500 --out p2-float.json
run p3-float $BIN/ink2-gpu $M --class float --tape $T/p3_logscale.exact_f32 --context 8448 --score --out p3-float.json
run p3-attn-probe $BIN/ink2-gpu $M --class float --tape $T/p3_logscale.exact_f32 --context 8448 --attn-probe 5,11,17,23,29,35,41,47 --out p3-attn-probe.json
run p2floor512-float $BIN/ink2-gpu $M512 --class float --tape $T/p2_floor512.exact_f32 --score --out p2floor512-float.json
run p1-declared-forced $BIN/ink2-gpu $M --class declared --tape $T/p1_chat.actq_f32 --score --force-router --force-moe-in --out p1-declared-forced-moe.json
run p2-declared-forced $BIN/ink2-gpu $M --class declared --tape $T/p2_window.actq_f32 --score --force-router --force-moe-in --out p2-declared-forced-moe.json
run p1-declared $BIN/ink2-gpu $M --class declared --tape $T/p1_chat.actq_f32 --score --gen 32 --chunkings 26,1 --limit 57 --out p1-declared.json
run atomic-float $BIN/ink2-gpu $M --class float --tape $T/p1_chat.exact_f32 --inject-failure 20 --limit 57 --out atomic-float.json
run atomic-float-red $BIN/ink2-gpu $M --class float --tape $T/p1_chat.exact_f32 --inject-failure 20 --limit 57 --red commit-on-error --out atomic-float-red.json
run atomic-declared $BIN/ink2-gpu $M --class declared --tape $T/p1_chat.exact_f32 --inject-failure 33 --limit 57 --out atomic-declared.json
run red-refusal $BIN/ink2-gpu $M --class declared --tape $T/p1_chat.exact_f32 --red dequant-bf16 --score --out red-refusal.json
run context-refusal $BIN/ink2-gpu $M --class declared --tape $T/p1_chat.exact_f32 --context 1048576 --gen 2 --out context-refusal.json
MAX=$(grep -o 'holds at most [0-9]*' context-refusal.log | grep -o '[0-9]*$')
echo "MAX_CONTEXT=$MAX"
if [ -n "$MAX" ]; then
  NEAR=$(( (MAX - 4096) / 1024 * 1024 ))
  run context-near-max $BIN/ink2-gpu $M --class declared --tape $T/p1_chat.exact_f32 --context $NEAR --gen 8 --out context-near-max.json
fi
echo "MARK timed-start $(date +%T.%N)"
for i in 1 2 3; do
  run gen-float-$i $BIN/ink2-gpu $M --class float --tape $T/p1_chat.exact_f32 --gen 32 --out gen-float-$i.json
  run gen-declared-$i $BIN/ink2-gpu $M --class declared --tape $T/p1_chat.exact_f32 --gen 32 --out gen-declared-$i.json
done
echo "MARK timed-end $(date +%T.%N)"
kill $SMI
