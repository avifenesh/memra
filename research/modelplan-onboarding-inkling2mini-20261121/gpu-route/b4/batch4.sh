#!/bin/bash
L=/home/avifenesh/.cache/inkling2mini/gpu-logs
T=/home/avifenesh/.cache/inkling2mini/oracle/tapes
BIN=/home/avifenesh/.cache/inkling2mini/gpu-frozen4
M=/home/avifenesh/models/inkling2-mini
O=$L/b4
mkdir -p $O
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader
cat /proc/loadavg
$BIN/ink2-gpu-gate > $O/gate.txt 2>&1; echo "GATE_RC=$?"; tail -1 $O/gate.txt
$BIN/ink2-gpu $M --class declared --tape $T/p2_window.actq_f32 --score --force-router --force-moe-in --out $O/p2-declared-forced-moe.json; echo "RC=$?"
$BIN/ink2-gpu $M --class declared --tape $T/p1_chat.actq_f32 --score --force-router --force-moe-in --out $O/p1-declared-forced-moe.json; echo "RC=$?"
$BIN/ink2-gpu $M --class declared --tape $T/p3_logscale.actq_f32 --context 8448 --score --force-router --force-moe-in --out $O/p3-declared-forced-moe.json; echo "RC=$?"
$BIN/ink2-gpu $M --class declared --tape $T/p1_chat.actq_f32 --score --gen 32 --chunkings 26,1 --limit 57 --out $O/p1-declared.json; echo "RC=$?"
$BIN/ink2-gpu $M --class declared --tape $T/p1_chat.exact_f32 --score --out $O/p1-declared-vs-float.json; echo "RC=$?"
$BIN/ink2-gpu $M --class declared --tape $T/p2_window.exact_f32 --score --chunkings 512,300 --limit 1500 --out $O/p2-declared-vs-float.json; echo "RC=$?"
$BIN/ink2-gpu $M --class float --tape $T/p2_window.exact_f32 --score --red dequant-bf16 --out $O/p2-float-red-dequant.json; echo "RC=$?"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader
