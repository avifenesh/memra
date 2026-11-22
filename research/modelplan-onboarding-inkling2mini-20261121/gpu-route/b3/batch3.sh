#!/bin/bash
L=/home/avifenesh/.cache/inkling2mini/gpu-logs
T=/home/avifenesh/.cache/inkling2mini/oracle/tapes
BIN=/home/avifenesh/.cache/inkling2mini/gpu-frozen
M=/home/avifenesh/models/inkling2-mini
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader
cat /proc/loadavg
$BIN/ink2-gpu-gate > $L/gate.txt 2>&1; echo "GATE_RC=$?"; cat $L/gate.txt
$BIN/ink2-gpu $M --class float --tape $T/p1_chat.exact_f32 --score --gen 32 --chunkings 26,1 --limit 57 --out $L/p1-float.json; echo "RC=$?"
$BIN/ink2-gpu $M --class declared --tape $T/p1_chat.actq_f32 --score --gen 32 --chunkings 26,1 --limit 57 --out $L/p1-declared.json; echo "RC=$?"
$BIN/ink2-gpu $M --class declared --tape $T/p2_window.actq_f32 --score --force-router --force-moe-in --out $L/p2-declared-forced-moe.json; echo "RC=$?"
$BIN/ink2-gpu $M --class float --tape $T/p2_window.exact_f32 --score --chunkings 512,300 --limit 1500 --out $L/p2-float.json; echo "RC=$?"
$BIN/ink2-gpu $M --class declared --tape $T/p1_chat.exact_f32 --score --out $L/p1-declared-vs-float-oracle.json; echo "RC=$?"
