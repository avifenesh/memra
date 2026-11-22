#!/bin/bash
L=/home/avifenesh/.cache/inkling2mini/gpu-logs
T=/home/avifenesh/.cache/inkling2mini/oracle/tapes
BIN=/home/avifenesh/.cache/inkling2mini/target-gpu/release
M=/home/avifenesh/models/inkling2-mini
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader
cat /proc/loadavg
$BIN/ink2-gpu-gate; echo "GATE_RC=$?"
$BIN/ink2-gpu $M --class float --tape $T/p1_chat.exact_f32 --score --gen 32 --chunkings 26,1 --limit 57 --out $L/p1-float.json; echo "RC=$?"
$BIN/ink2-gpu $M --class float --tape $T/p1_chat.exact_f32 --score --red dequant-bf16 --out $L/p1-float-red-dequant.json; echo "RC=$?"
$BIN/ink2-gpu $M --class float --tape $T/p1_chat.bf16w_f32 --score --red dequant-bf16 --out $L/p1-bf16w-red-dequant.json; echo "RC=$?"
$BIN/ink2-gpu $M --class float --tape $T/p1_chat.bf16w_f32 --score --out $L/p1-bf16w-float.json; echo "RC=$?"
$BIN/ink2-gpu $M --class float --tape $T/p1_chat.exact_f32 --score --red conv-state --out $L/p1-float-red-conv.json; echo "RC=$?"
$BIN/ink2-gpu $M --class float --tape $T/p1_chat.exact_f32 --score --red ties-high --out $L/p1-float-red-ties.json; echo "RC=$?"
