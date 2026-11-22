#!/bin/bash
L=/home/avifenesh/.cache/inkling2mini/gpu-logs
T64=/home/avifenesh/.cache/inkling2mini/oracle/tapes-var/p3.exact_f64.chunk1024.as_f32
BIN=/home/avifenesh/.cache/inkling2mini/gpu-frozen6
M=/home/avifenesh/models/inkling2-mini
O=$L/b7
mkdir -p $O
cd $O
cat $BIN/SHA256SUMS
$BIN/ink2-gpu $M --class float --tape $T64 --context 8448 --score --out p3-float-vs-f64.json > p3-float-vs-f64.log 2>&1; echo "RC=$?"; tail -n 2 p3-float-vs-f64.log
$BIN/ink2-gpu $M --class declared --tape $T64 --context 8448 --score --out p3-declared-vs-f64.json > p3-declared-vs-f64.log 2>&1; echo "RC=$?"; tail -n 2 p3-declared-vs-f64.log
