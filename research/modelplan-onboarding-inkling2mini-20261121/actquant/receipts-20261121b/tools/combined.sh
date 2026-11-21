#!/bin/bash
set -u
cd /home/avifenesh/.cache/inkling2mini/actq-fix/runs
./ink2-actq-gate.r3 --capture capture-r3.bin > gate-r3.log 2>&1; echo gate-rc=$? >> gate-r3.log
bash /home/avifenesh/.cache/inkling2mini/actq-fix/bench/run_bench.sh /home/avifenesh/.cache/inkling2mini/actq-fix/runs/bench-r3 /home/avifenesh/.cache/inkling2mini/actq-fix/runs/ink2-actq-gate.r3; echo bench-script-rc=$? >> gate-r3.log
