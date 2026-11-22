#!/bin/bash
set -u
cd /home/avifenesh/.cache/inkling2mini/actq-fix/runs
./ink2-actq-gate.r4 --capture capture-r4.bin > gate-r4.log 2>&1; echo gate-rc=$? >> gate-r4.log
bash /home/avifenesh/.cache/inkling2mini/actq-fix/bench/run_bench.sh /home/avifenesh/.cache/inkling2mini/actq-fix/runs/bench-r4 /home/avifenesh/.cache/inkling2mini/actq-fix/runs/ink2-actq-gate.r3 /home/avifenesh/.cache/inkling2mini/actq-fix/runs/ink2-actq-gate.r4; echo bench-script-rc=$? >> gate-r4.log
