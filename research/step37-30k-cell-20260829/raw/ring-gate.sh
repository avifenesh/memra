#!/bin/bash
set -u
for i in $(seq 1 240); do grep -q BUILD-DONE /root/ring-fix-build.log 2>/dev/null && break; sleep 5; done
grep -q BUILD-DONE /root/ring-fix-build.log || { echo "ABORT: build never finished" > /root/ring-gate.txt; echo RING-GATE-DONE >> /root/ring-gate.txt; exit 1; }
exec 9>/root/gemmprime.lock
flock -w 28800 9 || { echo "lock timeout" > /root/ring-gate.txt; echo RING-GATE-DONE >> /root/ring-gate.txt; exit 1; }
BASE=$(grep "^ENVV=" /root/agentic8.sh | sed "s/^ENVV=//; s/^\"//; s/\"$//")
POLICY="MEMRA_LOAD_MTP=1 MEMRA_MTP_HEADS=3 MEMRA_SPEC_K=3 MEMRA_SPEC_PMIN=0.5 MEMRA_SPEC_PMIN0=1"
BIN=/root/memra-server.ringfix
OUT=/root/ring-gate.txt; : > $OUT
md5sum $BIN | cut -c1-12 >> $OUT
strings $BIN | grep -q "checkpoint SWA restore refused" || { echo "ABORT: fix marker missing" >> $OUT; echo RING-GATE-DONE >> $OUT; exit 1; }
echo "marker=verified" >> $OUT
P=18777
env $BASE $POLICY RUST_BACKTRACE=full MEMRA_AFFINITY=1 MEMRA_CTX=262144 MEMRA_SERVE_SPEC=1 \
  MEMRA_MODELS="step37=/root/models/step37-flash-nvfp4" MEMRA_ADDR=127.0.0.1:$P \
  nohup setsid $BIN > /root/ring-gate-server.log 2>&1 &
SRV=$!
for i in $(seq 1 200); do curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$P/health 2>/dev/null | grep -q 200 && break; sleep 5; done
P=$P python3 /root/panic-repro.py >> $OUT 2>&1
sleep 5
echo "panics=$(grep -c panicked /root/ring-gate-server.log)" >> $OUT
echo "fullprime=$(grep -c "dropping session, full prime" /root/ring-gate-server.log)" >> $OUT
echo "rewound_line=$(grep "spec-affinity: rewound" /root/ring-gate-server.log | tail -1)" >> $OUT
echo "grew_line=$(grep "spec-affinity: grew" /root/ring-gate-server.log | tail -1)" >> $OUT
kill $SRV 2>/dev/null; pkill -f "memra-server[.]ringfix" 2>/dev/null
echo RING-GATE-DONE >> $OUT
