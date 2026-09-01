#!/bin/bash
# 30k affinity A/B, take 2 (fixed binary + fixed instrument):
# - unique port per arm x round: health 200 can never be another server (arm identity law)
# - pgrep-clear wait before every boot (teardown can outlive 120s)
# - kill by PID + bracketed basename pkill; never a stale path pattern
set -u
exec 9>/root/gemmprime.lock
flock -w 28800 9 || { echo "lock timeout" >&2; exit 1; }
BASE=$(grep "^ENVV=" /root/agentic8.sh | sed "s/^ENVV=//; s/^\"//; s/\"$//")
POLICY="MEMRA_LOAD_MTP=1 MEMRA_MTP_HEADS=3 MEMRA_SPEC_K=3 MEMRA_SPEC_PMIN=0.5 MEMRA_SPEC_PMIN0=1"
OUT=/root/long30k2.txt; : > $OUT; : > /root/long30k-rows.txt
BIN=/root/memra-server.ringfix
MD5=$(md5sum $BIN | cut -c1-12)
for M in MEMRA_ROWS_TAB_RESTAGE MEMRA_STEP_GEMM_PRIME_SUFFIX ckpt-bounds-take-v2 "checkpoint SWA restore refused"; do
  strings $BIN | grep -q "$M" || { echo "ABORT: $BIN lacks marker $M" >> $OUT; echo LONG30K2-DONE >> $OUT; exit 1; }
done
echo "bin=$MD5 markers=verified(incl-ringfix)" >> $OUT
for RND in 1 2 3; do
 for AFF in 1 0; do
  for i in $(seq 1 60); do pgrep -f "memra-server[.]" > /dev/null || break; sleep 5; done
  pgrep -f "memra-server[.]" > /dev/null && { echo "rnd=$RND aff=$AFF ABORT stale server survives" >> $OUT; break 2; }
  P=$((19000 + RND*10 + AFF))
  LOG=/root/long30k2-$AFF-$RND.log
  env $BASE $POLICY MEMRA_AFFINITY=$AFF MEMRA_CTX=262144 MEMRA_SERVE_SPEC=1 \
    MEMRA_MODELS="step37=/root/models/step37-flash-nvfp4" MEMRA_ADDR=127.0.0.1:$P \
    nohup setsid $BIN > $LOG 2>&1 &
  SRV=$!
  for i in $(seq 1 500); do curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$P/health 2>/dev/null | grep -q 200 && break; sleep 5; done
  if ! curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$P/health | grep -q 200; then
    echo "rnd=$RND aff=$AFF booted=NO - arm invalid" >> $OUT; kill $SRV 2>/dev/null; sleep 15; continue
  fi
  kill -0 $SRV 2>/dev/null || { echo "rnd=$RND aff=$AFF ABORT health-200 but boot PID dead (foreign server?)" >> $OUT; break 2; }
  P=$P ARM=aff$AFF RND=$RND python3 /root/long30k.py
  echo "rnd=$RND aff=$AFF rewound=$(grep -ac "plain-affinity rewound" $LOG) illegal=$(grep -ac ILLEGAL $LOG) trap87=$(grep -ac "#87" $LOG) lap=$(grep -ac lapped $LOG) panics=$(grep -ac panicked $LOG) fullprime=$(grep -ac "full prime" $LOG)" >> $OUT
  kill $SRV 2>/dev/null; sleep 5; pkill -f "memra-server[.]ringfix" 2>/dev/null; sleep 10
 done
done
echo "=== rows:" >> $OUT
cat /root/long30k-rows.txt >> $OUT
echo LONG30K2-DONE >> $OUT
