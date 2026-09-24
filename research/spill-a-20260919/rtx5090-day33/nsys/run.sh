#!/usr/bin/env bash
# WP-A day 33: two Nsight Systems boots of the stall cell's promote arm (the day-32 H2D binary, the day-33 binary with the
# timeline fields), 9B on the local RTX 5090. The nsys precheck ran before this (no lock, no GPU). One bounded wait for
# /tmp/memra-5090.lock (90 x 120 s) behind the other lanes, never inside another hold; under the hold: per binary one boot
# with a 60 s bound on server readiness (not ready: stop it and move on), stall_cell.py --mode promote --n 2, stop. The
# lock is released before any post-processing. Output stays under /tmp/wt-a-d33/nsys2.
set -uo pipefail
OUT=/tmp/wt-a-d33/nsys2
export TMPDIR=/tmp/wt-a-d33/nsystmp
cd /home/avifenesh/projects/wt-spill-a
MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
PORT=18134
exec 9>/tmp/memra-5090.lock
echo "$(date -u +%FT%TZ) waiting" > $OUT/run.log
got=0
for i in $(seq 1 90); do flock -w 120 9 && { got=1; break; }; done
[ $got = 1 ] || { echo "$(date -u +%FT%TZ) NOT RUN: the lock never freed" >> $OUT/run.log; exit 2; }
echo "$(date -u +%FT%TZ) hold start; card: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')" >> $OUT/run.log
for tag in d33t d32; do
  BIN=/tmp/wt-a-d33/bin/memra-server-t; [ $tag = d32 ] && BIN=/tmp/wt-a-d33/bin/memra-server-d32
  systemd-run --user --scope -q -p CPUQuota=800% -p MemoryMax=16G env TMPDIR=$TMPDIR CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai \
      "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 \
      MEMRA_PREFIX_CACHE_MB=64 MEMRA_KV_HOST_MB=8192 MEMRA_KV_HOST_CONTRACTS=1 \
      nsys profile -t cuda,osrt --sample=none --cpuctxsw=none -o $OUT/$tag -f true "$BIN" > $OUT/$tag-server.log 2>&1 &
  NPID=$!
  ready=0
  for _ in $(seq 1 30); do curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }; kill -0 $NPID 2>/dev/null || break; sleep 2; done
  if [ $ready = 1 ]; then
    python3 research/spill-a-20260919/stall_cell.py --port $PORT --mode promote --n 2 --server-log $OUT/$tag-server.log --out $OUT/$tag-stall --tag nsys-$tag > $OUT/$tag-stall.log 2>&1
    echo "$(date -u +%FT%TZ) $tag stall rc=$?" >> $OUT/run.log
  else
    echo "$(date -u +%FT%TZ) $tag NOT READY within 60 s; stopped" >> $OUT/run.log
  fi
  SPID=$(pgrep -f "^$BIN\$" | head -1)
  [ -n "$SPID" ] && kill -TERM "$SPID"
  for _ in $(seq 1 60); do kill -0 $NPID 2>/dev/null || break; sleep 1; done
  kill -0 $NPID 2>/dev/null && { echo "$(date -u +%FT%TZ) $tag nsys still running after 60 s; killed" >> $OUT/run.log; pkill -P $NPID; kill $NPID; }
  wait $NPID 2>/dev/null
  echo "$(date -u +%FT%TZ) $tag done" >> $OUT/run.log
done
echo "$(date -u +%FT%TZ) hold end; card: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')" >> $OUT/run.log
flock -u 9
echo NSYS-RUN-DONE >> $OUT/run.log
