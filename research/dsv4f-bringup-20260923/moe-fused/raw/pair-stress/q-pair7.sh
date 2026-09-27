#!/usr/bin/env bash
# q-pair7.sh (2x RTX PRO 6000 WS pod), after q-pair6.sh: the fused2 hang stress on the pod where
# it happened (short/plain/r13-fused2). The same stress on the other pair ran 288 fused2 and 288
# main requests with no stall. Here: fused2 (899aac4d0) and main (6978f5fac), 4 boots each of
# 64 greedy ignore-eos requests plus 8 sampled, order F M M F F M M F. The pod refuses ptrace, so
# a watchdog records every thread's /proc state, wchan, syscall and user/system ticks twice, 5 s
# apart, once forward progress stalls past 90 s, and then stops the server.
while ! grep -q PAIR6_DONE /root/rcpt/q-pair6.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-pair7.summary; R=/root/rcpt/stress7; mkdir -p $R
threads() {  # threads <pid> <out>
  for t in /proc/$1/task/*; do
    printf '%s %s %s wchan=%s syscall=%s ticks=%s\n' "${t##*/}" "$(cat $t/comm 2>/dev/null)" \
      "$(awk '{print $3}' $t/stat 2>/dev/null)" "$(cat $t/wchan 2>/dev/null)" \
      "$(cat $t/syscall 2>/dev/null | cut -d' ' -f1-3)" "$(awk '{print $14"u/"$15"s"}' $t/stat 2>/dev/null)"
  done > $2
}
watchdog() {  # watchdog <row-dir>
  local d=$1 port key pid age
  while [[ ! -f $d/controller.log ]] || ! grep -q READY $d/controller.log; do sleep 5; [[ -f $d/controller.log ]] && grep -q TERMINAL $d/controller.log && return; done
  port=$(grep -o 'port=[0-9]*' $d/controller.log | head -1 | cut -d= -f2)
  while ! grep -q TERMINAL $d/controller.log 2>/dev/null; do
    pid=$(pgrep -x memra-server | head -1)
    key=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep '^MEMRA_API_KEY=' | cut -d= -f2)
    age=$(curl -s -m 5 -H "authorization: Bearer $key" http://127.0.0.1:$port/health | python3 -c "import json,sys;d=json.load(sys.stdin);r=d['worker']['routes'][0];print(r['forward_progress_age_ms'] if r['phase']=='busy' else 0)" 2>/dev/null || echo 0)
    if [[ ${age:-0} -gt 90000 ]]; then
      threads $pid $d/hang-threads-a.txt; nvidia-smi --query-gpu=index,utilization.gpu,clocks.sm,power.draw --format=csv > $d/hang-gpu.csv
      sleep 5; threads $pid $d/hang-threads-b.txt
      echo "HANG captured age_ms=$age" >> $d/controller.log
      kill -TERM $pid
      return
    fi
    sleep 10
  done
}
i=0
for arm in fused2 main main fused2 fused2 main main fused2; do
  i=$((i+1)); d=$R/r$i-$arm
  while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
  watchdog $d &
  W=$!
  /root/box/cell.sh $d /root/lane/target-$arm/release/memra-server /root/box/cells-stress2.txt > $d.out 2>&1
  rc=$?; kill $W 2>/dev/null
  echo "stress r$i $arm rc=$rc $(grep -hE 'CELL |HANG' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-120 | tr '\n' ' ')" >> $S
done
echo PAIR7_DONE >> $S
