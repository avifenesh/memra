#!/usr/bin/env bash
# q-v3d.sh (replaces q-v3c.sh: 4 boots per arm of 64 ignore-eos requests; a VM where gdb can attach), after q-v3b.sh: stress for the r13-fused2 hang
# on the WS pair campaign (the serve thread spun on the CPU with both GPUs idle during a greedy
# ignore-eos request). fused2 lane binary (899aac4d0) and its base (main 6978f5fac), plain, 4 boots
# each of 64 greedy ignore-eos requests plus 8 sampled; the campaign hit 1 hang in about 105 fused requests. A watchdog polls /health; if forward progress stalls past 90 s it captures
# `gdb thread apply all bt` of the server before the cell times out.
while ! grep -q V3B_DONE /root/rcpt/q-v3b.summary 2>/dev/null && ! grep -q V3_DONE /root/rcpt/q-v3.summary 2>/dev/null; do sleep 30; done
while ! grep -q V3_DONE /root/rcpt/q-v3.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-v3d.summary; R=/root/rcpt/stress; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
command -v gdb > /dev/null || apt-get install -y -qq gdb > /dev/null 2>&1
cd /root/lane/memra && git fetch -q origin lane/dsv4-moe-fused-20260923 main
git worktree add -f /root/lane/t-fused2 899aac4d0 > /dev/null 2>&1; git -C /root/lane/t-fused2 checkout -q --detach 899aac4d0
git worktree add -f /root/lane/t-main 6978f5fac > /dev/null 2>&1; git -C /root/lane/t-main checkout -q --detach 6978f5fac
for a in fused2 main; do
  [[ -d /root/lane/target-$a ]] || cp -a /root/lane/target-pipe /root/lane/target-$a
  bash /root/box/build.sh /root/lane/t-$a /root/lane/target-$a $a
  echo "build $a $(git -C /root/lane/t-$a rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-$a.log | tr '\n' ' ')" >> $S
done
watchdog() {  # watchdog <row-dir>
  local d=$1
  while [[ ! -f $d/controller.log ]] || ! grep -q READY $d/controller.log; do sleep 5; [[ -f $d/controller.log ]] && grep -q TERMINAL $d/controller.log && return; done
  local port key pid
  port=$(grep -o 'port=[0-9]*' $d/controller.log | head -1 | cut -d= -f2)
  while ! grep -q TERMINAL $d/controller.log 2>/dev/null; do
    pid=$(pgrep -f "memra-server" | head -1)
    key=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep '^MEMRA_API_KEY=' | cut -d= -f2)
    age=$(curl -s -m 5 -H "authorization: Bearer $key" http://127.0.0.1:$port/health | python3 -c "import json,sys;d=json.load(sys.stdin);r=d['worker']['routes'][0];print(r['forward_progress_age_ms'] if r['phase']=='busy' else 0)" 2>/dev/null || echo 0)
    if [[ ${age:-0} -gt 90000 ]]; then
      gdb -p $pid -batch -ex "info threads" -ex "thread apply all bt 25" > $d/hang-bt.txt 2>&1
      echo "HANG captured age_ms=$age" >> $d/controller.log
      kill -TERM $pid  # a stuck serve thread would hold every later request of the row
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
echo V3D_DONE >> $S
