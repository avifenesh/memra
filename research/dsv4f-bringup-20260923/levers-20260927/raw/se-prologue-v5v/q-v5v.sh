#!/usr/bin/env bash
# q-v5v.sh (SE pair):
#   1. #500 past the stall bound: a 48k-token prime carrying a 600 s timeout_ms, on the naked main
#      binary, while the two trees below build.
#   2. The PDL-prologue lane $1 against main $2: builds; the dense-fast component gate (fast against
#      exact-tail bits, both trees, with its isolated kernel timings); on the lane the long gate
#      program hash (PDL on and off), the TP/EP rows gate, the position-split gate and the DSpark
#      TP/EP proposal shas.
#   3. Served, one boot per row, order M P P M M P P M M P (N=5 each), naked defaults.
set -u
PSHA=$1; MSHA=$2
S=/root/rcpt/q-v5v.summary; R=/root/rcpt/prefetch-v5v; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json; T=/root/box/tape-rebuild.txt
rm -rf /root/lane/target-join /root/lane/target-ks /root/lane/target-main10 /root/lane/target-pdl /root/lane/target-vhead
cd /root/lane/memra && git fetch -q origin main lane/dsv4-pdl-prefetch-20260926
for t in pref pmain; do
  sha=$PSHA; [[ $t == pmain ]] && sha=$MSHA
  git worktree add -f /root/lane/t-$t $sha > /dev/null 2>&1; git -C /root/lane/t-$t checkout -q --detach $sha
  [[ -d /root/lane/target-$t ]] || cp -a /root/lane/target-push /root/lane/target-$t
done
( bash /root/box/build.sh /root/lane/t-pmain /root/lane/target-pmain pmain; bash /root/box/build.sh /root/lane/t-pref /root/lane/target-pref pref; echo done > /root/rcpt/v5v-builds.done ) &
# ---- 1. #500 on the naked main binary (functional cell; the builds share the CPU)
R5=/root/rcpt/route-receipt-v5v; mkdir -p $R5
X=/root/lane/target-push/release/memra-server
( exec 9>/tmp/memra-gpu.lock; while ! flock -n 9; do sleep 10; done
  PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()'); KEY="route-$RANDOM"
  env MEMRA_ENV_AUDIT=on MEMRA_MODELS="dsv4f=$M" MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_API_KEY="$KEY" MEMRA_GPU_LOCK=/tmp/memra-gpu.lock setsid nohup $X > $R5/serve.log 2>&1 9>&- &
  pid=$!
  for _ in $(seq 1 900); do sleep 1; [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:$PORT/readyz)" == 200 ]] && break; done
  python3 /root/box/routecell3.py --port $PORT --key $KEY --out $R5/cells.jsonl > $R5/routecell.out 2>&1
  kill -TERM $pid; for _ in $(seq 1 90); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -9 $pid 2>/dev/null
  echo "route 500 $(grep -c '"poll"' $R5/cells.jsonl) polls, $(grep '"request done"' $R5/cells.jsonl | cut -c1-160)" >> $S )
until [[ -f /root/rcpt/v5v-builds.done ]]; do sleep 20; done; rm -f /root/rcpt/v5v-builds.done
for t in pmain pref; do echo "build $t $(git -C /root/lane/t-$t rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-$t.log | tr '\n' ' ') server $(sha256sum /root/lane/target-$t/release/memra-server | cut -c1-16)" >> $S; done
XM=/root/lane/target-pmain/release; XP=/root/lane/target-pref/release
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
sha256sum $XM/memra-server $XP/memra-server $XP/dsv4_tp_replay_long_gate $XP/dsv4_rows_gate $XP/dsv4_kv_split_gate $XP/dsv4-gpu-dspark-gate > $R/binaries.sha256
# ---- 2. gates
for t in pmain pref; do
  mkdir -p /root/lane/cgate-$t
  ( cd /root/lane/t-$t && nice -n 10 nvcc -t 2 -std=c++17 -O3 -fmad=false -Xcompiler=-ffp-contract=off -arch=sm_120a -lineinfo -Xptxas=-v tools/dsv4-dense-fast-gate.cu -lcublasLt -lcublas -ldl -o /root/lane/cgate-$t/component > $R/cgate-build-$t.log 2>&1 )
  locked $R/component-$t /root/lane/cgate-$t/component
  echo "component $t rc=$? $(grep -hE 'PASS|FAIL' $R/component-$t/gate.log | tail -2 | cut -c1-160 | tr '\n' ' ')" >> $S
done
for p in 1 0; do
  locked $R/long-304-pdl$p env MEMRA_DSV4_PDL=$p $XP/dsv4_tp_replay_long_gate $M $T 304
  echo "long-304 lane PDL=$p rc=$? $(grep -hE 'PASS:|FAILED|FIRST|PROGRAM_SHA256|panicked' $R/long-304-pdl$p/gate.log | cut -c1-200 | tr '\n' ' ')" >> $S
done
locked $R/rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $XP/dsv4_rows_gate $M $T 24 64
echo "rows tp_ep rc=$? $(grep -hE 'PASS|FAIL|panicked' $R/rows-tpep/gate.log | tail -3 | cut -c1-200 | tr '\n' ' ')" >> $S
locked $R/split $XP/dsv4_kv_split_gate $M $T 3000 300
echo "split rc=$? $(grep -hE 'PASS: position|FAILED|panicked' $R/split/gate.log | cut -c1-200 | tr '\n' ' ')" >> $S
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
d=$R/dspark-tpep; mkdir -p $d; wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d $XP/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "dspark-tpep lane $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'verdict|GATE \[|GATE_DONE' $d/gate.log | head -n 6 | cut -c1-160 | tr '\n' ' ')" >> $S
# ---- 3. served
bin() { case $1 in M) echo $XM ;; *) echo $XP ;; esac; }
i=0
for arm in M P P M M P P M M P; do
  i=$((i+1)); d=$R/r$i-$arm; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d $(bin $arm)/memra-server /root/box/cells-pdl.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $arm rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
echo "V5V_DONE $PSHA" >> $S
