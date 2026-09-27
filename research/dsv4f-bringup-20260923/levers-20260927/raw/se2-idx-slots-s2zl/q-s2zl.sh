#!/usr/bin/env bash
# q-s2zl.sh (second SE pair): main ($1), the B-row lane rebased on it ($2) and the eight-per-CTA
# replay indexer scores on top ($3). The B-row step census at 16 and 4 rows on $2 (captured TP/EP
# steps, PDL off); the merge gates on $2 and $3 (long gate hash, TP/EP rows, wide 16, KV split,
# DSpark on $2); the long gate N B S S B N N B S; served cells-pdl N B S S B N.
set -u
NSHA=$1; BSHA=$2; ISHA=$3
S=/root/rcpt/q-s2zl.summary; R=/root/rcpt/idx-slots-s2zl; mkdir -p $R
until grep -q S2ZK_DONE /root/rcpt/q-s2zk.summary 2>/dev/null; do sleep 60; done
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 24 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; FX=/root/box/dspark-fx-tape416.json; T=/root/box/tape-rebuild.txt
X=/root/lane/target-hm/release
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
# q-s2zk's first main c24 row found no cells file on this box: its replacement, first.
d=/root/rcpt/brow-diet-s2zk/r7-M-c24
if ! grep -q "zk-r7 M c24 rc=0" $S 2>/dev/null; then
  wait_lock
  timeout -k 30 3600 /root/box/cell.sh $d /root/lane/bin-ziM/memra-server /root/box/cells-c24.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve zk-r7 M c24 rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
fi
cd /root/lane/memra && git fetch -q origin main lane/dsv4-brow-diet-20260927 lane/dsv4-idx-slots-20260927
for v in N:$NSHA B:$BSHA I:$ISHA; do IFS=: read n sha <<< "$v"
  git worktree add -f /root/lane/t-zl$n $sha > /dev/null 2>&1; git -C /root/lane/t-zl$n checkout -q --detach $sha
  [[ "$(git -C /root/lane/t-zl$n rev-parse HEAD)" == "$sha" ]] || { echo "TREE_MISMATCH $n" >> $S; exit 1; }
  bash /root/box/build.sh /root/lane/t-zl$n /root/lane/target-hm zl$n
  mkdir -p /root/lane/bin-zl$n; cp $X/memra-server $X/dsv4_tp_replay_long_gate $X/dsv4_rows_gate $X/dsv4_kv_split_gate $X/dsv4-gpu-dspark-gate /root/lane/bin-zl$n/
  echo "build $n $(git -C /root/lane/t-zl$n rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-zl$n.log | tr '\n' ' ')" >> $S
done
sha256sum /root/lane/bin-zl*/* > $R/binaries.sha256
for b in 16 4; do
  d=$R/census-b$b; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; NVIDIA_TF32_OVERRIDE=0 MEMRA_DSV4_PDL=0 DSV4_ROWS_GATE_TOPOLOGY=tp_ep DSV4_ROWS_GATE_PROFILE=$b DSV4_ROWS_GATE_PROFILE_GRAPH=1 timeout 3000 nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,osrt --cuda-graph-trace=node -o $d/nsys -f true /root/lane/bin-zlB/dsv4_rows_gate $M $T 24 64 9>&- > $d/gate.log 2>&1 )
  nsys export --type sqlite -o $d/nsys.sqlite -f true $d/nsys.nsys-rep > /dev/null 2>&1
  python3 /root/box/nsys-ana.py $d/nsys.sqlite 64 > $d/ana.txt 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv -o $d/kern $d/nsys.nsys-rep > /dev/null 2>&1
  rm -f $d/nsys.sqlite $d/nsys.nsys-rep
  echo "census b$b $(grep -hE 'PROFILE|FAILED|panicked' $d/gate.log | cut -c1-120 | tr '\n' ' ') | $(head -3 $d/ana.txt | tr '\n' ' ')" >> $S
done
TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
for n in B I; do XL=/root/lane/bin-zl$n
  locked $R/$n-long-304 $XL/dsv4_tp_replay_long_gate $M $T 304
  echo "long-304 $n rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $R/$n-long-304/gate.log | sort -u | tr '\n' ' ')" >> $S
  locked $R/$n-rows-tpep env DSV4_ROWS_GATE_TOPOLOGY=tp_ep $XL/dsv4_rows_gate $M $T 24 64
  echo "rows tp_ep $n rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/$n-rows-tpep/gate.log | tail -3 | cut -c1-100 | tr '\n' ' ')" >> $S
  locked $R/$n-rows-wide16 env DSV4_ROWS_GATE_TOPOLOGY=tp_ep DSV4_ROWS_GATE_WIDE=16 $XL/dsv4_rows_gate $M $T 24 64
  echo "rows wide16 $n rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/$n-rows-wide16/gate.log | cut -c1-90 | tr '\n' ' ') | $(grep -hE 'WIDE B=' $R/$n-rows-wide16/gate.log | tr '\n' ' ')" >> $S
  locked $R/$n-kv-split env DSV4_KV_SPLIT_GATE_MAX_SEQ=1048576 $XL/dsv4_kv_split_gate $M $T 3000 100
  echo "kv-split $n rc=$? $(grep -hE '^PASS|FAIL|panicked' $R/$n-kv-split/gate.log | tail -3 | cut -c1-100 | tr '\n' ' ')" >> $S
done
d=$R/B-dspark-tpep; mkdir -p $d; wait_lock
( export $TPEP MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ATTENTION_TP_GATE=1; /root/box/gate.sh $d /root/lane/bin-zlB/dsv4-gpu-dspark-gate $M $FX $d/out 2 0,1 --served --tpep )
echo "dspark-tpep B $(grep -hoE 'proposal sha [0-9a-f]+' $d/gate.log | awk '{print $3}' | cut -c1-16 | tr '\n' ' ') | $(grep -hE 'GATE \[|GATE_DONE' $d/gate.log | cut -c1-120 | tr '\n' ' ')" >> $S
bin() { case $1 in N) echo /root/lane/bin-zlN ;; B) echo /root/lane/bin-zlB ;; *) echo /root/lane/bin-zlI ;; esac; }
i=0
for a in N B S S B N N B S; do
  i=$((i+1)); d=$R/long-t$i-$a
  locked $d $(bin $a)/dsv4_tp_replay_long_gate $M $T 304
  echo "long t$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}' $d/gate.log) $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
i=0
for a in N B S S B N; do
  i=$((i+1)); d=$R/r$i-$a; wait_lock
  timeout -k 30 2700 /root/box/cell.sh $d $(bin $a)/memra-server /root/box/cells-pdl.txt MEMRA_ENV_AUDIT=on > $d.out 2>&1
  echo "serve r$i $a rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-150 | tr '\n' ' ')" >> $S
done
for n in N B I; do git -C /root/lane/memra worktree remove --force /root/lane/t-zl$n; done; git -C /root/lane/memra worktree prune
echo "S2ZL_DONE $ISHA" >> $S
