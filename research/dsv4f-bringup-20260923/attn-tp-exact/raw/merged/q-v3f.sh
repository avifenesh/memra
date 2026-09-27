#!/usr/bin/env bash
# q-v3f.sh (2x RTX PRO 6000 Server Edition), after q-v3e.sh: the exact attention TP replay gate
# rerun (memra #679) with the epoch census fixed for the two row gathers per layer
# (lane/dsv4-attn-tp-exact-20260923 d2a97c826 + topology-lane.patch).
while ! grep -q V3E_DONE /root/rcpt/q-v3e.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-v3f.summary; R=/root/rcpt/attn; mkdir -p $R
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
cd /root/lane/memra && git fetch -q origin lane/dsv4-attn-tp-exact-20260923
git worktree add -f /root/lane/t-attn2 d2a97c826 > /dev/null 2>&1; git -C /root/lane/t-attn2 checkout -q --detach d2a97c826
cd /root/lane/t-attn2 && git apply /root/box/topology-lane.patch
echo "tree attn2 $(git rev-parse HEAD) + topology-lane.patch $(sha256sum /root/box/topology-lane.patch | cut -c1-16)" >> $S
[[ -d /root/lane/target-attn2 ]] || cp -a /root/lane/target-attn /root/lane/target-attn2
bash /root/box/build.sh /root/lane/t-attn2 /root/lane/target-attn2 attn2
X=/root/lane/target-attn2/release
echo "build attn2 $(grep -hE 'EXIT' /root/build-attn2.log | tr '\n' ' ') gate $(sha256sum $X/dsv4_tp_ep_sampled_perf_gate | cut -c1-16)" >> $S
M=/data/dsv4f/nvfp4; TAPE=/root/box/tape-rebuild.txt; TPEP="MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8 MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device MEMRA_DSV4_PREFILL_MOE=reference"
for at in 0 1; do
  wait_lock
  ( export $TPEP MEMRA_DSV4_ATTENTION_TP_GATE=$at; /root/box/gate.sh $R/tpep-gate-m-at$at $X/dsv4_tp_ep_gate $M $TAPE )
  echo "tpep-gate merged at$at $(grep -hE '^(DIGEST|PASS)|panicked|GATE_DONE' $R/tpep-gate-m-at$at/gate.log | cut -c1-220 | tr '\n' ' ')" >> $S
done
d0=$(grep -h '^DIGEST' $R/tpep-gate-m-at0/gate.log); d1=$(grep -h '^DIGEST' $R/tpep-gate-m-at1/gate.log)
[[ -n "$d0" && "$d0" == "$d1" ]] && echo "DIGEST_EQUAL merged yes" >> $S || echo "DIGEST_EQUAL merged no" >> $S
wait_lock
( export $TPEP MEMRA_DSV4_ATTENTION_TP_GATE=1 MEMRA_DSV4_SAMPLER=device
  /root/box/gate.sh $R/replay-at1b $X/dsv4_tp_ep_sampled_perf_gate $M $TAPE --full-token-replay )
echo "replay at1b $(grep -hE '^PASS|panicked|GATE_DONE' $R/replay-at1b/gate.log | cut -c1-200 | tr '\n' ' ') $(grep -c '^MEASURE' $R/replay-at1b/gate.log) measures" >> $S
echo V3F_DONE >> $S
