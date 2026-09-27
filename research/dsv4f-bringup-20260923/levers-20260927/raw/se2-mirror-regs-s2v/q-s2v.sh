#!/usr/bin/env bash
# q-s2v.sh (second SE pair): the fused MoE mirror, four programs. M = main (target-small2),
# H = the published h mirror (target-hm, q-s2t), M2 = main with the register-held, exact-reciprocal
# mirror ($1), H2 = H with it ($2). Builds share one scratch target; each variant's binaries are
# copied out. Long gate hash on M2 and H2, then replay ms/token M M2 H H2 H2 H M2 M.
set -u
M2SHA=$1; H2SHA=$2
S=/root/rcpt/q-s2v.summary; R=/root/rcpt/mirror-regs-s2v; mkdir -p $R
until grep -q S2T_DONE /root/rcpt/q-s2t.summary 2>/dev/null; do sleep 60; done
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
M=/data/dsv4f/nvfp4; T=/root/box/tape-rebuild.txt
for v in M:target-small2 H:target-hm; do IFS=: read n tg <<< "$v"
  mkdir -p /root/lane/bin-$n; cp /root/lane/$tg/release/dsv4_tp_replay_long_gate /root/lane/$tg/release/memra-server /root/lane/bin-$n/; done
rm -rf /root/lane/target-small2
[[ $(df --output=avail -BG / | tail -1 | tr -dc 0-9) -ge 30 ]] || { echo "DISK_SHORT" >> $S; exit 1; }
cd /root/lane/memra && git fetch -q origin lane/dsv4-moe-mirror-regs-20260927 lane/dsv4-moe-hmirror-20260927
git worktree add -f /root/lane/t-m2 $M2SHA > /dev/null 2>&1; git -C /root/lane/t-m2 checkout -q --detach $M2SHA
git -C /root/lane/t-hm checkout -q --detach $H2SHA
[[ "$(git -C /root/lane/t-m2 rev-parse HEAD)" == "$M2SHA" && "$(git -C /root/lane/t-hm rev-parse HEAD)" == "$H2SHA" ]] || { echo "TREE_MISMATCH" >> $S; exit 1; }
for v in H2:t-hm M2:t-m2; do IFS=: read n tr <<< "$v"
  bash /root/box/build.sh /root/lane/$tr /root/lane/target-hm v$n
  mkdir -p /root/lane/bin-$n; cp /root/lane/target-hm/release/dsv4_tp_replay_long_gate /root/lane/target-hm/release/memra-server /root/lane/bin-$n/
  echo "build $n $(git -C /root/lane/$tr rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-v$n.log | tr '\n' ' ')" >> $S
done
sha256sum /root/lane/bin-*/dsv4_tp_replay_long_gate /root/lane/bin-*/memra-server > $R/binaries.sha256
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
locked() { local d=$1; shift; mkdir -p $d; wait_lock
  ( exec 9>/tmp/memra-gpu.lock; flock -n 9 || exit 75; env NVIDIA_TF32_OVERRIDE=0 timeout 3600 "$@" 9>&- > $d/gate.log 2>&1 ); }
i=0
for a in M M2 H H2 H2 H M2 M; do
  i=$((i+1)); d=$R/long-t$i-$a
  locked $d /root/lane/bin-$a/dsv4_tp_replay_long_gate $M $T 304
  echo "long t$i $a rc=$? $(grep -hoE 'PROGRAM_SHA256 [0-9a-f]{16}|PASS: 304|FAILED[^ ]*|panicked' $d/gate.log | sort -u | tr '\n' ' ') $(grep -hoE 'replay_ms_per_token=[0-9.]+' $d/gate.log | tr '\n' ' ')" >> $S
done
echo "S2V_DONE $M2SHA $H2SHA" >> $S
