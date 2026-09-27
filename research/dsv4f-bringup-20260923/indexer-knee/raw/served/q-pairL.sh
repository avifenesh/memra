#!/usr/bin/env bash
# q-pairL.sh (2x RTX PRO 6000 WS pod): the long-context A/B of q-pair.sh, rerun. The first run is
# void (long-void-408): every 32k/64k request hit the server's default 90 s timeout_ms ceiling
# before its first token. MEMRA_TIMEOUT_MS_MAX=900000 is the documented measurement-cell override
# for offline long-prefill cells (docs/FLAGS.md). base = main 6978f5fac, idx = lane 798e12674.
# Order l b b l l b b l, plain then DSpark; writes PAIR_DONE so q-pair2.sh starts after it.
set -u
S=/root/rcpt/q-pair.summary; R=/root/rcpt
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
bin() { [[ $1 == base ]] && echo /root/lane/target-main/release/memra-server || echo /root/lane/target-$1/release/memra-server; }
row() {  # row <dir> <arm> <cells> [env...]
  local dir=$1 arm=$2 cells=$3; shift 3; wait_lock
  /root/box/cell.sh $dir $(bin $arm) $cells MEMRA_TIMEOUT_MS_MAX=900000 "$@" > $dir.out 2>&1
  local rc=$?; echo "${dir#$R/} rc=$rc $(grep -hE 'CELL ' $dir/controller.log 2>/dev/null | grep -v warmup | cut -c1-240 | tr '\n' ' ')" >> $S
}
mkdir -p $R/long/plain $R/long/spec
i=0; for arm in idx base base idx idx base base idx; do i=$((i+1)); row $R/long/plain/r$i-$arm $arm /root/box/cells-long.txt; done
i=0; for arm in idx base base idx idx base base idx; do i=$((i+1)); row $R/long/spec/r$i-$arm $arm /root/box/cells-long.txt MEMRA_DSV4_DRAFTER=dspark; done
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,clocks.sm --format=csv,noheader >> $S
echo PAIR_DONE >> $S
