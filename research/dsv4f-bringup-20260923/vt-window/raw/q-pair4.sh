#!/usr/bin/env bash
# q-pair4.sh (2x RTX PRO 6000 WS pod), after q-pair2.sh: DSpark verify-depth and confidence-window
# sweep on main 25bbb91f5 (target-main5 from q-pair2). Served DSpark cells, one boot per row,
# two balanced passes: depth 6 (the default, unbounded = the drafter's 5 drafts + 1), 5, 4, 3,
# and MEMRA_DSV4_VT=slot@0.5. Then one MEMRA_DSV4_ROUND_PROFILE=1 boot for the round anatomy.
while ! grep -q P2P_DONE /root/rcpt/q-pair3b.summary 2>/dev/null; do sleep 30; done
set -u
S=/root/rcpt/q-pair4.summary; R=/root/rcpt/depth; mkdir -p $R
wait_lock() { while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done; }
B=/root/lane/target-main5/release/memra-server
echo "base $(sha256sum $B | cut -c1-16)" >> $S
arm_env() {
  case $1 in
    d6) echo "MEMRA_DSV4_DRAFTER=dspark" ;;
    d5) echo "MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_SPEC_DEPTH=5" ;;
    d4) echo "MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_SPEC_DEPTH=4" ;;
    d3) echo "MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_SPEC_DEPTH=3" ;;
    vt) echo "MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_VT=slot@0.5" ;;
    prof) echo "MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_ROUND_PROFILE=1" ;;
  esac
}
i=0
for arm in d6 d4 d3 d5 vt vt d5 d3 d4 d6 prof; do
  i=$((i+1)); wait_lock; d=$R/r$i-$arm
  # shellcheck disable=SC2046
  /root/box/cell.sh $d $B /root/box/cells-spec.txt $(arm_env $arm) > $d.out 2>&1
  echo "depth r$i $arm rc=$? $(grep -hE 'CELL ' $d/controller.log 2>/dev/null | grep -v warmup | cut -c1-170 | tr '\n' ' ') $(grep -h FATAL $d/serve.log 2>/dev/null | head -1 | cut -c1-160)" >> $S
done
echo PAIR4_DONE >> $S
