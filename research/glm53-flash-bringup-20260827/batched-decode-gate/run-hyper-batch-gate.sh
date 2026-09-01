#!/usr/bin/env bash
# glm5-hyper-batch-gate knob matrix.
#
# The batched hc walk vs each session's ISOLATED serial walk, bit for bit. Staggered
# depths, B=1 class pin, device-sample greedy — see the gate's header for the arm map.
# `stages>1` arms run BOTH reference and batched walks with the ppN door open (PpNRt
# freezes its placement at first build, so one placement per invocation).
#
# RIG SCOPE (one card): same-device only; cross-device batched-hyper arms need
# MEMRA_PP_DEVICES=0,1 on a multi-card box (ask the owner for box time).
#
# Rig law: exactness only. No timing number is read out of this script.
set -u
BIN=${BIN:-./target/release/glm5-hyper-batch-gate}
OUT=${OUT:-research/glm53-flash-bringup-20260827/batched-decode-gate}
fails=0

run() { # run <logname> [ENV=V ...] -- <args...>
  local log="$1"; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  echo "########## $log :: ${envs[*]:-(no env)} $* ##########"
  flock /tmp/memra-5090.lock env NVIDIA_TF32_OVERRIDE=0 "${envs[@]}" \
    timeout 900 "$BIN" "$@" 2>&1 | grep -v '^\[loader-law\]' | tee "$OUT/$log"
  local rc=${PIPESTATUS[0]}
  echo "exit=$rc" | tee -a "$OUT/$log"
  [ "$rc" -eq 0 ] || fails=$((fails + 1))
}

#                                                      args: B P N stages
run 10-b3-default.log                                     -- 3 5 8 1
run 11-b8-wide.log                                        -- 8 5 8 1
run 12-b2-longer.log                                      -- 2 12 24 1
run 13-b3-ppn2.log                                        -- 3 5 8 2
run 14-b3-ppn2-streams0.log  MEMRA_PP_STREAMS=0           -- 3 5 8 2
run 15-b3-ppn4.log                                        -- 3 5 8 4
run 16-b8-ppn2.log                                        -- 8 5 8 2
# The derived cap (hyper_batch_cap = PRIME_MIN_T-1 = 15, the shexp decode-exact knee):
run 17-b12.log                                            -- 12 5 8 1
run 18-b15-cap.log                                        -- 15 5 8 1
run 19-b15-ppn2.log                                       -- 15 5 8 2
# B=16 (over-cap), 31-KNEE-b16-forced, and the B=15 mutation re-proof are banked
# separately (30/31/92 logs): 30 must STOP on the engine's named refusal, 31 forces the
# cap up one via a temporary banked edit and must go RED at the first tick.

echo "=========================================================="
if [ "$fails" -eq 0 ]; then
  echo "glm5-hyper-batch-gate matrix: ALL ARMS PASS"
else
  echo "glm5-hyper-batch-gate matrix: $fails ARM(S) FAILED"
fi
exit "$fails"

# NOT RUN HERE — needs a second card (ask the owner for box time):
#   MEMRA_PP_DEVICES=0,1 (stages=2)       cross-device batched hyper decode
#   MEMRA_PP_DEVICES=0,1 MEMRA_PP_SHARD=0 bring-up placement
