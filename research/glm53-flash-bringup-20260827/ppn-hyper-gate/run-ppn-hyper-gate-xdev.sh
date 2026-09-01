#!/usr/bin/env bash
# glm5-hyper-ppn-gate CROSS-DEVICE matrix — the arms the one-card rig could not run.
#
# This is the block that was scripted and left UNRUN at the foot of run-ppn-hyper-gate.sh.
# It is the difference between "the door is proven as a SPLIT" and "the door is proven as a
# PLACEMENT": peer transport, sharded weight load, and cross-device per-stage cache placement
# are exercised here and nowhere else.
#
# Runs on a two-card box. One placement per invocation (`PpNRt` freezes its stage/device map at
# first build). Exactness only — no timing number is read out of this script.
set -u
BIN=${BIN:-./target/release/glm5-hyper-ppn-gate}
OUT=${OUT:-.}
fails=0

run() { # run <logname> [ENV=V ...] -- <args...>
  local log="$1"; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  echo "########## $log :: ${envs[*]:-(no env)} $* ##########"
  env NVIDIA_TF32_OVERRIDE=0 "${envs[@]}" timeout 1800 "$BIN" "$@" 2>&1 \
    | grep -v '^\[loader-law\]' | tee "$OUT/$log"
  local rc=${PIPESTATUS[0]}
  echo "exit=$rc" | tee -a "$OUT/$log"
  [ "$rc" -eq 0 ] || fails=$((fails + 1))
}

# --- the four arms the coordinator named ---
run 20-xdev-n2-dev01.log         MEMRA_PP_DEVICES=0,1                        -- 2 6 8
run 21-xdev-n2-dev01-shard0.log  MEMRA_PP_DEVICES=0,1 MEMRA_PP_SHARD=0       -- 2 6 8
run 23-xdev-n4-dev0101.log       MEMRA_PP_DEVICES=0,1,0,1                    -- 4 6 8

# HOST BOUNCE, and the shape of it is a finding rather than a knob. `pp.rs` refuses
# MEMRA_PP_HOST_BOUNCE=1 unless the primary engine sits on the LAST/head stage — otherwise the
# returned logits and hidden state stay peer reads, which is the thing the bounce arm exists to
# avoid. This gate's primary IS devices[0], so a two-stage 0,1 placement can never satisfy it:
# the requested `MEMRA_PP_DEVICES=0,1 x HOST_BOUNCE=1` arm is structurally refused BY DESIGN, not
# broken. Both halves are recorded: the placement that satisfies the guard, and the refusal.
run 22-xdev-n4-dev0110-bounce.log MEMRA_PP_DEVICES=0,1,1,0 MEMRA_PP_HOST_BOUNCE=1 -- 4 6 8

# --- three more this lane adds, because they are cheap and they probe real assumptions ---
# reversed placement: the head lands on dev1's stage while the primary engine is dev1 too,
# which is the topology the gemmaaux note says serving picks (primary == head stage).
run 24-xdev-n2-dev10.log         MEMRA_PP_DEVICES=1,0                        -- 2 6 8
# asymmetric cut across the device boundary: one card owns 3 of 4 layers.
run 25-xdev-n2-dev01-split1.log  MEMRA_PP_DEVICES=0,1 MEMRA_PP_SPLITS=1      -- 2 6 8
# longer prompt + a 40-step tape, cross-device.
run 26-xdev-n2-dev01-longer.log  MEMRA_PP_DEVICES=0,1                        -- 2 16 24

# The refusal itself, banked as a receipt: a guard that has never been seen to fire is a guard
# nobody has tested. Expected to exit NON-ZERO with the primary-on-last-stage message.
echo "########## 22b-xdev-n2-dev01-bounce-EXPECTED-REFUSAL.log ##########"
env NVIDIA_TF32_OVERRIDE=0 MEMRA_PP_DEVICES=0,1 MEMRA_PP_HOST_BOUNCE=1 \
  timeout 1800 "$BIN" 2 6 8 2>&1 | grep -v '^\[loader-law\]' \
  | tee "$OUT/22b-xdev-n2-dev01-bounce-EXPECTED-REFUSAL.log"
rc=${PIPESTATUS[0]}
echo "exit=$rc (NON-ZERO IS THE EXPECTED RESULT: the primary-on-last-stage guard must fire)" \
  | tee -a "$OUT/22b-xdev-n2-dev01-bounce-EXPECTED-REFUSAL.log"
if [ "$rc" -eq 0 ]; then
  echo "UNEXPECTED: the host-bounce guard did NOT fire on a 0,1 placement"
  fails=$((fails + 1))
fi

echo "=========================================================="
if [ "$fails" -eq 0 ]; then
  echo "glm5-hyper-ppn-gate CROSS-DEVICE matrix: ALL ARMS PASS"
else
  echo "glm5-hyper-ppn-gate CROSS-DEVICE matrix: $fails ARM(S) FAILED"
fi
exit "$fails"
