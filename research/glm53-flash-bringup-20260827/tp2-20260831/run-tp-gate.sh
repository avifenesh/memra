#!/usr/bin/env bash
# glm5-tp-gate runner (lane/glm5-tp2). One invocation = the full arm matrix (the binary
# owns its own env discipline: it clears/sets MEMRA_GLM5_TP / _GATE_RED / _GATE_SAME_DEV
# per arm internally, so no knob matrix is needed here).
#
# Rig law: exactness only. No timing number is read out of this script.
set -u
BIN=${BIN:-./target/release/glm5-tp-gate}
OUT=${OUT:-research/glm53-flash-bringup-20260827/tp2-20260831/stage34-gates}
P=${P:-16}
N=${N:-12}

echo "########## glm5-tp-gate P=$P N=$N ##########"
flock /tmp/memra-5090.lock env NVIDIA_TF32_OVERRIDE=0 \
  timeout 1800 "$BIN" "$P" "$N" 2>&1 | grep -v '^\[loader-law\]' \
  | tee "$OUT/01-tp-gate-p${P}-n${N}.log"
rc=${PIPESTATUS[0]}
echo "exit=$rc" | tee -a "$OUT/01-tp-gate-p${P}-n${N}.log"
exit "$rc"
