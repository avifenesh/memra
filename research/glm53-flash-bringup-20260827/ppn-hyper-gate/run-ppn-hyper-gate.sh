#!/usr/bin/env bash
# glm5-hyper-ppn-gate knob matrix.
#
# One placement per invocation: `PpNRt` freezes its stage/device map at first build, so the
# matrix is driven by re-invoking the binary, never by looping inside it.
#
# RIG SCOPE (one card): every arm here is same-device. That still exercises the whole split
# walk — per-stage streams, contexts, the boundary transport, per-stage caches and the shared
# trunk exits. It does NOT exercise cross-device peer transport or weight sharding; those need
# MEMRA_PP_DEVICES=0,1 on a multi-card box and are listed at the bottom, unrun.
#
# Rig law: exactness only. No timing number is read out of this script.
set -u
BIN=${BIN:-./target/release/glm5-hyper-ppn-gate}
OUT=${OUT:-research/glm53-flash-bringup-20260827/ppn-hyper-gate}
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

run 10-n2-even.log                                            -- 2 6 8
run 11-n2-split1.log       MEMRA_PP_SPLITS=1                  -- 2 6 8
run 12-n2-split3.log       MEMRA_PP_SPLITS=3                  -- 2 6 8
run 13-n2-streams0.log     MEMRA_PP_STREAMS=0                 -- 2 6 8
run 14-n2-overlap0.log     MEMRA_PP_OVERLAP=0                 -- 2 6 8
run 15-n2-shard0.log       MEMRA_PP_SHARD=0                   -- 2 6 8
run 16-n3-asym.log         MEMRA_PP_SPLITS=1,3                -- 3 6 8
run 17-n4-even.log                                            -- 4 6 8
run 18-n4-streams0.log     MEMRA_PP_STREAMS=0                 -- 4 6 8
run 19-n2-longer.log                                          -- 2 16 24

echo "=========================================================="
if [ "$fails" -eq 0 ]; then
  echo "glm5-hyper-ppn-gate matrix: ALL ARMS PASS"
else
  echo "glm5-hyper-ppn-gate matrix: $fails ARM(S) FAILED"
fi
exit "$fails"

# NOT RUN HERE — needs a second card (message the lane owner for box time):
#   MEMRA_PP_DEVICES=0,1                  cross-device peer transport + weight sharding
#   MEMRA_PP_DEVICES=0,1 MEMRA_PP_SHARD=0 bring-up placement (weights all-primary)
#   MEMRA_PP_DEVICES=0,1 MEMRA_PP_HOST_BOUNCE=1
#   MEMRA_PP_DEVICES=0,1,0,1 (N=4)
