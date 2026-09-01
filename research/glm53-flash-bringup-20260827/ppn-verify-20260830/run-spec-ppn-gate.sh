#!/usr/bin/env bash
# glm5-spec-ppn-gate knob matrix (lane/glm5-ppn-verify).
#
# One placement per invocation: `PpNRt` freezes its stage/device map at first build, so the
# matrix is driven by re-invoking the binary, never by looping inside it.
#
# RIG SCOPE (one card): every arm here is same-device. That still exercises the whole split
# walk — per-stage streams, contexts, the boundary transport, per-stage cache/ckpt seams,
# the per-stage rollback and the last-stage MTP chain. It does NOT exercise cross-device
# peer transport or weight sharding; those need MEMRA_PP_DEVICES on a multi-card box and
# are listed at the bottom, unrun — the lane's named final gate.
#
# Rig law: exactness only. No timing number is read out of this script.
set -u
BIN=${BIN:-./target/release/glm5-spec-ppn-gate}
OUT=${OUT:-research/glm53-flash-bringup-20260827/ppn-verify-20260830}
fails=0

run() { # run <logname> [ENV=V ...] -- <args...>
  local log="$1"; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  echo "########## $log :: ${envs[*]:-(no env)} $* ##########"
  flock /tmp/memra-5090.lock env NVIDIA_TF32_OVERRIDE=0 "${envs[@]}" \
    timeout 1800 "$BIN" "$@" 2>&1 | grep -v '^\[loader-law\]' | tee "$OUT/$log"
  local rc=${PIPESTATUS[0]}
  echo "exit=$rc" | tee -a "$OUT/$log"
  [ "$rc" -eq 0 ] || fails=$((fails + 1))
}

# stages=2: the SPLITS=24 serving class. Even split [0,2,4] separates KDA/MLA state
# classes; splits=1 and 3 skew the fence to put a lone layer on one side.
run 10-n2-even.log                                  -- 2 24 20
run 11-n2-split1.log     MEMRA_PP_SPLITS=1          -- 2 24 20
run 12-n2-split3.log     MEMRA_PP_SPLITS=3          -- 2 24 20
run 13-n2-streams0.log   MEMRA_PP_STREAMS=0         -- 2 24 20
run 14-n2-overlap0.log   MEMRA_PP_OVERLAP=0         -- 2 24 20
# stages=3: the SPLITS=15,30 3-card serving shape's class. Default even = [0,1,2,4];
# the asym arm gives the middle stage both state classes.
run 16-n3-even.log                                  -- 3 24 20
run 17-n3-asym.log       MEMRA_PP_SPLITS=1,3        -- 3 24 20
run 18-n3-streams0.log   MEMRA_PP_STREAMS=0         -- 3 24 20

echo "=========================================================="
if [ "$fails" -eq 0 ]; then
  echo "glm5-spec-ppn-gate matrix: ALL ARMS PASS"
else
  echo "glm5-spec-ppn-gate matrix: $fails ARM(S) FAILED"
fi
exit "$fails"

# NOT RUN HERE — needs a multi-card box (coordinate through the lane owner; do NOT ssh):
#   MEMRA_PP_DEVICES=0,1                    stages=2 cross-device peer transport + sharding
#   MEMRA_PP_DEVICES=0,1 MEMRA_PP_SHARD=0   bring-up placement (weights all-primary)
#   MEMRA_PP_DEVICES=0,1,2                  stages=3, the 3-card serving shape's class
#   MEMRA_PP_DEVICES=0,1 MEMRA_PP_HOST_BOUNCE=1 (requires primary on the LAST stage)
