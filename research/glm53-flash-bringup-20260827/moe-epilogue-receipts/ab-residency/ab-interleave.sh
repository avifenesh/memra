#!/bin/bash
# Interleaved x5 A/B driver for MEMRA_MOE_FUSED_EPI on the residency config.
# The interleave unit is a BOOT (process-level env behind OnceLock; METHOD.txt deviation).
# V0 (=0, the validation boot) already ran; this alternates the remaining nine:
#   E1(=1) Z2(=0) E2(=1) Z3(=0) E3(=1) Z4(=0) E4(=1) Z5(=0) E5(=1)
# giving five boots per arm including V0. idle-check runs inside abres.sh before every boot.
set -u
for spec in Z2:0 E2:1 Z3:0 E3:1 Z4:0 E4:1 Z5:0 E5:1; do
  TAG=${spec%%:*}; EPI=${spec##*:}
  echo "=== BOOT $TAG EPI=$EPI $(date -u +%FT%TZ) ==="
  bash ~/abres.sh "$TAG" "$EPI" > ~/ab-$TAG.txt 2>&1
  rc=$?
  tail -1 ~/ab-$TAG.txt
  if [ $rc -ne 0 ]; then echo "BOOT $TAG FAILED rc=$rc — stopping the ladder"; exit $rc; fi
done
echo "AB-LADDER-DONE $(date -u +%FT%TZ)"
