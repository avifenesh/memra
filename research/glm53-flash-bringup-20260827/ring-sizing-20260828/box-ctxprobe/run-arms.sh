#!/bin/bash
# Three arms at MEMRA_CTX=8192 (usable-vs-configured is the question), one env flag or one
# binary apart. MEMRA_PREFIX_CACHE_MB=0 pinned inside serve.sh on every arm.
#   A  MERGED-RING-ON   consolidated head f929dda914, ring default ON  (the fix under test)
#   B  MERGED-RING-OFF  same binary, MEMRA_DSA_INDEX_RING=0            (rollback seam)
#   C  UNFIXED-RING-ON  ~/memra-r2 binary: carries the ring UNFIXED (pre-fix guard message in
#      strings, f7ec ancestor, fix absent). NOT the pre-ring arm; the true pre-ring arm is
#      UNAVAILABLE on this box (every present binary postdates f7ec) and is stated as such
#      rather than substituted. C reproduces the RED regression in this session on this box.
set -u
R=$HOME/lane-ringsizing-vast-20260829
MERGED=${1:?merged-head binary}
UNFIXED=${2:?unfixed r2 binary}

run() {  # run <tag> <binary> <outfile> [extra env]
  local tag=$1 bin=$2 out=$3; shift 3
  echo "================================================================"
  date -u +%FT%TZ
  echo "=== ARM: $tag ==="
  bash $R/serve.sh "$tag" "$bin" MEMRA_CTX=8192 "$@" || { echo "ARM $tag: LOAD FAILED"; return 1; }
  python3 $R/ctxprobe.py "$tag" "$out"
}

{
  run MERGED-RING-ON  "$MERGED"  $R/ctx-A-merged-ring-on.json
  run MERGED-RING-OFF "$MERGED"  $R/ctx-B-merged-ring-off.json MEMRA_DSA_INDEX_RING=0
  run UNFIXED-RING-ON "$UNFIXED" $R/ctx-C-unfixed-ring-on.json
  echo "=== ARM: PRE-RING unavailable on this box: ~/memra-r2 strings census shows ring-flag=3,"
  echo "    pre-fix-msg=1 (f7ec IS its ancestor), so no present binary predates the ring. Stated,"
  echo "    not substituted. The 08-28 prior-box receipt (7312 tokens) remains the pre-ring reference."
  echo "================================================================"
  date -u +%FT%TZ
  echo "ARMSDONE"
} 2>&1 | tee $R/02-ctx-arms.txt
