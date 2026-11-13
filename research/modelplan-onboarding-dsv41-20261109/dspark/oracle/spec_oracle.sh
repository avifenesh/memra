#!/bin/bash
# usage: spec_oracle.sh OUT_DIR [spec_oracle args...]   (inside a lease: CUDA_VISIBLE_DEVICES is the leased card)
# The drafting lane's copy of the vendor oracle with the DSpark tapes (spec_oracle.py); the shared oracle/ is untouched.
set -euo pipefail
OUT=$1; shift
D=${LANE:?LANE names the working directory on the run box (checkpoint, oracle, dec ids, spec/, target-spec/)}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
exec nice -n 5 ${TORCH_PY:?a python with torch} -I "$D/spec/oracle/spec_oracle.py" \
  "$D/DeepSeek-V4.1-Flash-1017" "$D/ref" "$D/oracle/v2_cases.json" "$OUT" --threads 8 "$@"
