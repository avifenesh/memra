#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: interleaved run-gen prime A/B (MEMRA_PP_PRIME=1, the
# served prime path) on one record prompt, arms selected by env, last-row logits byte-compared.
# usage: pp_ab.sh OUT NREC "ARM ARM ..."   arms: M (main binary) or a lane arm from arm_env below
#   LANE=<lane bin dir>  MAIN=<main bin dir>
set -u
D=$HOME/.cache/memra-gb10-next
OUT=$1; NREC=$2; ORDER=$3
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
mkdir -p "$OUT"
python3 - "$OUT/prompt-$NREC.txt" "$NREC" <<'EOF'
import sys
WORDS = "alpha river copper lantern orbit meadow cipher harbor violet granite ember falcon".split()
n = int(sys.argv[2])
lines = [f"Record {i}: " + " ".join(WORDS[(i * 7 + j * 3) % 12] for j in range(14)) + f" {i % 97}." for i in range(n)]
open(sys.argv[1], "w").write("\n".join(lines))
EOF
arm_env() {
  case $1 in
    M) echo "" ;;
    B) echo "MEMRA_GDN_DB_GQA=0 MEMRA_GDN_HFAST=0 MEMRA_GDN_W32=1" ;;
    D) echo "MEMRA_GDN_HFAST=0 MEMRA_GDN_W32=1" ;;
    W) echo "MEMRA_GDN_HFAST=0" ;;
    F) echo "" ;;
    N0) echo "MEMRA_PRIME_NORM_FUSE=0" ;;
    N1) echo "MEMRA_PRIME_NORM_FUSE=1" ;;
    N2) echo "MEMRA_PRIME_NORM_FUSE=2" ;;
    N3) echo "MEMRA_PRIME_NORM_FUSE=3" ;;
    Z) echo "MEMRA_GDN_DB_GQA=0 MEMRA_GDN_HFAST=0 MEMRA_GDN_W32=1 MEMRA_PRIME_NORM_FUSE=0 MEMRA_DP4A_TILED=0 MEMRA_Q8_GROUP=0" ;;
    G) echo "MEMRA_PRIME_NORM_FUSE=0 MEMRA_DP4A_TILED=0 MEMRA_Q8_GROUP=0" ;;
    G1) echo "MEMRA_PRIME_NORM_FUSE=1 MEMRA_DP4A_TILED=0 MEMRA_Q8_GROUP=0" ;;
    G3) echo "MEMRA_PRIME_NORM_FUSE=3 MEMRA_DP4A_TILED=0 MEMRA_Q8_GROUP=0" ;;
    GQ) echo "MEMRA_DP4A_TILED=0" ;;
    A) echo "" ;;
    *) echo "${ARM_ENV_EXTRA:-}" ;;
  esac
}
i=0
for arm in $ORDER; do
  i=$((i + 1))
  bin=$LANE; [ "$arm" = M ] && bin=$MAIN
  tag=$(printf "%02d-%s" "$i" "$arm")
  read -r -a envs <<< "$(arm_env "$arm")"
  echo "$tag bin=$(head -1 "$bin/COMMIT") env=${envs[*]:-none} start $(date -Is)" >> "$OUT/runs.txt"
  env "${envs[@]}" MEMRA_PP_ONLY=1 MEMRA_PP_PRIME=1 MEMRA_PP_REPS=3 MEMRA_PP_WARMUP=0 \
    MEMRA_PP_LOGITS="$OUT/$tag.logits" MEMRA_PROMPT_FILE="$OUT/prompt-$NREC.txt" \
    "$bin/run-gen" "$G8" > "$OUT/$tag.log" 2>&1
  rc=$?
  med=$(grep -o "pp-only MEDIAN: .*" "$OUT/$tag.log")
  sha=$(sha256sum "$OUT/$tag.logits" 2>/dev/null | cut -c1-16)
  echo "$tag rc=$rc logits=$sha $med" | tee -a "$OUT/summary.txt"
done
echo PP-AB-DONE >> "$OUT/summary.txt"
