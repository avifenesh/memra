#!/usr/bin/env bash
# Re-run of the KDA-only sub-shard arm under the FIXED identity rule (the diet announce is
# demanded only when the boot preflight reports moe_ep>0; a 0-2 KDA shard legitimately
# dietes nothing). New run tag keeps the first boot banked, and the two boots byte-compare
# as a determinism receipt on top.
set -uo pipefail
OUT=/root/out-tpd
bash $OUT/tpd_arm.sh dietsub tape $OUT/prompts-tiny tinykda2 \
  BOXP_FORCE_DIR=$OUT/plain1-tape-tinyref MEMRA_GLM5_TP=0-2@0,1 BOXP_MAX_NEW=32
echo "C1FIX_RC=$?"
python3 $OUT/compare.py $OUT/plain1-tape-tinyref $OUT/dietsub-tape-tinykda2 \
  > $OUT/analysis/cmp-tinykda2-vs-plain.txt 2>&1
echo "cmp_vs_plain_rc=$?" | tee -a $OUT/analysis/cmp-tinykda2-vs-plain.txt
python3 $OUT/compare.py $OUT/dietsub-tape-tinykda $OUT/dietsub-tape-tinykda2 \
  > $OUT/analysis/cmp-tinykda-boot-determinism.txt 2>&1
echo "boot_determinism_rc=$?" | tee -a $OUT/analysis/cmp-tinykda-boot-determinism.txt
tail -2 $OUT/analysis/cmp-tinykda2-vs-plain.txt
tail -2 $OUT/analysis/cmp-tinykda-boot-determinism.txt
cat $OUT/logs/probe-dietsub-tape-tinykda2.identity
echo C1FIX_DONE
