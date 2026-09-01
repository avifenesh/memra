#!/bin/bash
# Interleaved x5 fresh-boot A/B for MEMRA_MOE_GROUPED_PREFILL, per AB-PLAN.md.
# usage: run-ab.sh            (smoke boot + 5 OFF/ON pairs)
set -u
R=$HOME/gpf-ab
cd $R

engagement() { # engagement <tag>
  local log=$R/serve-$1.log
  echo "  engagement[$1]: announce=$(grep -c 'moe-grouped-prefill\] flag=' $log)" \
       "flagline=$(grep -o 'moe-grouped-prefill\] flag=o[nf]*' $log | head -1)" \
       "execute_lines=$(grep -c 'moe-grouped-prefill\] execute' $log)" \
       "prof_lines=$(grep -c 'moe-grouped-prefill-prof' $log)"
}

boot_and_probe() { # boot_and_probe <tag> [extra env...]
  local tag=$1; shift
  echo "================ $(date -u +%FT%TZ) BOOT $tag ================"
  bash $R/serve.sh "$tag" "$@" || { echo "BOOT $tag FAILED"; return 1; }
  python3 $R/probe.py "$tag" $R/rows-$tag.json
  engagement "$tag"
}

{
  echo "AB START $(date -u +%FT%TZ)"
  cd $HOME/memra && git log -1 --format="engine: %H %s" && cd $R
  sha256sum $HOME/memra/target/release/memra-server
  sha256sum $R/prompts.json

  # SMOKE (not counted): OFF-shape boot to verify the pipeline + prompt token counts.
  boot_and_probe smoke MEMRA_MOE_GROUPED_PREFILL=0

  for i in 1 2 3 4 5; do
    boot_and_probe off$i MEMRA_MOE_GROUPED_PREFILL=0
    boot_and_probe on$i  MEMRA_MOE_GROUPED_PREFILL=1
  done
  echo "AB DONE $(date -u +%FT%TZ)"
} 2>&1 | tee -a $R/ab-run.log
