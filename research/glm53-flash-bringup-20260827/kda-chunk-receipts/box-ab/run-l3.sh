#!/bin/bash
# Interleaved x5 fresh-boot TWO-ARM A/B for L3 (chunked KDA prefill scan), per
# memra research/glm53-flash-bringup-20260827/kda-chunk-receipts/README.md.
# BASE ENV = L2's arm C (L1 grouped prefill ON + MEMRA_BF16_MMV=1 + MEMRA_PP_BF16=1);
# the two arms are ONE flag apart: k0 = base C, k1 = base C + MEMRA_KDA_CHUNKED=1.
set -u
R=$HOME/l3-ab
cd $R
BASE=(MEMRA_MOE_GROUPED_PREFILL=1 MEMRA_BF16_MMV=1 MEMRA_PP_BF16=1)

engagement() { # engagement <tag>
  local log=$R/serve-$1.log
  echo "  engagement[$1]:" \
    "kda_flag=$(grep -o "kda-chunked\] flag=o[nf]*" $log | head -1)" \
    "kda_execute=$(grep -c "kda-chunked\] execute" $log)" \
    "kda_diff=$(grep -c "kda-diff" $log)" \
    "mmv_resident=$(grep -c "bf16-mmv\] RESIDENT" $log)" \
    "bf16tc_flag=$(grep -o "bf16-tc\] flag=o[nf]*" $log | head -1)" \
    "bf16tc_engaged=$(grep -c "bf16-tc\] ENGAGED" $log)" \
    "gpf_flag=$(grep -o "moe-grouped-prefill\] flag=o[nf]*" $log | head -1)" \
    "gpf_execute=$(grep -c "moe-grouped-prefill\] execute" $log)"
  grep -o "kda-chunked\] execute t=[0-9]* nc=[0-9]* c=[0-9]*" $log | sort | uniq -c | sed "s/^/    /"
}

boot_and_probe() { # boot_and_probe <tag> [extra env...]
  local tag=$1; shift
  echo "================ $(date -u +%FT%TZ) BOOT $tag ================"
  bash $R/serve.sh "$tag" "${BASE[@]}" "$@" || { echo "BOOT $tag FAILED"; return 1; }
  python3 $R/probe.py "$tag" $R/rows-$tag.json
  engagement "$tag"
}

{
  echo "L3 AB START $(date -u +%FT%TZ)"
  cd $HOME/memra && git log -1 --format="engine: %H %s" && cd $R
  BIN=$HOME/memra/target/release/memra-server
  sha256sum $BIN
  sha256sum $R/prompts.json
  # rebuild-after-checkout law: the binary must be NEWER than every source it was built from
  NEWEST_SRC=$(find $HOME/memra/crates $HOME/memra/Cargo.toml -name "*.rs" -o -name "*.cu" -o -name "*.toml" 2>/dev/null | xargs stat -c %Y 2>/dev/null | sort -n | tail -1)
  BIN_T=$(stat -c %Y $BIN)
  echo "binary-newer-than-sources: bin=$BIN_T newest_src=$NEWEST_SRC $([ "$BIN_T" -gt "$NEWEST_SRC" ] && echo PASS || echo FAIL)"

  # SMOKE (not counted): base C shape, KDA flag OFF; verifies pipeline + announce line.
  boot_and_probe smoke

  # DIFF boot (not a perf row: the oracle syncs per KDA call and keeps the sequential path):
  # real-weights band receipt across all three prompt widths + the sampled twin.
  boot_and_probe diff MEMRA_KDA_DIFF=1
  grep "kda-diff" $R/serve-diff.log > $R/kda-diff-lines.txt
  echo "  diff lines banked: $(wc -l < $R/kda-diff-lines.txt)"

  for i in 1 2 3 4 5; do
    boot_and_probe k0$i
    boot_and_probe k1$i MEMRA_KDA_CHUNKED=1
  done
  echo "L3 AB DONE $(date -u +%FT%TZ)"
} 2>&1 | tee -a $R/l3-run.log
