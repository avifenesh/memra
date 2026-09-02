#!/bin/bash
# Coverage arm for the one changed consumer no other arm reaches: mtp_reference_weights(),
# which real_gate takes under --draft-gate. It reads the MTP bank off the mmap for the HOST
# reference twin, then from_loaded_checkpoint_dual reads it again for the device model.
# Cap 230G, not 150G: --draft-gate clones the whole trunk f32 weight set for the twin
# (~20 GB on top of the load peak), which is its own pre-existing cost and would confuse a
# 150 GiB result. Shared measurement lock, card 1, no fallback.
set -uo pipefail
lbl=new-draftgate
bin=/root/realgate/bin/qwen4exp_real_gate.loader
out=/root/realgate/loaderout
log="$out/$lbl.log"
{
  printf "# lane\tq4e-loader-stream-20260901 (memra issue #48)\n"
  printf "# label\t%s\n# binary\t%s\n# binary_sha256\t%s\n" "$lbl" "$bin" "$(sha256sum "$bin" | cut -d" " -f1)"
  printf "# arm\t--draft-gate (covers mtp_reference_weights)\n"
  printf "# cap\tMemoryMax=230G MemorySwapMax=0\n# card\tCUDA_VISIBLE_DEVICES=1\n"
  printf "# started\t%s\n" "$(date -u +%FT%TZ)"
} > "$log"
exec 9>>/tmp/q48fn-measure.lock
flock -s 9
printf "# lock\tacquired -s after %ss of waiting\n" "$SECONDS" >> "$log"
env CUDA_VISIBLE_DEVICES=1 MEMRA_Q4E_SEAMS=idxsel \
  systemd-run --scope --unit="q4e-$lbl" -p MemoryMax=230G -p MemorySwapMax=0 \
    /usr/bin/time -v "$bin" /root/data/q48fn-yarn1m "$out" --label "$lbl" --mtp --draft-gate \
      --goldens /root/realgate/dump --prompts /root/realgate/shapes/thinkon-prompts.tsv \
  >> "$log" 2>&1
rc=$?
exec 9>&-
printf "# rc\t%s\n# finished\t%s\n" "$rc" "$(date -u +%FT%TZ)" >> "$log"
printf "[%s] %s rc=%s\n" "$(date -u +%FT%TZ)" "$lbl" "$rc" >> "$out/CELL.log"
