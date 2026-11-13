#!/bin/bash
# b23d: r1-dig scratch (the layer gate's saved sets of the bisect arms and of r2, ~22 GB; the research binary).
# Kept: bins/r2 (r2's 121a binaries), out/ (logs, small), the overlays.
cd /home/avifenesh/.cache || exit 1
rm -rf /home/avifenesh/.cache/dsv41-speed/save-r1x-none /home/avifenesh/.cache/dsv41-speed/save-r1x-router \
  /home/avifenesh/.cache/dsv41-speed/save-r1x-comp /home/avifenesh/.cache/dsv41-speed/save-r2 \
  /home/avifenesh/.cache/dsv41-speed/bins/r1x /home/avifenesh/.cache/dsv41-speed/bench-gb10 \
  /home/avifenesh/.cache/dsv41-speed/out/r1x-tapes
df -h /home/avifenesh | tail -1
echo CLEANUP-DONE
