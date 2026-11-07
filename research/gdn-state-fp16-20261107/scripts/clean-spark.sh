#!/usr/bin/env bash
# Runs ON a GB10 unit: remove this lane's scratch (sources, target dirs, binaries, run logs).
set -u
cd "$HOME" || exit 1
nice -n 19 ionice -c3 rm -rf /home/avifenesh/.cache/memra-gdnfp16
ls -d /home/avifenesh/.cache/memra-gdnfp16 2>/dev/null || echo "memra-gdnfp16 scratch removed"
