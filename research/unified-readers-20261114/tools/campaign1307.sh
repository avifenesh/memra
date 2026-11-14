#!/usr/bin/env bash
# Runs ON a GB10 host inside a sparklock lease (memra#1307): the instrument probe and the served
# cells, one after another, into ~/.cache/memra-1307/runs/CAMPAIGN/.
#   probe: probe_alloc.py with the page cache dropped (quiet), filled (full), and filled with two
#          readers kept going (live).
#   mtp:   the draft-state bracket, Qwen3.8-27B MTP: B and T alternating with the page cache live
#          (01-B 02-T 03-T 04-B), then one quiet cell per arm (05-B 06-T).
#   mtpboot: the same, B and T alternating (21-B 22-T 23-T 24-B), with the readers going from before
#          the boot, through the spec calibration's captures that set the draft-state high-water.
#   dflash: the pooled DFlash2 verify width, cohorts of two: B and T alternating, live (11-B 12-T
#          13-T 14-B), then one quiet cell per arm (15-B 16-T).
#   dflash4: four requests at once under the served spec gate (31-B 32-T 33-T 34-B live, 35-B 36-T
#          quiet). The gate admits speculation for two sessions at most (MEMRA_SPEC_GATE_LOW=2), so
#          these decode plain and plan no pooled forward: they show that, and nothing about the width.
#   dflashagain: one more live pair cell per arm (17-T, 18-B), after a cell of the first set failed
#          to boot.
#   dflash4low: cohorts of four with the gate's band raised (MEMRA_SPEC_GATE_LOW=4, HIGH=5), an
#          operator setting: 61-B 62-T 63-T 64-B live.
#   bootfull: the draft-state bracket of the boot's calibration, arm D (the head with the deep
#          dumps of diag1307.py --deep), no requests: two boots from a dropped page cache (51-D, 52-D),
#          then six with the page cache filled before each boot (53-D .. 58-D).
#   dflashshort: a cohort of four (the same band) decoding while MemAvailable is taken down to
#          TARGET_MB (41-T, 42-B).
# B = ~/.cache/memra-1307/bins/B (the base: main with memra#1302), T = bins/T (the head); both carry
# the [diag1307] lines. A quiet cell starts from `echo 1 > /proc/sys/vm/drop_caches` (page cache
# only; the kernel's reclaimable cache, nothing else is touched).
#   probe2: as probe without the quiet state, 16 samples of 34, 256 and 1,024 MiB per state.
# usage: [TARGET_MB=n] campaign1307.sh CAMPAIGN [probe] [probe2] [mtp] [mtpboot] [dflash] [dflash4] [bootfull] [dflashagain] [dflash4low] [dflashshort]
set -u
CAMP=$1; shift
H=$(cd "$(dirname "$0")" && pwd)
R=$HOME/.cache/memra-1307
CD=$R/runs/$CAMP; mkdir -p "$CD"
export TMPDIR=$R/tmp; mkdir -p "$TMPDIR"
echo "$$" >> "$R/pids.txt"
drop() { sync; echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null; echo "$(date -Is) drop_caches: $(grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo | tr -s ' ' | tr '\n' ';')" >> "$CD/campaign.log"; }
for part in "$@"; do
  case $part in
    probe)
      P=$CD/probe; mkdir -p "$P"
      drop
      python3 "$H/probe_alloc.py" "$P/probe.jsonl" quiet > "$P/quiet.log" 2>&1
      bash "$H/../../admit-headroom-unified-20261113/tools/fill_cache2.sh" 3000 300 4 > "$P/full-fill.log" 2>&1
      python3 "$H/probe_alloc.py" "$P/probe.jsonl" full > "$P/full.log" 2>&1
      bash "$H/live_fill.sh" 3000 2 600 "$P/stop" "$P/memfree.log" > "$P/live-fill.log" 2>&1
      python3 "$H/probe_alloc.py" "$P/probe.jsonl" live > "$P/live.log" 2>&1
      touch "$P/stop"
      echo "$(date -Is) probe done" >> "$CD/campaign.log" ;;
    probe2)
      # The page cache full (MemFree under 3,000 MB, no reader), then with two readers kept going;
      # 16 samples of 34, 256 and 1,024 MiB each, with every meminfo field the delta could use.
      P=$CD/probe2; mkdir -p "$P"
      bash "$H/../../admit-headroom-unified-20261113/tools/fill_cache2.sh" 3000 600 4 > "$P/full-fill.log" 2>&1
      python3 "$H/probe_alloc.py" "$P/probe.jsonl" full 34,256,1024 16 > "$P/full.log" 2>&1
      bash "$H/live_fill.sh" 3000 2 900 "$P/stop" "$P/memfree.log" > "$P/live-fill.log" 2>&1
      python3 "$H/probe_alloc.py" "$P/probe.jsonl" live 34,256,1024 16 > "$P/live.log" 2>&1
      touch "$P/stop"
      echo "$(date -Is) probe2 done" >> "$CD/campaign.log" ;;
    mtp)
      for c in 01-B 02-T 03-T 04-B; do bash "$H/cell1307.sh" "$CD/mtp" "$c" "$R/bins/${c#*-}" mtp live; done
      drop; bash "$H/cell1307.sh" "$CD/mtp" 05-B "$R/bins/B" mtp quiet
      drop; bash "$H/cell1307.sh" "$CD/mtp" 06-T "$R/bins/T" mtp quiet ;;
    mtpboot)
      for c in 21-B 22-T 23-T 24-B; do bash "$H/cell1307.sh" "$CD/mtp" "$c" "$R/bins/${c#*-}" mtp liveboot; done ;;
    dflash)
      for c in 11-B 12-T 13-T 14-B; do bash "$H/cell1307.sh" "$CD/dflash" "$c" "$R/bins/${c#*-}" dflash live; done
      drop; bash "$H/cell1307.sh" "$CD/dflash" 15-B "$R/bins/B" dflash quiet
      drop; bash "$H/cell1307.sh" "$CD/dflash" 16-T "$R/bins/T" dflash quiet ;;
    dflash4)
      for c in 31-B 32-T 33-T 34-B; do bash "$H/cell1307.sh" "$CD/dflash4" "$c" "$R/bins/${c#*-}" dflash4 live; done
      drop; bash "$H/cell1307.sh" "$CD/dflash4" 35-B "$R/bins/B" dflash4 quiet
      drop; bash "$H/cell1307.sh" "$CD/dflash4" 36-T "$R/bins/T" dflash4 quiet ;;
    bootfull)
      for c in 51-D 52-D; do drop; bash "$H/cell1307.sh" "$CD/bootfull" "$c" "$R/bins/D" mtp bootonly; done
      for c in 53-D 54-D 55-D 56-D 57-D 58-D; do
        bash "$H/../../admit-headroom-unified-20261113/tools/fill_cache2.sh" 3000 300 4 >> "$CD/bootfull-fill.log" 2>&1
        bash "$H/cell1307.sh" "$CD/bootfull" "$c" "$R/bins/D" mtp bootonly
      done ;;
    dflashagain)
      for c in 17-T 18-B; do bash "$H/cell1307.sh" "$CD/dflash" "$c" "$R/bins/${c#*-}" dflash live; done ;;
    dflash4low)
      export EXTRA="MEMRA_SPEC_GATE_LOW=4 MEMRA_SPEC_GATE_HIGH=5"
      for c in 61-B 62-T 63-T 64-B; do bash "$H/cell1307.sh" "$CD/dflash4low" "$c" "$R/bins/${c#*-}" dflash4 live; done
      unset EXTRA ;;
    dflashshort)
      export EXTRA="MEMRA_SPEC_GATE_LOW=4 MEMRA_SPEC_GATE_HIGH=5"
      drop; bash "$H/cell1307.sh" "$CD/dflashshort" 41-T "$R/bins/T" dflash4long shortlate
      drop; bash "$H/cell1307.sh" "$CD/dflashshort" 42-B "$R/bins/B" dflash4long shortlate
      unset EXTRA ;;
  esac
done
echo "$(date -Is) CAMPAIGN-1307-DONE $*" >> "$CD/campaign.log"
