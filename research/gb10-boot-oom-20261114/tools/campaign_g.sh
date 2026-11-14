#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1316): campaign G, the real server's boots.
#   build   arms T (the head) and B (the red arm) from ~/.cache/memra-1316/tree
#   sweep   boot-only cells, arms B and T alternating, 2 s apart: the host as found (g0, 40
#           boots), then the page cache full with 2, 4 and 8 readers live (g2, g4, g8: 80 each)
#   main R N   the page cache full with R readers live, N boots, arms B and T alternating,
#              GAP_MS apart (2000 unless set)
#   hold R N   as main, GAP_MS 1000 unless set, with a HOLD_GIB (24) holder exiting again and again
#   smoke      two full serving cycles of arm T (boot, ready, four requests, stop): a normal boot
# usage: campaign_g.sh JOB PHASE...   (from ~/.cache/memra-1316; ends by creating jobs/JOB.done)
set -u
T=research/gb10-boot-oom-20261114/tools
JOB=$1; shift
while [ $# -gt 0 ]; do
  case $1 in
    build)
      bash $T/build_unit.sh T 9m && bash $T/build_unit.sh B 5m ;;
    sweep)
      NAME=g0-asis READERS=0 N=40 GAP_MS=2000 ARMS="B T" bash $T/boot_cell.sh
      for r in 2 4 8; do NAME=g$r-live$r READERS=$r N=80 GAP_MS=2000 ARMS="B T" bash $T/boot_cell.sh; done ;;
    hold)
      NAME=g-hold-live$2 READERS=$2 N=$3 GAP_MS=${GAP_MS:-1000} ARMS="B T" HOLD_GIB=${HOLD_GIB:-24} bash $T/boot_cell.sh; shift 2 ;;
    smoke)
      FILL_CAP_S=20 NAME=${SMOKE_NAME:-g-smoke} CYCLES=2 ARMS="T" bash $T/serve_cycle.sh ;;
    main)
      NAME=g-main-live$2 READERS=$2 N=$3 GAP_MS=${GAP_MS:-2000} ARMS="B T" bash $T/boot_cell.sh; shift 2 ;;
  esac
  shift
done
touch "jobs/$JOB.done"
