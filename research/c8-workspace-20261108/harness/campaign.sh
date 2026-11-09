#!/usr/bin/env bash
# The scored campaign, one gpulock lease per arm or boot: item 2's eight fault boots, then item 1's
# interleaved A/B (A = the fix-stack tip, B = + the held workspace charge).
#   S=.. C1=.. C2=.. WT=.. campaign.sh [phase ...]   (default: every phase in order)
set -u
H=<harness>
R=<runs>
phases=${*:-"oom c8 g8 mixed c4 c1"}
for ph in $phases; do
  [ -e <lane>/YIELD ] && { echo "yield before $ph"; break; }
  echo "$(TZ=Asia/Jerusalem date -Is) phase $ph start"
  case $ph in
    oom) OFF_BIN=$C1 ON_BIN=$C2 WT=$WT bash "$H/oom_boots.sh" "$R/oom" O1-off:off O1-on:on O2-on:on O2-off:off \
           P1-off:off-plain P1-on:on-plain P2-on:on-plain P2-off:off-plain ;;
    c8) A=$S B=$C1 bash "$H/phase.sh" c8 "A B B A A B" agentic:8:1.0:768 ;;
    c8rest) A=$S B=$C1 N0=2 bash "$H/phase.sh" c8 "B A A B" agentic:8:1.0:768 ;;
    oomrest) OFF_BIN=$MAINB ON_BIN=$MAINB DOOR=1 WT=$MAINWT bash "$H/oom_boots.sh" "$R/oom-main" O2-off:off \
           P1-off:off-plain P1-on:on-plain P2-on:on-plain P2-off:off-plain ;;
    grow12) for a in 1 2; do AB_ARMS=$a gpulock run --ttl 50m --wait 4h --for "memra c8r2 grow-rewind replay arm $a (S vs C1)" -- \
           bash <growrewind-harness>/grow_ab.sh "$S" "$C1" "$R/grow"; done ;;
    grow34) for a in 3 4; do AB_ARMS=$a gpulock run --ttl 50m --wait 4h --for "memra c8r2 grow-rewind replay arm $a (S vs C1)" -- \
           bash <growrewind-harness>/grow_ab.sh "$S" "$C1" "$R/grow"; done ;;
    g8) A=$S B=$C1 bash "$H/phase.sh" g8 "A B" agentic:8:0:256 ;;
    mixed) A=$S B=$C1 bash "$H/phase.sh" mixed "A B B A A B" mixed MEMRA_MAX_SESSIONS=12 ;;
    c4) A=$S B=$C1 bash "$H/phase.sh" c4 "A B B A A B" agentic:4:1.0:768 ;;
    c1) A=$S B=$C1 bash "$H/phase.sh" c1 "A B" agentic:1:1.0:768 ;;
    *) echo "unknown phase $ph" ;;
  esac
  echo "$(TZ=Asia/Jerusalem date -Is) phase $ph done"
done
echo CAMPAIGN_DONE
