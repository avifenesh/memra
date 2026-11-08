#!/usr/bin/env bash
# The #1090 owed PRO 6000 Max-Q run, one card (GPU 3), every cell a fresh boot, three runs per arm
# interleaved A B B A A B (A = main 2288c4bf82, B = #1090 rebased on it, 19316c0049; the code diff is
# byte-identical to the draft head b6686c4144's):
#   m12  spec-cache-mixed shape at MEMRA_MAX_SESSIONS=12 (the finding's box), then one cache-off run of B
#   mdef the same shape at the default caps (the two-entry floor is expected on both arms)
#   c1   agentic c1 at MEMRA_MAX_SESSIONS=12 (where the budget moves; at the default caps both arms hold
#        the floor), T=1, 768 new tokens
#   c4   agentic c4, same settings
# usage: phases-1090b.sh OUT [PHASES]   (b: per-campaign OUT/PAUSE, phase m12x = runs 7-10 of m12, B A A B)
set -u
. <lane>/env.sh
O=$1; PHASES=${2:-m12 mdef c1 c4}
A=$W/bins/memra-server-${MAIN_SHA:0:10}
B=$W/bins/memra-server-${PR_SHA:0:10}
H=$W/arm.sh
mkdir -p "$O"
log() { echo "$(TZ=Asia/Jerusalem date '+%F %T') $*" | tee -a "$O/campaign.log"; }
run4x() { # phase workload env...: arms 7-10 (B A A B), a second balanced block for a five-per-arm phase
  local ph=$1 wl=$2; shift 2
  local i=6 a bin
  for a in B A A B; do
    i=$((i + 1))
    bin=$A; [ $a = B ] && bin=$B
    log "$ph $i $a start"
    bash $H "$ph-$i-$a" "$bin" "$O/$ph" "$wl" "$@"
    log "$ph $i $a rc=$? $(grep -a -m1 'repeat_cached\|cached' "$O/$ph/$ph-$i-$a/result.json" 2>/dev/null | cut -c1-200)"
  done
}
run6() { # phase workload env...
  local ph=$1 wl=$2; shift 2
  local i=0 a bin
  for a in A B B A A B; do
    i=$((i + 1))
    bin=$A; [ $a = B ] && bin=$B
    log "$ph $i $a start"
    bash $H "$ph-$i-$a" "$bin" "$O/$ph" "$wl" "$@"
    log "$ph $i $a rc=$? $(grep -a -m1 'repeat_cached\|cached' "$O/$ph/$ph-$i-$a/result.json" 2>/dev/null | cut -c1-200)"
  done
}
for ph in $PHASES; do
  # A phase boundary is the only place the campaign yields: while $O/PAUSE exists (another scored
  # campaign on the box, e.g. the release perf record), no new phase starts. A phase's six arms never split.
  if [ -e $O/PAUSE ]; then
    log "paused before $ph: $(cat $O/PAUSE)"
    waitfor --until "test ! -e $O/PAUSE" --timeout 6h --every 30s --quiet || { log "pause never lifted"; exit 6; }
    log "resumed before $ph"
  fi
  case $ph in
    m12)  run6 m12 mixed MEMRA_MAX_SESSIONS=12
          log "m12 off start"; bash $H m12-off-B "$B" "$O/m12" mixed MEMRA_MAX_SESSIONS=12 MEMRA_PREFIX_CACHE_MB=0
          log "m12 off rc=$?" ;;
    mdef) run6 mdef mixed ;;
    c1)   run6 c1 agentic:1 MEMRA_MAX_SESSIONS=12 ;;
    c4)   run6 c4 agentic:4 MEMRA_MAX_SESSIONS=12 ;;
    m12x) run4x m12 mixed MEMRA_MAX_SESSIONS=12 ;;
    exact) # agentic greedy exactness at the budget-moving setting: cache off, then #1090 and main with the cache
          for arm in off:$B:MEMRA_PREFIX_CACHE_MB=0 B:$B: A:$A:; do
            IFS=: read -r lab bin extra <<< "$arm"
            log "exact $lab start"; bash $H exact-$lab "$bin" "$O/exact" agentic:1:0:256 MEMRA_MAX_SESSIONS=12 $extra
            log "exact $lab rc=$?"
          done ;;
  esac
done
log CAMPAIGN_DONE
