#!/bin/bash
# step37 decode headroom, RE-BASELINED ON THE LANE TIP with the ROWS_TAB_RESTAGE spec fix.
#
# The earlier plan ran MEMRA_SERVE_SPEC=0 because spec-on had an unfixed use-after-free that
# both made the oracle nondeterministic and bricked the process. That fault is fixed
# (7694c049f8), and the fix is NOT in the shared box checkout at all, so every arm here runs a
# binary built from the lane tip in a private worktree. Spec is now the single largest known
# lever toward >90 and it is measured as an arm rather than excluded.
#
# ARMS, all vendor-default SAMPLED, interleaved x5, alternating inside ONE stretch of box life
# (a one-shot A/B is not a claim: box clock drift). Each arm boots its own server because every
# door here is read once at startup.
#   plain   MEMRA_SERVE_SPEC=0                      the re-baseline, 1 MTP head
#   spec    MEMRA_SERVE_SPEC=1 + the serve-config policy knobs (K=3, 3 heads, PMIN)
#   specv   spec + MEMRA_W8_VIEW=1
#   plainv  plain + MEMRA_W8_VIEW=1
# specv-minus-spec and plainv-minus-plain give the W8_VIEW door its own delta in both regimes.
#
# ENGAGEMENT IS PROVED IN BOTH DIRECTIONS, from usage.spec in the RESPONSE BODY. A log-grep
# counter in this lane printed the same value with spec off and proved nothing.
set -u
BIN=/root/s37h-spec-server
# Wait for the lane-tip build OUTSIDE the lock. Waiting inside it would idle the one GPU while
# three other lanes queue. Bounded, and it reports WHY it gave up rather than dying quietly.
for i in $(seq 1 360); do
  grep -q "S37H-BUILD-DONE" /root/s37h-build.txt 2>/dev/null && break
  grep -q "BUILD FAILED\|REFUSED\|WORKTREE FAILED" /root/s37h-build.txt 2>/dev/null && { echo "build reported failure, not queueing" >&2; exit 1; }
  sleep 20
done
exec 9>/root/gemmprime.lock
flock -w 43200 9 || { echo "lock timeout" >&2; exit 1; }
OUT=/root/s37h-spec.txt; : > $OUT
[ -x $BIN ] || { echo "NO BINARY at $BIN - build did not finish - ABORT" >> $OUT; exit 1; }
cd /root/wt-s37h
{
  echo "date=$(date -Is)"
  echo "worktree tip=$(git log -1 --format='%h %s')"
  echo "bin=$BIN md5=$(md5sum $BIN | cut -c1-12)"
  # BINARY FINGERPRINT from strings, never cargo's Finished line.
  echo "strings ROWS_TAB_RESTAGE=$(strings -a $BIN | grep -c ROWS_TAB_RESTAGE) w8-view=$(strings -a $BIN | grep -c w8-view) step37-defaults=$(strings -a $BIN | grep -c step37-defaults)"
  echo "worktree dirty (must be lib.rs only): $(git status --short | tr '\n' ' ')"
} >> $OUT

BASE=$(grep "^ENVV=" /root/agentic8.sh | sed "s/^ENVV=//; s/^\"//; s/\"$//")
COMMON="MEMRA_LOAD_MTP=1 MEMRA_PREFILL_TICK=8192 MEMRA_CTX=262144 MEMRA_PP_BF16=0"
# POLICY knobs, not kernel doors: these are serve-config, exactly as the owner-flip row states.
SPECPOL="MEMRA_SERVE_SPEC=1 MEMRA_SPEC_K=3 MEMRA_MTP_HEADS=3 MEMRA_SPEC_PMIN=0.5 MEMRA_SPEC_PMIN0=1"
PLAINPOL="MEMRA_SERVE_SPEC=0 MEMRA_MTP_HEADS=1"
P=19220

for RND in 1 2 3 4 5; do
 for ARM in plain spec specv plainv; do
  case $ARM in
    plain)  POL="$PLAINPOL"; EXTRA="" ;;
    spec)   POL="$SPECPOL";  EXTRA="" ;;
    specv)  POL="$SPECPOL";  EXTRA="MEMRA_W8_VIEW=1" ;;
    plainv) POL="$PLAINPOL"; EXTRA="MEMRA_W8_VIEW=1" ;;
  esac
  LOG=/root/s37h-spec-$ARM-$RND.log
  env $BASE $COMMON $POL $EXTRA MEMRA_MODELS="step37=/root/models/step37-flash-nvfp4" \
    MEMRA_ADDR=127.0.0.1:$P nohup setsid $BIN > $LOG 2>&1 &
  for i in $(seq 1 600); do
    curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$P/health 2>/dev/null | grep -q 200 && break
    sleep 5
  done
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$P/health | grep -q 200; then
    LOADB=$(cut -d" " -f1 /proc/loadavg)
    BLD=$(pgrep -c -f "cargo|nvcc|cicc" 2>/dev/null || echo 0)
    ARM=$ARM P=$P RND=$RND python3 /root/s37h-spec-probe.py >> $OUT 2>&1
    echo "      vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' '/' | sed 's#/$##') load=$LOADB builders=$BLD w8view_mirrors=$(grep -ac '\[w8-view\] mirror built' $LOG) illegal=$(grep -ac 'ILLEGAL' $LOG)" >> $OUT
  else
    echo "rnd=$RND arm=$ARM booted=NO - CELL INVALID" >> $OUT
    tail -5 $LOG >> $OUT
  fi
  pkill -f "s37h-spec-server" 2>/dev/null; sleep 20
 done
done

python3 /root/s37h-spec-summary.py /root/s37h-spec.txt >> $OUT 2>&1

# ============================ GATES ========================================================
# TWO different bars, because the two candidates are different classes:
#  * SPEC is claimed NUMERICALLY EXACT (verify arbitrates), so its bar is FULL BYTE IDENTITY
#    against spec-off on real prompts, not merely a first token.
#  * MEMRA_W8_VIEW is a WEIGHT-PRECISION door in the MEMRA_STEP_TP_W8 class, so its bar is the
#    run-gen argmax gate: first token identical at max_tokens=1 (one forward, no cascade
#    possible), with the logit maxdiff class stated. Full-generation divergence on a
#    numeric-class door is ordinary tie-break cascade and is NOT a gate result.
echo "=== GATES: byte tape per arm (greedy, four real prompts) ===" >> $OUT
for ARM in plain spec plainv; do
  case $ARM in
    plain)  POL="$PLAINPOL"; EXTRA="" ;;
    spec)   POL="$SPECPOL";  EXTRA="" ;;
    plainv) POL="$PLAINPOL"; EXTRA="MEMRA_W8_VIEW=1" ;;
  esac
  LOG=/root/s37h-specgate-$ARM.log
  env $BASE $COMMON $POL $EXTRA MEMRA_MODELS="step37=/root/models/step37-flash-nvfp4" \
    MEMRA_ADDR=127.0.0.1:$P nohup setsid $BIN > $LOG 2>&1 &
  for i in $(seq 1 600); do curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$P/health 2>/dev/null | grep -q 200 && break; sleep 5; done
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$P/health | grep -q 200; then
    ARM=$ARM P=$P python3 /root/s37h-gate-probe.py > /root/s37h-specgate-$ARM.json 2>>$OUT
    echo "  gate arm=$ARM w8view_mirrors=$(grep -ac '\[w8-view\] mirror built' $LOG) illegal=$(grep -ac ILLEGAL $LOG)" >> $OUT
  else
    echo "  gate arm=$ARM booted=NO - GATE INVALID" >> $OUT
  fi
  pkill -f "s37h-spec-server" 2>/dev/null; sleep 20
done
python3 /root/s37h-gate-pairs.py >> $OUT 2>&1

echo "=== run-gen prime gate: the logit maxdiff CLASS for MEMRA_W8_VIEW ===" >> $OUT
if [ -x /root/s37h-spec-rungen ]; then
  python3 /root/s37h-mkprompts.py >> $OUT 2>&1
  for ARM in 0 1; do
    for N in curve-0400 curve-1000; do
      env $BASE $COMMON $PLAINPOL MEMRA_W8_VIEW=$ARM MEMRA_MAX_CTX=262144 \
        MEMRA_PROMPT_FILE=/root/s37h-$N.prompt MEMRA_NGEN=16 MEMRA_CHAT=1 \
        timeout 2400 /root/s37h-spec-rungen /root/models/step37-flash-nvfp4 \
        > /root/s37h-specprime-$ARM-$N.log 2>&1
      RC=$?
      L=$(grep -a "logit maxdiff" /root/s37h-specprime-$ARM-$N.log | tail -1)
      V=$(grep -ac '\[w8-view\] mirror built' /root/s37h-specprime-$ARM-$N.log)
      echo "  w8view=$ARM $N rc=$RC view_mirrors=$V ${L:-NO-PRIME-GATE-LINE (INVALID)}" >> $OUT
    done
  done
else
  echo "  run-gen was not built; the maxdiff class row is MISSING, not passing" >> $OUT
fi
echo S37H-SPEC-DONE >> $OUT
